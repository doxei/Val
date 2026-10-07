"""Yeux de la vigie : caméras, qualité d'image, détecteur Obico (ONNX).

Repris de PrintOS (même prétraitement que le code d'Obico : 416×416 bilinéaire, BGR→RGB,
/255, NCHW ; même post-traitement : max/argmax + NMS par classe à 0,45), avec trois
corrections : seuil 0,08 comme Obico (PrintOS jetait les détections faibles à 0,25), une
image floue n'est **plus** envoyée au détecteur (c'est ce qui a aveuglé PrintOS), et la
qualité de chaque image est mesurée et gardée.

`opencv-python-headless` et `onnxruntime` ne sont nécessaires que sur la machine (extra
`[vision]`) : le reste de Valdar et les tests s'en passent.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from valdar.config.loader import CameraSpec, DetectorSpec, FrameQualitySpec


class VisionError(RuntimeError):
    pass


# ------------------------------------------------------------------------ qualité
@dataclass
class Quality:
    brightness: float
    sharpness: float
    dark: bool
    blurry: bool
    frozen: bool

    @property
    def usable(self) -> bool:
        return not (self.dark or self.blurry or self.frozen)

    def describe(self) -> str:
        bad = [w for w, b in (("trop sombre", self.dark), ("floue", self.blurry),
                              ("figée", self.frozen)) if b]
        return ", ".join(bad) if bad else "nette"


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]          # BGR (OpenCV)
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32)


def _shrink(gray: np.ndarray, width: int = 320) -> np.ndarray:
    h, w = gray.shape
    if w <= width:
        return gray
    step = max(1, w // width)
    return gray[::step, ::step]


def sharpness(gray: np.ndarray) -> float:
    """Variance du laplacien (sur une image réduite), comme PrintOS."""
    g = _shrink(gray)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def digest(img: np.ndarray) -> str:
    g = _shrink(_gray(img), 32).astype(np.uint8)
    return hashlib.md5(g.tobytes()).hexdigest()


class QualityMeter:
    def __init__(self, spec: FrameQualitySpec):
        self.spec = spec
        self._last: str | None = None
        self._same = 0

    def assess(self, img: np.ndarray) -> Quality:
        gray = _gray(img)
        d = digest(img)
        self._same = self._same + 1 if d == self._last else 0
        self._last = d
        b, s = float(gray.mean()), sharpness(gray)
        return Quality(b, s, b < self.spec.dark, s < self.spec.blur,
                       self._same + 1 >= self.spec.frozen_frames)


# ------------------------------------------------------------------------ caméras
class Camera:
    """Instantané HTTP (crowsnest / mjpg-streamer derrière Moonraker)."""

    def __init__(self, spec: CameraSpec, base_url: str, client: Any = None):
        self.spec = spec
        url = spec.snapshot_url
        if not url.startswith(("http://", "https://")):
            url = base_url.rstrip("/") + "/" + url.lstrip("/")
        self.url = url
        self._client = client

    def grab(self, timeout: float = 6.0) -> np.ndarray:
        import httpx

        try:
            if self._client is not None:
                r = self._client.get(self.url, timeout=timeout)
            else:
                r = httpx.get(self.url, timeout=timeout)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise VisionError(f"{self.spec.name} : instantané impossible ({exc})") from exc
        return decode(r.content, self.spec.rotate, self.spec.name)


def decode(data: bytes, rotate: int = 0, name: str = "caméra") -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise VisionError("opencv manquant (installe l'extra [vision])") from exc
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise VisionError(f"{name} : image illisible ({len(data)} octets)")
    if rotate:
        img = np.ascontiguousarray(np.rot90(img, k=-(rotate // 90)))
    return img


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise VisionError("encodage JPEG impossible")
    return buf.tobytes()


# ------------------------------------------------------------------------ détecteur
@dataclass
class Detection:
    confidence: float
    box: tuple[float, float, float, float]      # x1, y1, x2, y2 normalisés (0..1)


class Detector(Protocol):
    def detect(self, img: np.ndarray) -> list[Detection]: ...


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[np.where(iou <= thresh)[0]]
    return keep


def postprocess(outputs: list[np.ndarray], conf_thresh: float, nms_thresh: float
                ) -> list[Detection]:
    """Sorties du modèle Obico : boîtes [1,N,1,4] (coins normalisés) et scores [1,N,C]."""
    if len(outputs) < 2:
        raise VisionError("sorties ONNX inattendues (2 tenseurs attendus)")
    boxes, confs = np.asarray(outputs[0]), np.asarray(outputs[1])
    if boxes.ndim == 4:
        boxes = boxes[:, :, 0, :]
    boxes, confs = boxes[0], confs[0]
    if confs.ndim == 1:
        confs = confs[:, None]
    best, cls = confs.max(axis=1), confs.argmax(axis=1)
    mask = best >= conf_thresh
    boxes, best, cls = boxes[mask], best[mask], cls[mask]
    out: list[Detection] = []
    for c in np.unique(cls):
        sel = cls == c
        for k in nms(boxes[sel], best[sel], nms_thresh):
            b = np.clip(boxes[sel][k], 0.0, 1.0)
            out.append(Detection(float(best[sel][k]), (float(b[0]), float(b[1]),
                                                       float(b[2]), float(b[3]))))
    return out


class ObicoDetector:
    def __init__(self, model_path: str, spec: DetectorSpec):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise VisionError("onnxruntime manquant (installe l'extra [vision])") from exc
        providers = ["CPUExecutionProvider"]
        if spec.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.h = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 416
        self.w = int(inp.shape[3]) if isinstance(inp.shape[3], int) else 416
        self.spec = spec

    def detect(self, img: np.ndarray) -> list[Detection]:
        import cv2

        x = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.expand_dims(np.transpose(x, (2, 0, 1)), 0)
        outputs = self.session.run(None, {self.input_name: x})
        return postprocess(outputs, self.spec.threshold, self.spec.nms)


class ScriptedDetector:
    """Pour les tests et les répétitions : rejoue une liste de confiances par image."""

    def __init__(self, script: list[list[float]]):
        self.script = list(script)

    def detect(self, img: np.ndarray) -> list[Detection]:
        confs = self.script.pop(0) if self.script else []
        return [Detection(c, (0.4, 0.4, 0.5, 0.5)) for c in confs]
