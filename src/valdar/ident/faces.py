"""Reconnaître les visages (phase 4) : YuNet trouve les visages, SFace en fait une empreinte.

Deux petits modèles ONNX d'OpenCV, sur le processeur. Même principe que la voix : des
vecteurs par personne, comparés au cosinus (seuil recommandé par OpenCV : 0,363). Aucune
image n'est gardée, seulement les vecteurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from valdar.config.loader import IdentConfig
from valdar.ident.store import Prints

KIND = "visage"


@dataclass
class Face:
    box: tuple[int, int, int, int]       # x, y, largeur, hauteur
    score: float
    vec: np.ndarray
    person: str | None = None
    similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"box": list(self.box), "person": self.person,
                "similarity": round(self.similarity, 3)}


class FaceEngine:
    def __init__(self, detector_path: str, model_path: str):
        import cv2

        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(detector_path, "", (320, 320), 0.7, 0.3, 20)
        self.rec = cv2.FaceRecognizerSF.create(model_path, "")
        self._size: tuple[int, int] | None = None

    def faces(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], float,
                                                       np.ndarray]]:
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self.det.setInputSize((w, h))
            self._size = (w, h)
        _, found = self.det.detect(frame)
        out = []
        for row in found if found is not None else []:
            x, y, fw, fh = (int(v) for v in row[:4])
            if min(fw, fh) < 40:          # trop petit, trop loin : empreinte peu fiable
                continue
            aligned = self.rec.alignCrop(frame, row)
            vec = np.asarray(self.rec.feature(aligned), dtype=np.float32).reshape(-1)
            out.append(((x, y, fw, fh), float(row[14]), vec))
        return out


class FaceID:
    def __init__(self, cfg: IdentConfig, prints: Prints, engine: Any):
        self.cfg = cfg
        self.prints = prints
        self.engine = engine

    def look(self, frame: np.ndarray) -> list[Face]:
        out = []
        for box, score, vec in self.engine.faces(frame):
            f = Face(box, score, vec)
            ranked = self.prints.match(KIND, vec)
            if ranked:
                best, sim = ranked[0]
                second = ranked[1][1] if len(ranked) > 1 else 0.0
                f.similarity = sim
                if sim >= self.cfg.face_threshold and sim - second >= 0.05:
                    f.person = best
            out.append(f)
        return out

    def enroll_shot(self, person: str, frame: np.ndarray) -> bool:
        """Une photo d'enrôlement : le plus grand visage de l'image."""
        found = self.engine.faces(frame)
        if not found:
            return False
        box, _, vec = max(found, key=lambda f: f[0][2] * f[0][3])
        self.prints.add(KIND, person, vec, cap=60)
        return True
