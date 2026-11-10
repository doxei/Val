"""L'œil de la pièce (phase 4) : qui est là, et que fait le chien.

Une image par seconde de la webcam, analysée sur le processeur :
- **objets** (YOLO, ultralytics) : personnes, chien, table ;
- **visages** (`ident/faces.py`) : qui c'est, parmi le foyer.

Ce que Valdar en tire :
- la **présence** : qui il a vu et quand (pour l'état du monde et pour savoir à qui il parle
  quand la voix hésite) ;
- le **chien** : « Sony est dans la pièce » ; et s'il monte sur la table alors que personne
  n'est là, Valdar lui dit de descendre (réflexe, sans passer par Gemma, avec un temps de
  repos entre deux rappels à l'ordre).

Aucune image n'est gardée : chaque image est analysée puis jetée.
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from valdar.config.loader import VisionConfig

PERSON, DOG, TABLE = "person", "dog", "dining table"


@dataclass
class Box:
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(1.0, self.y2 - self.y1)


def on_table(dog: Box, table: Box) -> bool:
    """Le chien est-il SUR la table ? Il la recouvre en largeur, et ses pattes (le bas de sa
    boîte) sont dans le haut de la table, pas en dessous."""
    overlap = max(0.0, min(dog.x2, table.x2) - max(dog.x1, table.x1))
    if overlap < 0.5 * dog.w:
        return False
    return dog.y2 <= table.y1 + 0.45 * table.h and dog.y1 < table.y1


@dataclass
class Scene:
    t: float
    persons: int = 0
    dog: Box | None = None
    dog_on_table: bool = False
    faces: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"t": self.t, "persons": self.persons, "dog": self.dog is not None,
                "dog_on_table": self.dog_on_table, "faces": self.faces}


class YoloDetector:
    def __init__(self, model: str, conf: float):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.conf = conf

    def boxes(self, frame: np.ndarray) -> list[Box]:
        res = self.model.predict(frame, conf=self.conf, verbose=False, device="cpu",
                                 imgsz=640)[0]
        names = res.names
        out = []
        for xyxy, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                              res.boxes.cls.tolist(), strict=False):
            out.append(Box(str(names[int(k)]), float(c), *map(float, xyxy)))
        return out


class Camera:
    """Webcam ou flux réseau (OpenCV), une image à la demande.

    Une caméra absente, débranchée ou prise par un autre programme ne bloque rien : `frame`
    renvoie None, et on retente de l'ouvrir de temps en temps (elle peut revenir)."""

    def __init__(self, source: int | str, retry_seconds: float = 30.0,
                 opener: Callable[[int | str], Any] | None = None,
                 clock: Callable[[], float] = time.time):
        self.source = source
        self.retry_seconds = retry_seconds
        self.clock = clock
        self._opener = opener or self._cv_open
        self.cap: Any = None
        self._misses = 0
        self._next_try = 0.0
        self._open()

    @staticmethod
    def _cv_open(source: int | str) -> Any:
        import cv2

        if isinstance(source, int) and hasattr(cv2, "CAP_DSHOW"):
            return cv2.VideoCapture(source, cv2.CAP_DSHOW)
        return cv2.VideoCapture(source)

    def _open(self) -> bool:
        self._next_try = self.clock() + self.retry_seconds
        try:
            cap = self._opener(self.source)
            if cap is not None and cap.isOpened():
                self.cap, self._misses = cap, 0
                return True
            if cap is not None:
                cap.release()
        except Exception:
            pass
        self.cap = None
        return False

    @property
    def available(self) -> bool:
        return self.cap is not None

    def frame(self) -> np.ndarray | None:
        if self.cap is None and (self.clock() < self._next_try or not self._open()):
            return None
        try:
            ok, img = self.cap.read()
        except Exception:
            ok, img = False, None
        if ok:
            self._misses = 0
            return img
        self._misses += 1
        if self._misses >= 5:            # débranchée : on lâche, on réessaiera
            self.close()
            self._next_try = self.clock() + self.retry_seconds
        return None

    def close(self) -> None:
        if self.cap is not None:
            with contextlib.suppress(Exception):
                self.cap.release()
        self.cap = None


class Watch:
    def __init__(self, cfg: VisionConfig, grab: Callable[[], np.ndarray | None],
                 detector: Any = None, faces: Any = None,
                 on_scene: Callable[[Scene], None] | None = None,
                 on_dog_table: Callable[[], None] | None = None,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.grab = grab
        self.detector = detector
        self.faces = faces
        self.on_scene = on_scene
        self.on_dog_table = on_dog_table
        self.clock = clock
        self.seen: dict[str, float] = {}           # personne → dernière fois vue
        self.dog_seen = 0.0
        self.last: Scene | None = None
        self._table_streak = 0
        self._last_scold = float("-inf")
        self.enroll: tuple[str, int] | None = None  # (personne, photos restantes)
        self.enroll_done: Callable[[str, int], None] | None = None
        self._enrolled = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def step(self) -> Scene | None:
        frame = self.grab()
        if frame is None:
            return None
        now = self.clock()
        scene = Scene(now)
        boxes = self.detector.boxes(frame) if self.detector is not None else []
        scene.persons = sum(1 for b in boxes if b.label == PERSON)
        dogs = [b for b in boxes if b.label == DOG]
        tables = [b for b in boxes if b.label == TABLE]
        if dogs:
            scene.dog = max(dogs, key=lambda b: b.conf)
            self.dog_seen = now
            scene.dog_on_table = any(on_table(scene.dog, t) for t in tables)
        if self.faces is not None:
            if self.enroll is not None:
                person, left = self.enroll
                if self.faces.enroll_shot(person, frame):
                    self._enrolled += 1
                    left -= 1
                self.enroll = (person, left) if left > 0 else None
                if self.enroll is None and self.enroll_done is not None:
                    self.enroll_done(person, self._enrolled)
                    self._enrolled = 0
            for f in self.faces.look(frame):
                scene.faces.append(f.to_dict())
                if f.person:
                    self.seen[f.person] = now
            scene.persons = max(scene.persons, len(scene.faces))
        del frame                                    # rien n'est gardé
        self._dog_rule(scene, now)
        self.last = scene
        if self.on_scene is not None:
            self.on_scene(scene)
        return scene

    def _dog_rule(self, scene: Scene, now: float) -> None:
        if not (self.cfg.table_rule and scene.dog_on_table and scene.persons == 0):
            self._table_streak = 0
            return
        self._table_streak += 1
        if (self._table_streak >= self.cfg.table_frames
                and now - self._last_scold >= self.cfg.table_cooldown_seconds):
            self._last_scold = now
            if self.on_dog_table is not None:
                self.on_dog_table()

    def present(self, now: float | None = None) -> list[str]:
        now = self.clock() if now is None else now
        return [p for p, t in self.seen.items() if now - t <= self.cfg.seen_memory_seconds]

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                t0 = time.time()
                try:
                    self.step()
                    self.error = None
                except Exception as exc:     # l'œil qui plante ne tue pas Valdar
                    self.error = str(exc)
                self._stop.wait(max(0.05, self.cfg.interval_seconds - (time.time() - t0)))

        self._thread = threading.Thread(target=loop, name="valdar-oeil", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
