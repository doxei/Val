"""La vigie : Valdar surveille les impressions en direct (caméra + télémétrie Klipper).

Par défaut il **observe et alerte** (niveau 0). Il ne met en pause tout seul (niveau 1) que si
la porte des 98 % est ouverte (`evaluation.gate`) **et** qu'Olivier a déverrouillé
l'autonomie lui-même (`valdar vigie --debloquer`). Chaque image laisse une trace (scores,
qualité) : c'est ce qui permet de mesurer, d'apprendre, et de prouver.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from valdar.config.loader import CameraSpec, PrintWatchConfig
from valdar.devices.moonraker import Moonraker, PrinterError
from valdar.printwatch.evaluation import GateReport, PrintRecord, gate
from valdar.printwatch.predict import FailurePredictor, Verdict
from valdar.printwatch.vision import (
    Camera,
    Detector,
    QualityMeter,
    VisionError,
    encode_jpeg,
)

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS prints(job TEXT PRIMARY KEY, file TEXT, started REAL,
        ended REAL, outcome TEXT, failed INTEGER, no_return REAL, labeled_by TEXT)""",
    """CREATE TABLE IF NOT EXISTS frames(id INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT,
        t REAL, camera TEXT, brightness REAL, sharpness REAL, usable INTEGER, score REAL,
        ewm REAL, adjusted REAL, alert INTEGER, pause INTEGER, path TEXT)""",
    """CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT,
        t REAL, kind TEXT, level TEXT, detail TEXT, acted INTEGER DEFAULT 0)""",
    "CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT)",
    "CREATE INDEX IF NOT EXISTS frames_job ON frames(job, t)",
]
ACTIVE = ("printing",)
FINAL = {"complete": False, "cancelled": None, "error": None}


@dataclass
class WatchEvent:
    kind: str            # alerte | pause | pause_faite | camera | telemetrie | fin | question
    text: str
    job: str = ""
    level: float = 0.0   # gravité 0..1, pour le cœur
    image: bytes | None = None
    data: dict[str, Any] = field(default_factory=dict)


class PrintWatch:
    def __init__(self, cfg: PrintWatchConfig, printer: Moonraker, db_path: Path,
                 frames_dir: Path, notify: Callable[[WatchEvent], None],
                 detector_factory: Callable[[], Detector] | None = None,
                 cameras: list[Any] | None = None, clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.printer = printer
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.frames_dir = Path(frames_dir)
        self.notify = notify
        self.clock = clock
        self._detector_factory = detector_factory
        self.detector: Detector | None = None
        self.vision_note = ""
        self._cameras = cameras
        self.meters: dict[str, QualityMeter] = {}
        with self._conn() as con:
            for s in _SCHEMA:
                con.execute(s)
        life = self._get_state("baseline")
        self.predictors: dict[str, FailurePredictor] = {}
        self._baseline = json.loads(life) if life else {}
        self.job: str | None = None
        self._said: set[str] = set()
        self._last_keep: dict[str, float] = {}
        self._temp_bad_since: dict[str, float] = {}
        self._progress = (0.0, 0.0)          # (progression, instant du dernier changement)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_status: dict[str, Any] | None = None

    # ------------------------------------------------------------------ stockage
    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _get_state(self, key: str) -> str | None:
        with self._conn() as con:
            row = con.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _set_state(self, key: str, value: str) -> None:
        with self._conn() as con:
            con.execute("INSERT OR REPLACE INTO state(key,value) VALUES(?,?)", (key, value))

    # ------------------------------------------------------------------ autonomie
    def records(self) -> list[PrintRecord]:
        with self._conn() as con:
            prints = con.execute("SELECT job, started, ended, failed, no_return FROM prints "
                                 "WHERE failed IS NOT NULL AND ended IS NOT NULL").fetchall()
            alerts = con.execute("SELECT job, t, level FROM alerts WHERE kind='vision' "
                                 "OR kind='telemetrie'").fetchall()
        by: dict[str, dict[str, list[float]]] = {}
        for job, t, level in alerts:
            d = by.setdefault(job, {"alerts": [], "pauses": []})
            d["alerts"].append(t)
            if level == "pause":
                d["pauses"].append(t)
        return [PrintRecord(job, s, e, bool(f), nr, by.get(job, {}).get("alerts", []),
                            by.get(job, {}).get("pauses", [])) for job, s, e, f, nr in prints]

    def gate_report(self) -> GateReport:
        a = self.cfg.autonomy
        return gate(self.records(), a.target, a.confidence, a.lead_min_seconds,
                    a.max_false_pause_per_100h, a.min_ok_hours)

    def level(self) -> int:
        """0 = observe et alerte ; 1 = met en pause tout seul."""
        if self.cfg.autonomy.max_level < 1 or self._get_state("unlocked") != "1":
            return 0
        return 1 if self.gate_report().open else 0

    def unlock(self) -> str:
        rep = self.gate_report()
        if not rep.open:
            return "Pas encore : " + "; ".join(rep.reasons)
        self._set_state("unlocked", "1")
        return "Autonomie de niveau 1 déverrouillée : je mettrai en pause tout seul."

    def lock(self) -> str:
        self._set_state("unlocked", "0")
        return "Autonomie reverrouillée : j'observe et je préviens seulement."

    def record_triage(self, job: str, result: dict[str, Any], t: float | None = None) -> None:
        """Garde l'avis de Gemma sur l'image (consultatif : hors porte des 98 %)."""
        with self._conn() as con:
            con.execute("INSERT INTO alerts(job,t,kind,level,detail) VALUES(?,?,?,?,?)",
                        (job, self.clock() if t is None else t, "triage", "avis",
                         json.dumps(result, ensure_ascii=False)))

    # ------------------------------------------------------------------ étiquettes
    def label(self, failed: bool, job: str | None = None, minutes_before_end: float | None = None,
              by: str = "olivier") -> str:
        with self._conn() as con:
            if job is None:
                row = con.execute("SELECT job, ended FROM prints WHERE ended IS NOT NULL "
                                  "ORDER BY ended DESC LIMIT 1").fetchone()
            else:
                row = con.execute("SELECT job, ended FROM prints WHERE job=?", (job,)).fetchone()
            if not row:
                return "aucune impression terminée à étiqueter."
            no_return = None
            if failed and minutes_before_end is not None and row[1]:
                no_return = row[1] - minutes_before_end * 60
            con.execute("UPDATE prints SET failed=?, no_return=?, labeled_by=? WHERE job=?",
                        (int(failed), no_return, by, row[0]))
        return f"noté : « {row[0]} » {'raté' if failed else 'réussi'}."

    # ------------------------------------------------------------------ boucle
    def start(self) -> None:
        if self._thread is not None or not self.cfg.enabled:
            return

        def loop() -> None:
            while not self._stop.is_set():
                t0 = time.time()
                try:
                    self.step()
                except Exception as exc:   # la vigie ne doit jamais tomber
                    self._say_once("err:" + type(exc).__name__, "camera",
                                   f"ma vigie a trébuché : {exc}")
                self._stop.wait(max(1.0, self.cfg.interval_seconds - (time.time() - t0)))

        self._thread = threading.Thread(target=loop, name="valdar-vigie", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def step(self, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        try:
            st = self.printer.status()
        except PrinterError:
            self.last_status = None
            return
        self.last_status = st
        state = st.get("state")
        if state in ACTIVE:
            if self.job is None:
                self._begin(st, now)
            self._watch(st, now)
        elif self.job is not None and state != "paused":   # fin normale, annulation, erreur, perte
            self._end(st, now)

    def _begin(self, st: dict[str, Any], now: float) -> None:
        started = now - float(st.get("duration_s") or 0)
        self.job = f"{st.get('file') or 'impression'}@{int(started)}"
        with self._conn() as con:
            con.execute("INSERT OR IGNORE INTO prints(job,file,started,outcome) VALUES(?,?,?,?)",
                        (self.job, st.get("file"), started, "printing"))
        for p in self.predictors.values():
            p.reset_print()
        self._said = set()
        self._temp_bad_since = {}
        self._progress = (float(st.get("progress") or 0), now)

    def _end(self, st: dict[str, Any], now: float) -> None:
        state = st.get("state") or "inconnu"
        failed = FINAL.get(state)
        with self._conn() as con:
            con.execute("UPDATE prints SET ended=?, outcome=?, failed=COALESCE(failed, ?) "
                        "WHERE job=?", (now, state, failed, self.job))
        self._save_baseline()
        job = self.job
        self.job = None       # l'impression est close quoi qu'il arrive ensuite
        if state == "complete":
            self.notify(WatchEvent("fin", "impression terminée", job, 0.0,
                                   data={"state": state}))
        else:
            self.notify(WatchEvent("question", f"l'impression s'est arrêtée ({state}) : "
                                   "c'était un raté ? Dis-le-moi, ça m'apprend à voir venir.",
                                   job, 0.3, data={"state": state}))
        with contextlib.suppress(OSError):   # image tenue par un autre programme : plus tard
            self.purge(now)

    def _save_baseline(self) -> None:
        """Ligne de base de chaque caméra : on fusionne, une caméra absente cette fois-ci
        garde la sienne (sinon 7 200 images d'apprentissage seraient perdues)."""
        base = dict(self._baseline)
        base.update({k: p.lifetime_state() for k, p in self.predictors.items()})
        self._baseline = base
        self._set_state("baseline", json.dumps(base))

    def purge(self, now: float | None = None) -> int:
        """Efface les images gardées depuis plus de `keep_days` (les scores restent : ils
        servent à prouver la porte des 98 %). Renvoie le nombre d'images effacées."""
        now = self.clock() if now is None else now
        limit = now - self.cfg.keep_days * 86400
        with self._conn() as con:
            rows = con.execute("SELECT id, path FROM frames WHERE path IS NOT NULL AND t < ?",
                               (limit,)).fetchall()
            done = []
            for fid, p in rows:
                try:
                    Path(p).unlink(missing_ok=True)
                    done.append((fid,))
                except OSError:
                    continue      # fichier verrouillé (Windows) : on réessaiera
            con.executemany("UPDATE frames SET path=NULL WHERE id=?", done)
        for d in self.frames_dir.glob("*"):
            if d.is_dir() and not any(d.iterdir()):
                with contextlib.suppress(OSError):
                    d.rmdir()
        return len(done)

    # ------------------------------------------------------------------ vision
    def cameras(self) -> list[Any]:
        if self._cameras is None:
            specs = list(self.cfg.cameras)
            if not specs and self.cfg.discover_webcams:
                specs = [CameraSpec(**c) for c in self.printer.webcams()]
            self._cameras = [Camera(s, self.printer.cfg.moonraker_url) for s in specs
                             if s.enabled]
        return self._cameras

    def _detector(self) -> Detector | None:
        if self.detector is None and self._detector_factory is not None:
            try:
                self.detector = self._detector_factory()
                self.vision_note = ""
            except Exception as exc:
                self.vision_note = str(exc)
                self._detector_factory = None
        return self.detector

    def _watch(self, st: dict[str, Any], now: float) -> None:
        self._telemetry(st, now)
        cams = self.cameras()
        if not cams:
            self._say_once("nocam", "camera", "aucune caméra n'est branchée : je surveille "
                           "l'impression à l'aveugle, juste avec les températures.")
            return
        det = self._detector()
        for cam in cams:
            name = cam.spec.name
            try:
                img = cam.grab()
            except VisionError as exc:
                self._say_once(f"grab:{name}", "camera", f"je ne vois rien par {name} : {exc}")
                continue
            q = self.meters.setdefault(name, QualityMeter(self.cfg.quality)).assess(img)
            verdict: Verdict | None = None
            if q.usable and det is not None:
                confs = [d.confidence for d in det.detect(img)]
                pred = self.predictors.get(name)
                if pred is None:
                    pred = FailurePredictor(self.cfg.predictor, self._baseline.get(name))
                    self.predictors[name] = pred
                verdict = pred.push(FailurePredictor.frame_score(
                    confs, self.cfg.detector.threshold))
            elif not q.usable:
                self._say_once(f"q:{name}", "camera", f"l'image de {name} est {q.describe()} : "
                               "je ne m'y fie pas tant que ça ne s'arrange pas.")
            path = self._keep(name, img, now, force=bool(verdict and verdict.alert))
            self._log_frame(name, now, q, verdict, path)
            if verdict is not None and (verdict.alert or verdict.pause):
                self._alarm(name, verdict, now, img)

    def _keep(self, cam: str, img: Any, now: float, force: bool) -> str | None:
        every = self.cfg.keep_frame_every_seconds
        if not force and (not every or now - self._last_keep.get(cam, 0) < every):
            return None
        self._last_keep[cam] = now
        try:
            data = encode_jpeg(img)
        except Exception:
            return None
        d = self.frames_dir / (self.job or "hors_impression").replace("/", "_").replace(":", "_")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{int(now)}_{cam}.jpg"
        path.write_bytes(data)
        return str(path)

    def _log_frame(self, cam: str, now: float, q: Any, v: Verdict | None,
                   path: str | None) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO frames(job,t,camera,brightness,sharpness,usable,score,ewm,adjusted,"
                "alert,pause,path) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.job, now, cam, q.brightness, q.sharpness, int(q.usable),
                 v.score if v else None, v.ewm if v else None, v.adjusted if v else None,
                 int(bool(v and v.alert)), int(bool(v and v.pause)), path))

    def _alarm(self, cam: str, v: Verdict, now: float, img: Any) -> None:
        level = "pause" if v.pause else "alert"
        acted = False
        if v.pause and self.level() >= 1 and f"paused:{self.job}" not in self._said:
            try:
                self.printer.pause()
                acted = True
                self._said.add(f"paused:{self.job}")
            except PrinterError:
                acted = False
        with self._conn() as con:
            con.execute("INSERT INTO alerts(job,t,kind,level,detail,acted) VALUES(?,?,?,?,?,?)",
                        (self.job, now, "vision", level,
                         json.dumps({"camera": cam, "score": v.score, "ewm": v.ewm,
                                     "adjusted": v.adjusted}), int(acted)))
        key = f"{level}:{self.job}"
        if key in self._said and not acted:
            return
        self._said.add(key)
        try:
            jpeg = encode_jpeg(img)
        except Exception:
            jpeg = None
        if acted:
            text = "j'ai mis l'impression en pause : ça ressemble à un raté (vu par " + cam + ")."
            kind = "pause_faite"
        elif v.pause:
            text = ("ça part en vrille sur " + cam + " : à ta place je mettrais en pause "
                    "(je ne le fais pas tout seul tant que je n'ai pas fait mes preuves).")
            kind = "pause"
        else:
            text = "quelque chose me chiffonne sur " + cam + ", je surveille de près."
            kind = "alerte"
        self.notify(WatchEvent(kind, text, self.job or "", 0.8 if v.pause else 0.5, jpeg,
                               {"camera": cam, "adjusted": v.adjusted}))

    # ------------------------------------------------------------------ télémétrie
    def _telemetry(self, st: dict[str, Any], now: float) -> None:
        t = self.cfg.telemetry
        for name in ("nozzle", "bed"):
            cur, target = st.get(name, (0, 0))
            if not target:
                self._temp_bad_since.pop(name, None)
                continue
            if abs(cur - target) > t.temp_tolerance and (st.get("duration_s") or 0) > 120:
                since = self._temp_bad_since.setdefault(name, now)
                if now - since >= t.temp_persist_seconds:
                    label = "la buse" if name == "nozzle" else "le plateau"
                    self._telemetry_alert(f"temp:{name}", now,
                                          f"{label} n'est plus à sa consigne ({cur:.0f} °C pour "
                                          f"{target:.0f}) depuis {now - since:.0f} s", 0.7)
            else:
                self._temp_bad_since.pop(name, None)
        prog = float(st.get("progress") or 0)
        if prog != self._progress[0]:
            self._progress = (prog, now)
        elif now - self._progress[1] > t.stall_seconds:
            self._telemetry_alert("stall", now, "la progression ne bouge plus depuis "
                                  f"{(now - self._progress[1]) / 60:.0f} min", 0.4)

    def _telemetry_alert(self, key: str, now: float, text: str, level: float) -> None:
        with self._conn() as con:
            con.execute("INSERT INTO alerts(job,t,kind,level,detail) VALUES(?,?,?,?,?)",
                        (self.job, now, "telemetrie", "alert", text))
        self._say_once(f"tele:{key}", "telemetrie", text, level)

    def _say_once(self, key: str, kind: str, text: str, level: float = 0.2) -> None:
        if key in self._said:
            return
        self._said.add(key)
        self.notify(WatchEvent(kind, text, self.job or "", level))

    # ------------------------------------------------------------------ résumé
    def status_text(self) -> str:
        lines = []
        cams = self._cameras if self._cameras is not None else None
        if cams is None:
            lines.append("caméras : pas encore cherchées (au prochain tour de vigie)")
        elif not cams:
            lines.append("caméras : aucune branchée")
        else:
            lines.append("caméras : " + ", ".join(c.spec.name for c in cams))
        if self.vision_note:
            lines.append(f"détecteur : indisponible ({self.vision_note})")
        lines.append(f"impression suivie : {self.job or 'aucune'}")
        with self._conn() as con:
            n = con.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        lines.append(f"images analysées depuis le début : {n}")
        lines.append(self.gate_report().text())
        lines.append(f"niveau d'autonomie actuel : {self.level()} "
                     "(0 = j'observe et je préviens, 1 = je mets en pause tout seul)")
        return "\n".join(lines)
