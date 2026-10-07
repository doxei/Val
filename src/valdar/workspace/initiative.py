"""Initiative (cahier v2 §8) : quand Valdar a envie de parler ou d'explorer, et les
garde-fous qui l'empêchent d'insister.

- Une envie = somme pondérée de sources du cœur (besoins, drives au-dessus du repos).
- Déclenchement au-dessus du seuil, avec hystérésis (l'envie doit retomber pour réarmer).
- Jamais pendant le sommeil, les heures calmes, le silence demandé ou trop tôt après la
  précédente initiative.
- Après `max_unanswered` messages sans réponse, Valdar se tait jusqu'au prochain message.
- Une initiative restée sans réponse déclenche le stimulus "ignored" (son état en garde la trace).
"""
from __future__ import annotations

from typing import Any

from valdar.config.loader import InitiativeConfig
from valdar.heart.dynamics import clamp, in_window
from valdar.heart.heart import Heart, local_hour


class Initiative:
    def __init__(self, cfg: InitiativeConfig):
        self.cfg = cfg
        self.armed: dict[str, bool] = {k: True for k in cfg.urges}
        self.last_at: float | None = None
        self.unanswered = 0
        self.pending_since: float | None = None
        self.silenced = False

    def urges(self, heart: Heart) -> dict[str, float]:
        src = heart.sources()
        return {kind: clamp(sum(w * src.get(s, 0.0) for s, w in weights.items()))
                for kind, weights in self.cfg.urges.items()}

    def check(self, heart: Heart) -> dict[str, Any] | None:
        """À appeler régulièrement. Renvoie une initiative quand Valdar prend la parole."""
        now = heart.now
        if (self.pending_since is not None
                and now - self.pending_since >= self.cfg.ignored_after_seconds):
            self.pending_since = None
            if self.cfg.ignored_stimulus:
                heart.fire(self.cfg.ignored_stimulus, source="initiative")

        u = self.urges(heart)
        for kind, v in u.items():
            if v < self.cfg.rearm_below:
                self.armed[kind] = True

        if not self._allowed(heart):
            return None
        ready = [k for k, v in u.items() if v >= self.cfg.threshold and self.armed[k]]
        if not ready:
            return None
        kind = max(ready, key=lambda k: u[k])
        self.armed[kind] = False
        self.last_at = now
        self.unanswered += 1
        self.pending_since = now
        event = {"t": now, "kind": kind, "urge": round(u[kind], 3), "emotion": heart.emotion,
                 "felt": heart.felt()}
        heart.journal.log("initiative", t=now, kind=kind, urge=event["urge"],
                          emotion=heart.emotion)
        return event

    def _allowed(self, heart: Heart) -> bool:
        if self.silenced or not heart.awake:
            return False
        if in_window(local_hour(heart.now), self.cfg.quiet_start, self.cfg.quiet_end):
            return False
        if self.unanswered >= self.cfg.max_unanswered:
            return False
        return self.last_at is None or heart.now - self.last_at >= self.cfg.min_interval_seconds

    def on_user_message(self) -> None:
        self.unanswered = 0
        self.pending_since = None

    def silence(self, on: bool = True) -> None:
        self.silenced = on

    def to_dict(self) -> dict[str, Any]:
        return {"armed": self.armed, "last_at": self.last_at, "unanswered": self.unanswered,
                "pending_since": self.pending_since, "silenced": self.silenced}

    def load_dict(self, d: dict[str, Any]) -> None:
        self.armed.update({k: bool(v) for k, v in d.get("armed", {}).items()
                           if k in self.armed})
        self.last_at = d.get("last_at")
        self.unanswered = int(d.get("unanswered", 0))
        self.pending_since = d.get("pending_since")
        self.silenced = bool(d.get("silenced", False))
