"""Double audition (avenant 5 §1) : le bruit brut va au « tronc cérébral », pas au langage.

Deux voies sortent du même micro :
- la voie haute (le portier) : parole → éveil → transcription → Gemma ;
- la voie basse (ici) : le **niveau** du son, trame par trame, sans rien comprendre.

La voie basse ne garde que des nombres (des niveaux en dB), jamais d'audio ni de texte.
Elle agit directement sur le cœur, sans passer par le modèle de langage :
- **sursaut** : un bruit fort et soudain (claquement, cri, chute) → stimulus `startle`.
  C'est un réflexe : il part avant toute compréhension, s'habitue s'il se répète (habituation
  du cœur), et ne part pas pendant que Valdar parle (sa propre voix dans le micro) ;
- **vacarme** : une pièce bruyante longtemps → stimulus `noise`, rappelé de temps en temps ;
- l'ambiance (calme, normale, bruyante) est donnée au Moi dans l'état du monde.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

from valdar.config.loader import AmbientConfig

Fire = Callable[[str, float, str], None]


def db(frame: np.ndarray) -> float:
    """Niveau d'une trame en dB pleine échelle (0 dB = signal maximal)."""
    x = frame.astype(np.float32)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    return 20.0 * math.log10(rms + 1e-6)


class Ambient:
    def __init__(self, cfg: AmbientConfig, frame_seconds: float, fire: Fire,
                 speaking: Callable[[], bool] | None = None,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.dt = frame_seconds
        self.fire = fire
        self.speaking = speaking or (lambda: False)
        self.clock = clock
        self.base: float | None = None      # fond sonore (moyenne lente)
        self.fast: float | None = None      # niveau du moment (moyenne rapide)
        self.loud_since: float | None = None
        self.last_startle = float("-inf")
        self.last_noise = float("-inf")
        self.stats = {"startle": 0, "noise": 0}

    def _ewm(self, prev: float | None, x: float, tau: float) -> float:
        if prev is None:
            return x
        a = 1.0 - math.exp(-self.dt / max(tau, self.dt))
        return prev + a * (x - prev)

    def feed(self, frame: np.ndarray, speech: float = 0.0) -> None:
        """Une trame du micro, avec la probabilité de parole du portier.

        Une voix qui commence près du micro est une attaque, elle aussi : quand le portier
        reconnaît de la parole, seul un cri (`shout_db`) fait sursauter."""
        if not self.cfg.enabled:
            return
        level = db(frame)
        now = self.clock()
        c = self.cfg
        # Un sursaut, c'est une **attaque** : plus fort que le fond ET que l'instant d'avant.
        # Un bruit qui dure ne fait sursauter qu'une fois.
        before = max(self.base if self.base is not None else level,
                     self.fast if self.fast is not None else level)
        floor = c.shout_db if speech >= 0.5 else c.startle_floor_db
        if (level >= floor and level - before >= c.startle_rise_db
                and now - self.last_startle >= c.refractory_seconds and not self.speaking()):
            excess = level - before - c.startle_rise_db
            scale = min(1.0, 0.4 + excess / max(1.0, c.startle_span_db))
            self.last_startle = now
            self.stats["startle"] += 1
            self.fire(c.startle_stimulus, scale, "voie_basse")
        # Le fond suit lentement ; un pic isolé ne le déplace presque pas.
        self.base = self._ewm(self.base, level, c.base_tau_seconds)
        self.fast = self._ewm(self.fast, level, c.fast_tau_seconds)
        if self.base >= c.loud_db:
            if self.loud_since is None:
                self.loud_since = now
            if (now - self.loud_since >= c.loud_seconds
                    and now - self.last_noise >= c.noise_repeat_seconds):
                scale = min(1.0, 0.3 + (self.base - c.loud_db) / 20.0)
                self.last_noise = now
                self.stats["noise"] += 1
                self.fire(c.noise_stimulus, scale, "voie_basse")
        elif self.base < c.loud_db - 3.0:          # petite hystérésis
            self.loud_since = None

    def mood(self) -> str:
        if self.base is None:
            return ""
        if self.base >= self.cfg.loud_db:
            return "bruyante"
        if self.base <= self.cfg.quiet_db:
            return "calme"
        return "normale"

    def line(self) -> str:
        m = self.mood()
        return f"la pièce (à l'oreille) : {m}" if m else ""
