"""Avenant 5 : double audition (voie basse) et charge cognitive."""
from __future__ import annotations

import numpy as np

from valdar.config import load
from valdar.config.loader import AmbientConfig, InteroceptionConfig
from valdar.ears.ambient import Ambient, db
from valdar.ears.gate import SR, EnergyVAD, Gate, Transcript
from valdar.heart.interoception import Load, TimedLLM
from valdar.llm.backend import ChatResult

FRAME = 512
DT = FRAME / SR


def tone(amp: float, n: int = FRAME) -> np.ndarray:
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def run(amb: Ambient, clock: Clock, amp: float, seconds: float) -> None:
    for _ in range(int(seconds / DT)):
        amb.feed(tone(amp))
        clock.t += DT


def make(**kw):
    fired: list[tuple[str, float, str]] = []
    clock = Clock()
    amb = Ambient(AmbientConfig(**kw), DT, lambda n, s, src: fired.append((n, s, src)),
                  clock=clock)
    return amb, clock, fired


def test_db_scale():
    assert -4 < db(tone(1.0)) < -2          # sinus pleine échelle ≈ −3 dB
    assert db(np.zeros(FRAME, np.float32)) < -100


def test_bang_startles_once_then_habituates_to_lasting_noise():
    amb, clock, fired = make()
    run(amb, clock, 0.002, 10)               # pièce calme (≈ −57 dB)
    assert amb.mood() == "calme" and not fired
    run(amb, clock, 0.5, 10)                 # un vacarme soudain qui dure 10 s
    startles = [f for f in fired if f[0] == "startle"]
    assert len(startles) == 1                # une attaque, un sursaut : pas un par seconde
    assert startles[0][2] == "voie_basse"


def test_normal_speech_does_not_startle():
    amb, clock, fired = make()
    run(amb, clock, 0.002, 10)
    run(amb, clock, 0.04, 3)                 # voix posée près du micro (≈ −31 dB)
    assert not fired


def test_no_startle_while_valdar_speaks():
    fired: list = []
    clock = Clock()
    amb = Ambient(AmbientConfig(), DT, lambda *a: fired.append(a), speaking=lambda: True,
                  clock=clock)
    run(amb, clock, 0.002, 5)
    run(amb, clock, 0.5, 1)
    assert not fired


def test_lasting_loud_room_fires_noise_with_repeat():
    amb, clock, fired = make(loud_seconds=20, noise_repeat_seconds=60, base_tau_seconds=5)
    run(amb, clock, 0.1, 150)                # pièce bruyante 2 min 30 (≈ −23 dB)
    noise = [f for f in fired if f[0] == "noise"]
    assert 2 <= len(noise) <= 3
    assert amb.mood() == "bruyante"


def test_ambient_keeps_only_numbers():
    amb, clock, _ = make()
    run(amb, clock, 0.1, 2)
    kept = [v for v in vars(amb).values() if isinstance(v, np.ndarray | bytes | list)]
    assert kept == []                        # aucun audio gardé


def test_gate_forwards_frames_to_low_road():
    seen: list[int] = []

    class NoWake:
        def heard(self, seg):
            return False

    class NoSTT:
        def transcribe(self, seg):
            return Transcript("")

    cfg = load().ears
    g = Gate(cfg, EnergyVAD(), NoWake(), NoSTT(), lambda h: None,
             on_frame=lambda f, p: seen.append(len(f)))
    g.feed(np.zeros(g.frame * 4, np.float32))
    assert seen == [g.frame] * 4


# ---------------------------------------------------------------- charge
def test_load_levels_with_hysteresis():
    ld = Load(InteroceptionConfig(weights={"flux": 1.0}, hysteresis=0.15))
    assert ld.update(0, None) == 0
    assert ld.update(2, None) == 1           # 2/3 ≥ 0.6 : chargé
    assert ld.update(3, None) == 2           # saturé
    ld.update(2, None)
    assert ld.level == 2                     # 0.67 ≥ 0.8 − 0.15 : reste saturé (hystérésis)
    assert ld.update(1, None) == 0
    assert ld.line() == ""


def test_slowness_from_latency():
    ld = Load(InteroceptionConfig())
    for _ in range(20):
        ld.record_latency(2.0, 100)          # 20 ms par jeton, habituel
    assert ld.slowness() == 0.0
    for _ in range(6):
        ld.record_latency(8.0, 100)          # 4 fois plus lent
    assert ld.slowness() > 0.8


def test_timed_llm_records_and_delegates():
    class Inner:
        model = "gemma"

        def chat(self, messages, system="", tools=None, params=None):
            return ChatResult("ok", usage={"eval_count": 10})

        def available(self):
            return True

    ld = Load(InteroceptionConfig())
    t = iter([0.0, 1.0])
    llm = TimedLLM(Inner(), ld, clock=lambda: next(t))
    assert llm.chat([]).content == "ok"
    assert llm.model == "gemma" and llm.available()
    assert list(ld._lat) == [0.1]


def test_runtime_saturation_stops_background_thought(runtime_factory):
    rt = runtime_factory()
    rt.load.level = 1
    called = []
    rt.thoughts.due = lambda *a: called.append(1) or True   # type: ignore[method-assign]
    rt._maybe_think(1e9)
    assert called == []
    amb = rt.make_ambient(DT)
    for _ in range(int(5 / DT)):
        amb.feed(tone(0.002))
    for _ in range(6):
        amb.feed(tone(0.9))
    kinds = []
    while not rt.events.empty():
        kinds.append(rt.events.get().kind)
    assert "sursaut" in kinds
    assert any("la pièce" in line for line in rt.world_lines())


def test_gate_keeps_leftover_samples_between_blocks():
    seen: list[int] = []

    class NoWake:
        def heard(self, seg):
            return False

    class NoSTT:
        def transcribe(self, seg):
            return Transcript("")

    g = Gate(load().ears, EnergyVAD(), NoWake(), NoSTT(), lambda h: None,
             on_frame=lambda f, p: seen.append(len(f)))
    for _ in range(10):
        g.feed(np.zeros(1600, np.float32))      # blocs de 100 ms du micro
    assert len(seen) == 16000 // g.frame        # 31 trames : aucun échantillon perdu


def test_voice_onset_does_not_startle_but_a_shout_does():
    amb, clock, fired = make()
    for _ in range(int(5 / DT)):
        amb.feed(tone(0.002), 0.0)
        clock.t += DT
    amb.feed(tone(0.2), 0.1)                    # début de phrase près du micro (≈ −17 dB) :
    for _ in range(6):                          # le détecteur de parole ne la reconnaît
        amb.feed(tone(0.2), 0.9)                # qu'au bout de quelques trames
    assert not fired
    clock.t += 10
    for _ in range(6):
        amb.feed(tone(0.002), 0.0)
    for _ in range(6):
        amb.feed(tone(0.9), 0.9)                # un cri (≈ −4 dB)
    assert [f[0] for f in fired] == ["startle"]


def test_wake_name_survives_bad_transcription():
    from valdar.ears.models import says_name

    names = ["Valdar", "Valdare", "Val dar"]
    for heard in ("Baldar tu m'entends", "Val d'arc, tu m'entends ?", "Valdard", "Wall dar"):
        assert says_name(heard, names), heard
    for other in ("valeur", "voilà", "valider le truc", "val de marne", "valable"):
        assert not says_name(other, names), other


def test_resampler_is_continuous_across_blocks():
    from valdar.ears.mic import Resampler

    r = Resampler(48000)
    t = np.arange(48000) / 48000
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    out = np.concatenate([r(x[i:i + 4800]) for i in range(0, len(x), 4800)])
    assert len(out) == 16000
    step = 2 * np.pi * 440 / 16000
    assert np.abs(np.diff(out[2000:])).max() < step * 1.05    # pas de clic entre les blocs
