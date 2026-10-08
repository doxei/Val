"""Portier : la pièce n'est jamais transcrite, Valdar n'entend que ce qui lui est adressé."""
import builtins
import pathlib

import numpy as np
import pytest

from valdar.config.loader import EarsConfig
from valdar.ears import EnergyVAD, Gate, Transcript, is_junk, says_name

SR = 16000


def speech(seconds, amp=0.3, f=180.0):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
            ).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), np.float32)


class FakeWake:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def heard(self, seg):
        self.calls += 1
        return self.answers.pop(0) if self.answers else False


class FakeSTT:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def transcribe(self, seg):
        self.calls += 1
        return Transcript(self.texts.pop(0) if self.texts else "", -0.3, 0.1)


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def make(wake, stt, **kw):
    heard, barge = [], []
    clock = Clock()
    g = Gate(EarsConfig(**kw), EnergyVAD(), wake, stt, heard.append,
             on_barge_in=lambda: barge.append(1), speaking=kw.pop("_speaking", None),
             clock=clock)
    return g, heard, barge, clock


def test_room_talk_is_never_transcribed():
    wake, stt = FakeWake([False, False]), FakeSTT([])
    g, heard, _, _ = make(wake, stt)
    for _ in range(2):
        g.feed(speech(1.5))
        g.feed(silence(1.0))
    assert wake.calls == 2 and stt.calls == 0 and heard == []
    assert g.stats["ignored"] == 2


def test_wake_then_engaged_window():
    wake, stt = FakeWake([True]), FakeSTT(["Valdar, quelle heure ?", "et demain ?"])
    g, heard, _, clock = make(wake, stt)
    g.feed(speech(1.5))
    g.feed(silence(1.0))
    assert [h.text for h in heard] == ["Valdar, quelle heure ?"]
    clock.t += 5                         # on lui répond sans redire son nom
    g.feed(speech(1.0))
    g.feed(silence(1.0))
    assert wake.calls == 1 and [h.text for h in heard][-1] == "et demain ?"
    clock.t += 60                        # plus tard : il faut de nouveau l'appeler
    g.feed(speech(1.0))
    g.feed(silence(1.0))
    assert wake.calls == 2


def test_whisper_hallucinations_are_dropped():
    assert is_junk("Sous-titres réalisés par la communauté d'Amara.org")
    assert not is_junk("rappelle-moi de sortir le chien")
    wake, stt = FakeWake([True]), FakeSTT(["Merci d'avoir regardé cette vidéo"])
    g, heard, _, _ = make(wake, stt)
    g.feed(speech(1.0))
    g.feed(silence(1.0))
    assert heard == [] and g.stats["junk"] == 1


def test_barge_in_when_valdar_speaks():
    g, _, barge, _ = make(FakeWake([False]), FakeSTT([]))
    g.speaking = lambda: True
    g.feed(speech(1.0, amp=0.02))     # d'abord l'écho de sa propre voix dans le micro
    assert barge == []
    g.feed(speech(1.0))               # puis quelqu'un lui parle, plus fort que l'écho
    assert barge == [1], "on lui parle pendant qu'il parle : il se tait"


def test_nothing_is_written_to_disk(monkeypatch):
    def forbid(*a, **k):
        raise AssertionError("écriture disque interdite")

    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        if any(c in mode for c in "wax+"):
            forbid()
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(pathlib.Path, "write_bytes", forbid)
    monkeypatch.setattr(pathlib.Path, "write_text", forbid)
    g, heard, _, _ = make(FakeWake([True, False]), FakeSTT(["Valdar, salut"]))
    for _ in range(2):
        g.feed(speech(1.2))
        g.feed(silence(1.0))
    assert len(heard) == 1


@pytest.mark.parametrize("text,ok", [("Valdar tu m'entends", True), ("val dar, salut", True),
                                     ("baldar quelle heure", True), ("Vald'ar !", True),
                                     ("on va au bar", False), ("le radar", False)])
def test_name_detection_tolerates_whisper(text, ok):
    assert says_name(text, ["Valdar"]) is ok
