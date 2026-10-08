"""Oreilles natives : Gemma 4 transcrit lui-même."""
import base64
import io
import wave

import numpy as np

from valdar.ears.gemma import GemmaSTT, check, to_wav
from valdar.llm.backend import ChatResult, LLMError
from valdar.llm.fake import FakeBackend

SR = 16000


def test_audio_goes_to_gemma_as_wav():
    llm = FakeBackend([ChatResult(content="« Valdar, quelle heure il est ? »")])
    tr = GemmaSTT(llm).transcribe(np.zeros(SR * 2, np.float32))
    msg = llm.calls[0]["messages"][0]
    raw = base64.b64decode(msg["images"][0])
    with wave.open(io.BytesIO(raw)) as wf:
        assert (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) == (SR, 1, 2)
        assert wf.getnframes() == 2 * SR
    assert llm.calls[0]["tools"] is None
    assert tr.text == "Valdar, quelle heure il est ?"


def test_silence_and_long_audio():
    llm = FakeBackend([ChatResult(content="[silence]")])
    assert GemmaSTT(llm).transcribe(np.zeros(SR, np.float32)).text == ""
    llm = FakeBackend([ChatResult(content="début"), ChatResult(content="fin")])
    tr = GemmaSTT(llm).transcribe(np.zeros(SR * 45, np.float32))
    assert len(llm.calls) == 2, "Gemma prend 30 s au plus par appel"
    assert tr.text == "début fin"


def test_ollama_without_audio_is_reported():
    def boom(*a, **k):
        raise LLMError("Ollama 400 : model does not support audio")

    llm = FakeBackend([])
    llm.chat = boom
    assert "audio" in check(llm)
    stt = GemmaSTT(llm)
    assert stt.transcribe(np.zeros(SR, np.float32)).text == "" and stt.last_error


def test_wav_is_clipped():
    raw = to_wav(np.array([2.0, -2.0], np.float32))
    with wave.open(io.BytesIO(raw)) as wf:
        pcm = np.frombuffer(wf.readframes(2), np.int16)
    assert pcm.tolist() == [32767, -32767]
