"""Voix : la chaîne d'effets historique à l'identique, synthèse XTTS (simulée), lecteur, import
des fichiers."""
import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly, sosfilt

from valdar.config import load
from valdar.config.loader import VoiceConfig
from valdar.migrate import AncienImport
from valdar.voice import (
    FakeTTS,
    NullSink,
    Speaker,
    TTSError,
    XttsBackend,
    apply,
    speakable,
    split_sentences,
)


# ---------------------------------------------------------------------------------------------
# Référence : code d'origine de la chaîne d'effets de la voix de Valdar (`io/voice.py`, profil
# MEGATRON), recopié tel quel.
def _ref_band_peaking(x, sr, fc, gain_db, q):
    w0 = 2.0 * np.pi * fc / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    b = np.array([1.0 + alpha * A, -2.0 * np.cos(w0), 1.0 - alpha * A])
    a = np.array([1.0 + alpha / A, -2.0 * np.cos(w0), 1.0 - alpha / A])
    sos = np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])
    return sosfilt(sos, x)


def _ref_high_shelf(x, sr, fc, gain_db):
    A = np.sqrt(10.0 ** (gain_db / 20.0))
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / np.sqrt(2.0)
    c = np.cos(w0)
    b = A * np.array([
        (A + 1) + (A - 1) * c + 2 * np.sqrt(A) * alpha,
        -2 * ((A - 1) + (A + 1) * c),
        (A + 1) + (A - 1) * c - 2 * np.sqrt(A) * alpha,
    ])
    a = np.array([
        (A + 1) - (A - 1) * c + 2 * np.sqrt(A) * alpha,
        2 * ((A - 1) - (A + 1) * c),
        (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * alpha,
    ])
    sos = np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])
    return sosfilt(sos, x)


def _ref_pitch_shift(x, semis):
    f = 2.0 ** (semis / 12.0)
    n = len(x)
    if abs(f - 1.0) < 1e-3 or n < 16:
        return x
    up, down = f, 1.0
    if f < 1.0:
        up, down = 1.0, 1.0 / f
    g = pow(10.0, 3.0 / 20.0)
    x2 = resample_poly(x, int(round(up * g)), max(1, int(round(down * g))))
    return resample_poly(x2, n, len(x2)) if len(x2) else x


class _Megatron:   # VoiceProfile MEGATRON d'origine (champs utilisés par _character)
    vibrato_hz, vibrato_depth, wobble_hz, wobble_depth = 1.2, 0.04, 0.0, 0.0
    nasal_gain, pitch_shift, echo_ms, echo_gain, lo_fi_bits = 0.0, -2.0, 0.0, 0.0, 15


def _ref_character(audio, sr, profile=_Megatron):
    x = audio.astype(np.float64) / 32768.0
    t = np.arange(len(x)) / sr
    if profile.vibrato_hz and profile.vibrato_depth:
        x = x * (1.0 + 0.02 * np.sin(2 * np.pi * profile.vibrato_hz * t))
    if profile.wobble_hz and profile.wobble_depth:
        x = x * (1.0 + profile.wobble_depth * np.sin(2 * np.pi * profile.wobble_hz * t))
    if profile.nasal_gain:
        x = _ref_band_peaking(x, sr, 1700.0, profile.nasal_gain, 1.0)
    if profile.pitch_shift:
        x = _ref_pitch_shift(x, profile.pitch_shift)
    if sr >= 800:
        x = _ref_high_shelf(x, sr, 400.0, -10.0)
    if profile.lo_fi_bits < 16:
        levels = 2 ** profile.lo_fi_bits
        x = np.round(x * (levels - 1)) / max(1, levels - 1)
    x = np.tanh(x * 1.4)
    peak = np.max(np.abs(x)) or 1.0
    x = x * (0.89 / peak)
    return np.clip(x, -1.0, 1.0)


def _speechlike(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 24000
    x = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) / (k + 1)
            for k, f in enumerate((140, 280, 700, 1500, 3200, 7500, 10000)))
    x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.05 * rng.standard_normal(n)
    return (x / np.max(np.abs(x)) * 20000).astype(np.int16)


@pytest.mark.parametrize("n", [24000, 36001, 10, 4801])
def test_character_is_bit_exact_with_original(n):
    cfg = load()
    audio = _speechlike(n, seed=n)
    ours = apply(audio, 24000, cfg.voice.character)
    ref = _ref_character(audio, 24000)
    assert ours.shape == ref.shape
    assert np.array_equal(ours, ref), "la voix doit sortir exactement comme à l'origine"


def test_original_pitch_setting_is_really_a_lowpass():
    """Documenté dans l'avenant 3 : « pitch −2 » ne change pas la hauteur, il coupe au-dessus
    de fs/4. On le garde, c'est le timbre de la voix."""
    from valdar.voice.character import _resample_tone

    sr = 24000
    t = np.arange(sr) / sr

    def through(f0):
        x = np.sin(2 * np.pi * f0 * t)
        y = _resample_tone(x, -2.0)
        peak_hz = np.argmax(np.abs(np.fft.rfft(y * np.hanning(sr)))) * sr / len(y)
        return peak_hz, np.std(y[2000:-2000]) / np.std(x[2000:-2000])

    for f0 in (220, 1000, 4000):
        hz, gain = through(f0)
        assert abs(hz - f0) < 2, "aucun changement de hauteur"
        assert gain > 0.99
    assert through(9000)[1] < 0.01, "coupé au-dessus de 6 kHz"


def test_speakable_and_sentences():
    assert speakable("**Salut** [rire] Olivier 😄 !") == "Salut Olivier !"
    long = "Alors voilà, " + "on parle de la buse, du plateau et de la vis, " * 12 + "fin."
    parts = split_sentences("Première phrase. Deuxième ? " + long, max_chars=120)
    assert parts[:2] == ["Première phrase.", "Deuxième ?"]
    assert all(len(p) <= 121 for p in parts)
    assert " ".join(parts).replace("  ", " ").startswith("Première phrase. Deuxième ? Alors")
    assert "".join(parts).replace(" ", "") == ("Première phrase. Deuxième ? " + long).replace(
        " ", "")


# ------------------------------------------------------------------------------- XTTS simulé
class _Conf(dict):
    pass


class _FakeXtts:
    def __init__(self, broken_inference=False):
        self.config = _Conf(max_ref_len=10, gpt_cond_len=12, gpt_cond_chunk_len=4,
                            sound_norm_refs=False, temperature=0.75, length_penalty=1.0,
                            repetition_penalty=5.0, top_k=50, top_p=0.85)
        self.latent_calls = []
        self.infer_calls = []
        self.synth_calls = []
        self.broken = broken_inference

    def get_conditioning_latents(self, **kw):
        self.latent_calls.append(kw)
        return "LAT", "EMB"

    def inference(self, text, language, lat, emb, **kw):
        if self.broken:
            raise TypeError("ancienne API")
        self.infer_calls.append((text, language, lat, emb, kw))
        return {"wav": np.array([0.0, 0.25, -0.5], dtype=np.float32)}

    def synthesize(self, text, cfg, speaker_wav=None, language=None):
        self.synth_calls.append((text, speaker_wav, language))
        return {"wav": np.array([0.1, -0.2])}


def _voice_files(tmp_path: Path) -> tuple[Path, Path]:
    mdir = tmp_path / "xtts-v2"
    mdir.mkdir()
    (mdir / "config.json").write_text("{}")
    (mdir / "model.pth").write_bytes(b"x")
    ref = tmp_path / "xtts_ref.wav"
    ref.write_bytes(b"RIFF")
    return mdir, ref


def test_xtts_latents_once_and_original_settings(tmp_path):
    mdir, ref = _voice_files(tmp_path)
    model = _FakeXtts()
    be = XttsBackend(mdir, ref, loader=lambda d, dev: (model, model.config))
    a = be.synthesize("Salut.")
    be.synthesize("Ça roule ?")
    assert len(model.latent_calls) == 1, "la voix n'est clonée qu'une fois"
    assert model.latent_calls[0] == {"audio_path": str(ref), "max_ref_length": 10,
                                     "gpt_cond_len": 12, "gpt_cond_chunk_len": 4,
                                     "sound_norm_refs": False}
    assert model.infer_calls[0][4] == {"temperature": 0.75, "length_penalty": 1.0,
                                       "repetition_penalty": 5.0, "top_k": 50, "top_p": 0.85}
    assert a.dtype == np.int16 and a.tolist() == [0, 16383, -32767]
    assert be.mode == "latents calculés une fois"


def test_xtts_falls_back_to_original_exact_call(tmp_path):
    mdir, ref = _voice_files(tmp_path)
    model = _FakeXtts(broken_inference=True)
    be = XttsBackend(mdir, ref, loader=lambda d, dev: (model, model.config))
    be.synthesize("Test.")
    assert model.synth_calls == [("Test.", str(ref), "fr")]
    assert be.mode == "appel d'origine à l'identique"


def test_xtts_missing_files_says_what_to_do(tmp_path):
    be = XttsBackend(tmp_path / "rien", tmp_path / "rien.wav", loader=lambda d, dev: 1 / 0)
    with pytest.raises(TTSError, match="valdar_voix.bat"):
        be.load()


# ------------------------------------------------------------------------------- lecteur
def test_speaker_plays_sentence_by_sentence():
    sink = NullSink()
    tts = FakeTTS()
    sp = Speaker(VoiceConfig(streaming=False), tts, sink=sink).start()
    sp.say("Salut Olivier. *Ça* va ? Moi oui.")
    assert sp.wait_done(timeout=10)
    sp.close()
    assert tts.texts == ["Salut Olivier.", "Ça va ?", "Moi oui."]
    assert len(sink.played) == 3 and sink.played[0][1] == 24000
    assert sp.stats.first_audio_seconds is not None and not sp.stats.errors


def test_speaker_interrupt_and_broken_voice_never_hang():
    class SlowSink(NullSink):
        def play(self, audio, sr):
            time.sleep(0.2)
            super().play(audio, sr)

    sink = SlowSink()
    sp = Speaker(VoiceConfig(), FakeTTS(), sink=sink).start()
    sp.say("Un. Deux. Trois. Quatre. Cinq. Six.")
    time.sleep(0.3)
    sp.interrupt()
    assert sp.wait_done(timeout=5)
    sp.close()
    assert len(sink.played) < 6

    broken = Speaker(VoiceConfig(), FakeTTS(fail=True), sink=NullSink()).start()
    broken.ready.wait(5)
    assert broken.load_error
    broken.say("Bonjour.")
    assert broken.wait_done(timeout=5)
    broken.close()
    assert broken.stats.errors


# ------------------------------------------------------------------------------- import
def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_import_takes_voice_and_nothing_personal(runtime_factory, tmp_path):
    ancien = tmp_path / "ancien"
    vdir = ancien / "data" / "voice"
    mdir = ancien / "data" / "models" / "xtts-v2"
    vdir.mkdir(parents=True)
    mdir.mkdir(parents=True)
    (ancien / "data" / "eleven.key").write_text("secret")
    for name in ("xtts_ref.wav", "ref_eleven_1.wav", "cal_Olivier_0.wav", "test_sony.wav"):
        (vdir / name).write_bytes(name.encode() * 50)
    rt = runtime_factory()
    data_dir = rt.cfg.storage_path("x").parent
    ref, xdir = data_dir / "voice" / "xtts_ref.wav", data_dir / "models" / "xtts-v2"

    def importer():
        return AncienImport(ancien, rt.facts, rt.stock, rt.reminders, rt.checklist,
                            rt.cfg.storage_path(rt.cfg.atelier.pinouts), data_dir,
                            voice_ref=ref, xtts_dir=xdir)

    first = importer().run()
    assert any(line.startswith("voix : voix introuvable dans l'ancienne installation")
               for line in first)
    for name in ("config.json", "model.pth", "dvae.pth", "mel_stats.pth", "vocab.json",
                 "speakers_xtts.pth"):
        (mdir / name).write_bytes(name.encode() * 100)
    before = _tree_hash(ancien)
    steps = []
    report = importer().run(on_step=steps.append)
    assert steps == ["voix"], "seule la voix restait à faire"
    assert any("Valdar parlera avec sa voix" in line for line in report)
    assert ref.read_bytes() == (vdir / "xtts_ref.wav").read_bytes()
    assert (ref.parent / "ref_eleven_1.wav").is_file()
    assert (xdir / "model.pth").read_bytes() == (mdir / "model.pth").read_bytes()
    copied = {p.name for p in data_dir.rglob("*") if p.is_file()}
    assert "cal_Olivier_0.wav" not in copied, "jamais les enregistrements d'une personne"
    assert "eleven.key" not in copied and "test_sony.wav" not in copied
    assert _tree_hash(ancien) == before, "l'ancienne installation n'est jamais modifiée"
    assert all("déjà importé" in line for line in importer().run())



def test_speaker_streams_chunks_with_one_continuous_chain():
    sink = NullSink()
    tts = FakeTTS()
    sp = Speaker(VoiceConfig(), tts, sink=sink).start()
    sp.say("Salut Olivier. Ça va ?")
    assert sp.wait_done(timeout=10)
    sp.close()
    assert len(sink.played) == 6                      # 3 morceaux par phrase, à la suite
    assert sum(n for n, _ in sink.played) == sum(max(240, 60 * len(t)) for t in tts.texts)
    assert sp.stats.first_chunk_seconds is not None


def test_character_stream_matches_the_original_chain():
    from valdar.voice.character import CharacterStream

    ch = load().voice.character
    sr = 24000
    x = _speechlike(sr * 3)
    ref = apply(x, sr, ch)
    cs = CharacterStream(sr, ch)
    st = np.concatenate([cs.process(x[i:i + 2048]) for i in range(0, len(x), 2048)])

    def spec(y):
        f = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        return 20 * np.log10(np.add.reduceat(f, np.linspace(0, len(f) - 1, 60).astype(int))
                             + 1e-9)

    assert np.corrcoef(spec(ref), spec(st))[0, 1] > 0.98          # même timbre
    rms = [20 * np.log10(np.sqrt(np.mean(y ** 2))) for y in (ref, st)]
    assert abs(rms[0] - rms[1]) < 1.5                              # même niveau
    assert np.abs(st).max() <= ch.peak + 1e-9                      # jamais de saturation
    jumps = [abs(st[i] - st[i - 1]) for i in range(2048, len(st), 2048)]
    assert max(jumps) <= np.abs(np.diff(st)).max()                 # pas de clic aux raccords


def test_xtts_stream_uses_cached_latents_and_original_settings(tmp_path):
    mdir, ref = _voice_files(tmp_path)
    model = _FakeXtts()
    calls = []

    def inference_stream(text, language, lat, emb, **kw):
        calls.append((text, lat, emb, kw))
        for i in range(3):
            yield np.full(100, 0.1 * (i + 1), dtype=np.float32)

    model.inference_stream = inference_stream
    be = XttsBackend(mdir, ref, loader=lambda d, dev: (model, model.config))
    chunks = list(be.stream("Salut.", chunk_size=15))
    assert [round(float(c[0]), 2) for c in chunks] == [0.1, 0.2, 0.3]
    text, lat, emb, kw = calls[0]
    assert (lat, emb) == ("LAT", "EMB") and kw["stream_chunk_size"] == 15
    assert kw["temperature"] == 0.75 and kw["enable_text_splitting"] is False
    assert len(model.latent_calls) == 1


def test_xtts_stream_falls_back_to_whole_sentence(tmp_path):
    mdir, ref = _voice_files(tmp_path)
    model = _FakeXtts()                          # pas d'inference_stream
    be = XttsBackend(mdir, ref, loader=lambda d, dev: (model, model.config))
    chunks = list(be.stream("Salut."))
    assert len(chunks) == 1 and chunks[0].dtype == np.float32
