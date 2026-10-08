"""Caractère de la voix : la chaîne d'effets historique de la voix de Valdar (profil
« megatron »), à l'identique.

Reproduction échantillon pour échantillon de la chaîne d'origine (`_character` et ses
filtres), vérifiée par `tests/test_voice.py` contre une copie de référence du code d'origine.
Deux particularités sont gardées exprès, parce que c'est ce son-là qu'Olivier a construit et
validé pour cette voix :
- la modulation d'amplitude utilise une profondeur fixe (2 %) ;
- le « décalage de hauteur » ne change pas la hauteur : il sous-échantillonne puis
  ré-échantillonne à la longueur d'origine, ce qui revient à un passe-bas à fs/4
  (6 kHz pour XTTS à 24 kHz). Ça fait partie du timbre « radio » de la voix.
"""
from __future__ import annotations

import re

import numpy as np

from valdar.config.loader import VoiceCharacter


def _band_peaking(x: np.ndarray, sr: int, fc: float, gain_db: float, q: float) -> np.ndarray:
    from scipy.signal import sosfilt

    w0 = 2.0 * np.pi * fc / sr
    a_ = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    b = np.array([1.0 + alpha * a_, -2.0 * np.cos(w0), 1.0 - alpha * a_])
    a = np.array([1.0 + alpha / a_, -2.0 * np.cos(w0), 1.0 - alpha / a_])
    sos = np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])
    return sosfilt(sos, x)


def _high_shelf(x: np.ndarray, sr: int, fc: float, gain_db: float) -> np.ndarray:
    from scipy.signal import sosfilt

    a_ = np.sqrt(10.0 ** (gain_db / 20.0))
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / np.sqrt(2.0)
    c = np.cos(w0)
    b = a_ * np.array([
        (a_ + 1) + (a_ - 1) * c + 2 * np.sqrt(a_) * alpha,
        -2 * ((a_ - 1) + (a_ + 1) * c),
        (a_ + 1) + (a_ - 1) * c - 2 * np.sqrt(a_) * alpha,
    ])
    a = np.array([
        (a_ + 1) - (a_ - 1) * c + 2 * np.sqrt(a_) * alpha,
        2 * ((a_ - 1) - (a_ + 1) * c),
        (a_ + 1) - (a_ - 1) * c - 2 * np.sqrt(a_) * alpha,
    ])
    sos = np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])
    return sosfilt(sos, x)


def _resample_tone(x: np.ndarray, semis: float) -> np.ndarray:
    """Le « pitch shift » d'origine, tel quel (voir l'en-tête du module)."""
    f = 2.0 ** (semis / 12.0)
    n = len(x)
    if abs(f - 1.0) < 1e-3 or n < 16:
        return x
    from scipy.signal import resample_poly

    up, down = f, 1.0
    if f < 1.0:
        up, down = 1.0, 1.0 / f
    g = pow(10.0, 3.0 / 20.0)
    x2 = resample_poly(x, int(round(up * g)), max(1, int(round(down * g))))
    return resample_poly(x2, n, len(x2)) if len(x2) else x


def _echo_tail(x: np.ndarray, sr: int, ms: float, gain: float) -> np.ndarray:
    d = max(1, int(sr * ms / 1000.0))
    n = len(x)
    if n <= d:
        return x
    taps = [gain, gain * 0.55, gain * 0.28]
    out = x.copy()
    for k, g in enumerate(taps, start=1):
        pad = k * d
        if n <= pad:
            break
        out[pad:] = out[pad:] + g * x[: n - pad]
    peak = np.max(np.abs(out)) or 1.0
    return out * (0.92 / peak)


def apply(audio: np.ndarray, sr: int, ch: VoiceCharacter) -> np.ndarray:
    """int16 (sortie du TTS) → float64 dans [-1, 1], avec le caractère de la voix."""
    x = audio.astype(np.float64) / 32768.0
    t = np.arange(len(x)) / sr
    if ch.am_hz and ch.am_depth:
        x = x * (1.0 + ch.am_depth * np.sin(2 * np.pi * ch.am_hz * t))
    if ch.wobble_hz and ch.wobble_depth:
        x = x * (1.0 + ch.wobble_depth * np.sin(2 * np.pi * ch.wobble_hz * t))
    if ch.nasal_gain_db:
        x = _band_peaking(x, sr, 1700.0, ch.nasal_gain_db, 1.0)
    if ch.pitch_shift:
        x = _resample_tone(x, ch.pitch_shift)
    if sr >= 800:
        x = _high_shelf(x, sr, ch.high_shelf_hz, ch.high_shelf_db)
    if ch.echo_ms and ch.echo_gain:
        x = _echo_tail(x, sr, ch.echo_ms, ch.echo_gain)
    if ch.lo_fi_bits < 16:
        levels = 2 ** ch.lo_fi_bits
        x = np.round(x * (levels - 1)) / max(1, levels - 1)
    x = np.tanh(x * ch.drive)
    peak = np.max(np.abs(x)) if len(x) else 0.0
    x = x * (ch.peak / (peak or 1.0))
    return np.clip(x, -1.0, 1.0)


def to_int16(x: np.ndarray) -> np.ndarray:
    return (x * 32767.0).astype(np.int16)


_TAG = re.compile(r"\[[a-zA-Z_-]+\.?\]")
_MARKUP = re.compile(r"[*_`#>]+")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")


def speakable(text: str) -> str:
    """Texte prêt à prononcer : balises [rire], markdown et émojis retirés."""
    text = _TAG.sub("", text)
    text = _MARKUP.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
