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


class CharacterStream:
    """La même chaîne « megatron », morceau par morceau (XTTS en flux).

    Chaque filtre garde son état d'un morceau au suivant (`sosfilt`/`lfilter` avec `zi`),
    la modulation garde sa phase, l'écho garde sa queue : le son est continu, sans clic aux
    raccords. Deux différences assumées avec `apply` (qui voit la phrase entière d'un coup) :
    - le « décalage de hauteur » d'origine (aller-retour de rééchantillonnage, donc un
      passe-bas à fs/4) devient le même passe-bas en filtre à état ;
    - le niveau final ne peut pas être normalisé sur un pic qu'on n'a pas encore vu : le gain
      part du premier morceau et ne fait que baisser si un pic plus fort arrive (jamais de
      saturation, jamais de pompage).
    Réutiliser le même objet pour toute une réponse (les phrases s'enchaînent sans raccord).
    """

    def __init__(self, sr: int, ch: VoiceCharacter):
        from scipy.signal import firwin

        self.sr, self.ch = sr, ch
        self.n = 0                                # échantillons déjà traités (phase)
        self._sos: list[tuple[np.ndarray, np.ndarray]] = []
        if ch.nasal_gain_db:
            s = _peaking_sos(sr, 1700.0, ch.nasal_gain_db, 1.0)
            self._sos.append((s, np.zeros((s.shape[0], 2))))
        self._fir: list[tuple[np.ndarray, np.ndarray]] = []
        if ch.pitch_shift:
            f = 2.0 ** (ch.pitch_shift / 12.0)
            if abs(f - 1.0) >= 1e-3:
                up, down = (f, 1.0) if f >= 1.0 else (1.0, 1.0 / f)
                g = pow(10.0, 3.0 / 20.0)
                u, d = int(round(up * g)), max(1, int(round(down * g)))
                hi, lo = max(u, d), min(u, d)
                if lo != hi:
                    taps = firwin(2 * 10 * hi + 1, lo / hi, window=("kaiser", 5.0))
                    for _ in range(2):            # descente puis remontée
                        self._fir.append((taps, np.zeros(len(taps) - 1)))
        self._shelf: tuple[np.ndarray, np.ndarray] | None = None
        if sr >= 800:
            s = _shelf_sos(sr, ch.high_shelf_hz, ch.high_shelf_db)
            self._shelf = (s, np.zeros((s.shape[0], 2)))
        self._echo_tail = np.zeros(0)
        self._in_gain: float | None = None        # entrée : comme XTTS normalisé à pleine échelle
        self._out_gain: float | None = None

    def process(self, audio: np.ndarray) -> np.ndarray:
        """int16 ou float (morceau du TTS) → float64 dans [-1, 1]."""
        from scipy.signal import lfilter, sosfilt

        ch, sr = self.ch, self.sr
        x = np.asarray(audio, dtype=np.float64)
        if np.issubdtype(np.asarray(audio).dtype, np.integer):
            x = x / 32768.0
        if not len(x):
            return x
        pk = float(np.max(np.abs(x)))
        if self._in_gain is None or pk * self._in_gain > 1.0:
            self._in_gain = 1.0 / max(pk, 0.05)
        x = x * self._in_gain
        t = (self.n + np.arange(len(x))) / sr
        self.n += len(x)
        if ch.am_hz and ch.am_depth:
            x = x * (1.0 + ch.am_depth * np.sin(2 * np.pi * ch.am_hz * t))
        if ch.wobble_hz and ch.wobble_depth:
            x = x * (1.0 + ch.wobble_depth * np.sin(2 * np.pi * ch.wobble_hz * t))
        for i, (sos, zi) in enumerate(self._sos):
            x, zi = sosfilt(sos, x, zi=zi)
            self._sos[i] = (sos, zi)
        for i, (taps, zi) in enumerate(self._fir):
            x, zi = lfilter(taps, 1.0, x, zi=zi)
            self._fir[i] = (taps, zi)
        if self._shelf is not None:
            sos, zi = self._shelf
            x, zi = sosfilt(sos, x, zi=zi)
            self._shelf = (sos, zi)
        if ch.echo_ms and ch.echo_gain:
            x = self._echo(x)
        if ch.lo_fi_bits < 16:
            levels = 2 ** ch.lo_fi_bits
            x = np.round(x * (levels - 1)) / max(1, levels - 1)
        x = np.tanh(x * ch.drive)
        pk = float(np.max(np.abs(x)))
        want = ch.peak / max(pk, 1e-6)
        self._out_gain = want if self._out_gain is None else min(self._out_gain, want)
        return np.clip(x * self._out_gain, -1.0, 1.0)

    def _echo(self, x: np.ndarray) -> np.ndarray:
        d = max(1, int(self.sr * self.ch.echo_ms / 1000.0))
        g = self.ch.echo_gain
        hist = np.concatenate([self._echo_tail, x])
        out = x.copy()
        base = len(self._echo_tail)
        for k, gk in enumerate([g, g * 0.55, g * 0.28], start=1):
            src = np.arange(len(x)) + base - k * d
            ok = src >= 0
            out[ok] += gk * hist[src[ok]]
        self._echo_tail = hist[-3 * d:]
        return out * 0.92


def _peaking_sos(sr: int, fc: float, gain_db: float, q: float) -> np.ndarray:
    w0 = 2.0 * np.pi * fc / sr
    a_ = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    b = np.array([1.0 + alpha * a_, -2.0 * np.cos(w0), 1.0 - alpha * a_])
    a = np.array([1.0 + alpha / a_, -2.0 * np.cos(w0), 1.0 - alpha / a_])
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])


def _shelf_sos(sr: int, fc: float, gain_db: float) -> np.ndarray:
    a_ = np.sqrt(10.0 ** (gain_db / 20.0))
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / np.sqrt(2.0)
    c = np.cos(w0)
    b = a_ * np.array([(a_ + 1) + (a_ - 1) * c + 2 * np.sqrt(a_) * alpha,
                       -2 * ((a_ - 1) + (a_ + 1) * c),
                       (a_ + 1) + (a_ - 1) * c - 2 * np.sqrt(a_) * alpha])
    a = np.array([(a_ + 1) - (a_ - 1) * c + 2 * np.sqrt(a_) * alpha,
                  2 * ((a_ - 1) - (a_ + 1) * c),
                  (a_ + 1) - (a_ - 1) * c - 2 * np.sqrt(a_) * alpha])
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])
