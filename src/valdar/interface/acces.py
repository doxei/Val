"""Qui a le droit de parler à Valdar depuis le réseau (le téléphone d'Olivier).

Le serveur de l'interface écoutait sur 127.0.0.1 : rien ne venait du réseau, donc rien à
vérifier. Pour le téléphone, il doit écouter sur le Wi-Fi — alors tout ce qui ne vient pas
de la machine elle-même doit montrer un **jeton**.

- Le jeton est tiré au sort à la première utilisation et gardé dans `data/tel.token`
  (jamais dans le dépôt : `data/` est ignoré par git). Il n'apparaît ni dans un journal,
  ni dans un message d'erreur.
- L'appairage se fait en scannant un QR code : l'adresse du PC avec le jeton dedans. Le
  téléphone le garde, et le renvoie à chaque fois.
- Le jeton dit « c'est le téléphone d'Olivier », avec une confiance réglable : plus basse
  que le clavier par défaut, donc le corps numérique n'est pas ouvert depuis le téléphone
  tant qu'Olivier ne le décide pas (`tel.corps: true`).
"""
from __future__ import annotations

import contextlib
import hmac
import ipaddress
import secrets
import socket
from pathlib import Path

from valdar.config.loader import Identity


def local_ips() -> list[str]:
    """Les adresses de ce PC sur le réseau local, la plus probable d'abord."""
    out: list[str] = []
    try:                      # l'adresse par laquelle on sortirait : celle du bon Wi-Fi
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.1.1", 9))
        out.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in out and not ip.startswith("127."):
                out.append(ip)
    except OSError:
        pass
    return out


def is_local(host: str) -> bool:
    """La machine elle-même (la fenêtre de Valdar) : pas besoin de jeton."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost", "")


class Access:
    def __init__(self, path: Path, confidence: float = 0.7, person: str = "olivier",
                 name: str = "Olivier", role: str = "owner"):
        self.path = path
        self.confidence = confidence
        self._who = (person, name, role)
        self._token = ""

    # ------------------------------------------------------------ le jeton
    @property
    def token(self) -> str:
        if self._token:
            return self._token
        if self.path.is_file():
            kept = self.path.read_text("utf-8").strip()
            if len(kept) >= 20:
                self._token = kept
                return self._token
        self._token = secrets.token_urlsafe(24)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self._token, "utf-8")
        with contextlib.suppress(OSError):   # sous Linux ça protège, sous Windows c'est sans effet
            self.path.chmod(0o600)
        return self._token

    def renew(self) -> str:
        """Téléphone perdu : on change le jeton, les appareils déjà appairés sont sortis."""
        self._token = ""
        if self.path.is_file():
            self.path.unlink()
        return self.token

    def ok(self, given: str) -> bool:
        return bool(given) and hmac.compare_digest(given, self.token)

    # ------------------------------------------------------------ qui parle
    def identity(self) -> Identity:
        person, name, role = self._who
        return Identity(person=person, name=name, role=role, confidence=self.confidence)

    # ------------------------------------------------------------ appairage
    def url(self, host: str, port: int) -> str:
        return f"http://{host}:{port}/tel.html#j={self.token}"

    def pairing(self, port: int) -> tuple[str, list[str]]:
        """L'adresse à scanner, et les autres adresses possibles du PC."""
        ips = local_ips()
        if not ips:
            return "", []
        return self.url(ips[0], port), [self.url(ip, port) for ip in ips[1:]]


# ------------------------------------------------------------------ le QR code
def qr_lines(text: str) -> list[str]:
    """Le QR code en caractères, pour le terminal. Vide si « qrcode » n'est pas installé."""
    try:
        import qrcode
    except ImportError:
        return []
    q = qrcode.QRCode(border=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
    q.add_data(text)
    q.make(fit=True)
    m = q.get_matrix()
    # deux lignes du QR par ligne de texte : les demi-blocs gardent le carré carré
    rows = []
    for y in range(0, len(m), 2):
        haut, bas = m[y], m[y + 1] if y + 1 < len(m) else [False] * len(m[y])
        line = ""
        for x in range(len(haut)):
            a, b = haut[x], bas[x]
            line += "█" if a and b else "▀" if a else "▄" if b else " "
        rows.append(line)
    return rows


def qr_png(text: str, dest: Path, box: int = 8) -> Path | None:
    """Le même QR code en image, pour l'afficher dans la fenêtre de Valdar."""
    try:
        import qrcode
    except ImportError:
        return None
    img = qrcode.make(text, box_size=box, border=2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    return dest
