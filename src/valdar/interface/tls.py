"""Le certificat de Valdar, pour que le téléphone soit en https.

Pourquoi c'est obligatoire : le micro et la caméra d'un téléphone ne s'ouvrent que sur une
page « sûre ». `http://192.168.1.x` n'en est pas une — en http, le navigateur refuse le
micro, point final. Donc Valdar fabrique son propre certificat (valable pour son adresse
locale, son adresse Tailscale et son nom de machine) et sert la page en https.

Comme le certificat est le sien et non celui d'une autorité, le téléphone affiche un
avertissement **la première fois** : « Avancé » puis « Continuer ». Une fois accepté, il ne
redemande plus, et le micro et la caméra marchent. Par Tailscale avec `tailscale serve`, il
n'y a même pas d'avertissement : c'est Tailscale qui fournit un vrai certificat.

Le certificat et sa clé vivent dans `data/` : jamais dans le dépôt, jamais dans un message.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import ipaddress
import json
import ssl
import subprocess
from pathlib import Path

WINDOWS_TAILSCALE = r"C:\Program Files\Tailscale\tailscale.exe"


def _tailscale_cmd() -> list[str] | None:
    import os
    import shutil

    found = shutil.which("tailscale")
    if found:
        return [found]
    if os.name == "nt" and Path(WINDOWS_TAILSCALE).is_file():
        return [WINDOWS_TAILSCALE]
    return None


def tailnet() -> dict[str, str]:
    """L'adresse Tailscale de ce PC et son nom MagicDNS, si Tailscale tourne."""
    cmd = _tailscale_cmd()
    if cmd is None:
        return {}
    try:
        out = subprocess.run([*cmd, "status", "--json"], capture_output=True, timeout=6,
                             text=True)
        data = json.loads(out.stdout or "{}")
    except Exception:
        return {}
    me = data.get("Self") or {}
    ips = [ip for ip in (me.get("TailscaleIPs") or []) if ":" not in ip]
    name = (me.get("DNSName") or "").rstrip(".")
    res = {}
    if ips:
        res["ip"] = ips[0]
    if name:
        res["nom"] = name
    if data.get("BackendState") and data["BackendState"] != "Running":
        res["etat"] = str(data["BackendState"])
    return res


def serve_url() -> str:
    """L'adresse https de Tailscale si `tailscale serve` est déjà en place."""
    cmd = _tailscale_cmd()
    if cmd is None:
        return ""
    try:
        out = subprocess.run([*cmd, "serve", "status", "--json"], capture_output=True,
                             timeout=6, text=True)
        data = json.loads(out.stdout or "{}")
    except Exception:
        return ""
    for host in (data.get("Web") or {}):
        if host:
            base = host.split(":")[0]
            port = host.split(":")[-1] if ":" in host else "443"
            return f"https://{base}" + ("" if port == "443" else f":{port}")
    return ""


# ------------------------------------------------------------------ certificat
def _names(hosts: list[str]) -> tuple[list[str], list[str]]:
    ips, dns = [], []
    for h in hosts:
        if not h:
            continue
        try:
            ipaddress.ip_address(h)
            if h not in ips:
                ips.append(h)
        except ValueError:
            if h not in dns:
                dns.append(h)
    return ips, dns


def ensure_cert(folder: Path, hosts: list[str]) -> tuple[Path, Path] | None:
    """Le certificat pour ces adresses. Refait si les adresses ont change."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        return None
    ips, dns = _names(hosts)
    dns = ["localhost", *dns]
    ips = ["127.0.0.1", *ips]
    folder.mkdir(parents=True, exist_ok=True)
    cert_file, key_file = folder / "valdar.crt", folder / "valdar.key"
    marque = folder / "valdar.crt.adresses"
    attendu = json.dumps({"ip": sorted(ips), "dns": sorted(dns)}, sort_keys=True)
    if cert_file.is_file() and key_file.is_file() and marque.is_file() \
            and marque.read_text("utf-8") == attendu:
        return cert_file, key_file

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sujet = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Valdar"),
                       x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Valdar (local)")])
    alt = [x509.DNSName(d) for d in dns] + \
          [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    now = dt.datetime.now(dt.UTC)
    cert = (x509.CertificateBuilder()
            .subject_name(sujet).issuer_name(sujet)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    key_file.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    marque.write_text(attendu, "utf-8")
    for f in (key_file, marque):
        with contextlib.suppress(OSError):
            f.chmod(0o600)
    return cert_file, key_file


def context(cert: Path, key: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
    return ctx


def fingerprint(cert: Path) -> str:
    """L'empreinte du certificat, pour qu'Olivier puisse verifier ce qu'il accepte."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
    except ImportError:
        return ""
    c = x509.load_pem_x509_certificate(cert.read_bytes())
    digest = c.fingerprint(hashes.SHA256()).hex().upper()
    return ":".join(digest[i:i + 2] for i in range(0, len(digest), 2))
