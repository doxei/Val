"""Home Assistant : la maison d'Olivier, et son téléphone, à portée de Valdar.

L'appli compagnon Home Assistant installée sur le téléphone accepte des ordres venus de
Home Assistant : afficher une page, lancer une appli ou de la musique, parler à voix haute,
notifier, allumer l'écran. Valdar passe par là — pas d'appli à fabriquer, et ça marche de
partout où Home Assistant est joignable (chez lui, ou par Tailscale).

Le jeton de Home Assistant vit dans `data/` (jamais dans le dépôt, jamais dans un message) :
soit dans le fichier dit par la configuration, soit dans la variable d'environnement
VALDAR_HA_TOKEN.

Garde-fous : seuls les services autorisés dans la configuration peuvent être appelés, ce qui
s'affiche ou démarre demande un « oui », et rien ici ne touche aux serrures ni aux alarmes
tant qu'Olivier ne les a pas mises dans la liste lui-même.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import httpx

TIMEOUT = 15.0


class HaError(RuntimeError):
    pass


def read_token(cfg: Any, root: Path) -> str:
    """Le jeton, pris dans l'environnement d'abord, puis dans son fichier. Jamais ailleurs."""
    env = os.environ.get("VALDAR_HA_TOKEN", "").strip()
    if env:
        return env
    p = Path(cfg.jeton)
    if not p.is_absolute():
        p = root / p
    if p.is_file():
        return p.read_text("utf-8").strip()
    return ""


class Maison:
    """Le lien avec Home Assistant. Tout ce qui sort d'ici est tracé et limité."""

    def __init__(self, cfg: Any, token: str, client: httpx.Client | None = None):
        self.cfg = cfg
        self._token = token
        self.http = client or httpx.Client(timeout=TIMEOUT)

    @property
    def ready(self) -> bool:
        return bool(self._token and self.cfg.url)

    def _call(self, chemin: str, data: dict[str, Any] | None = None) -> Any:
        if not self.ready:
            raise HaError("Home Assistant n'est pas configuré (adresse ou jeton manquant)")
        url = self.cfg.url.rstrip("/") + chemin
        head = {"Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json"}
        try:
            r = (self.http.post(url, headers=head, json=data or {}) if data is not None
                 else self.http.get(url, headers=head))
        except httpx.HTTPError as exc:
            raise HaError(f"Home Assistant injoignable : {exc}") from exc
        if r.status_code == 401:
            raise HaError("Home Assistant refuse mon jeton (expiré ?)")
        if r.status_code >= 400:
            raise HaError(f"Home Assistant répond {r.status_code} : {r.text[:200]}")
        try:
            return r.json()
        except ValueError:
            return r.text

    # ------------------------------------------------------------ le téléphone
    def _phone_service(self) -> str:
        nom = (self.cfg.telephone or "").strip()
        if not nom:
            raise HaError("je ne sais pas quel téléphone : mets son nom dans "
                          "domotique.telephone (ex. mobile_app_pixel_d_olivier)")
        return nom if nom.startswith("mobile_app_") else f"mobile_app_{nom}"

    def notify(self, message: str, titre: str = "", data: dict[str, Any] | None = None
               ) -> str:
        corps: dict[str, Any] = {"message": message}
        if titre:
            corps["title"] = titre
        if data:
            corps["data"] = data
        self._call(f"/api/services/notify/{self._phone_service()}", corps)
        return "envoyé sur ton téléphone"

    def commande(self, commande: str, data: dict[str, Any] | None = None) -> str:
        """Une commande de l'appli compagnon (command_webview, command_activity…)."""
        if commande not in self.cfg.commandes:
            raise HaError(f"« {commande} » n'est pas dans les commandes autorisées")
        return self.notify(commande, data=data)

    def afficher(self, adresse: str) -> str:
        if not adresse.startswith(("http://", "https://")):
            raise HaError("il me faut une adresse web complète")
        self.commande("command_webview", {"command": adresse})
        return f"affiché sur ton téléphone : {adresse}"

    def parler(self, texte: str) -> str:
        self.notify("TTS", data={"tts_text": texte, "media_stream": "music_stream"})
        return "dit à voix haute sur ton téléphone"

    def musique(self, quoi: str, enceinte: str = "") -> str:
        cible = enceinte or self.cfg.enceinte
        if cible:
            self._call("/api/services/media_player/play_media",
                       {"entity_id": cible, "media_content_id": quoi,
                        "media_content_type": "music"})
            return f"lancé sur {cible}"
        self.commande("command_activity",
                      {"command": "android.intent.action.VIEW", "intent_uri": quoi})
        return "lancé sur ton téléphone"

    # ------------------------------------------------------------ la maison
    def etat(self, entite: str) -> str:
        data = self._call(f"/api/states/{entite}")
        if not isinstance(data, dict):
            return str(data)[:300]
        attrs = data.get("attributes") or {}
        nom = attrs.get("friendly_name") or entite
        unite = attrs.get("unit_of_measurement") or ""
        return f"{nom} : {data.get('state')}{(' ' + unite) if unite else ''}"

    def service(self, domaine: str, service: str, data: dict[str, Any] | None = None) -> str:
        plein = f"{domaine}.{service}"
        if plein not in self.cfg.services:
            raise HaError(f"« {plein} » n'est pas dans les services autorisés")
        out = self._call(f"/api/services/{domaine}/{service}", data or {})
        n = len(out) if isinstance(out, list) else 1
        return f"{plein} fait ({n} chose(s) ont changé)"


# ------------------------------------------------------------------ les outils
def add_maison_tools(reg: Any, maison: Maison, journal: Any = None) -> None:
    """Pose les outils du téléphone et de la maison dans le registre."""
    from valdar.tools.registry import ELEVATED, Tool, params

    s = {"type": "string"}
    owner_min = 0.75

    def pose(nom: str, desc: str, p: dict[str, Any], fn: Any, confirme: Any = None,
             dire: Any = None) -> None:
        def run(**kw: Any) -> str:
            try:
                out = fn(**kw)
            except HaError as exc:
                return f"impossible : {exc}"
            if journal is not None:
                journal.write(nom, kw, "olivier", out)
            return out
        reg.add(Tool(nom, desc, p, run, ELEVATED, "maison", min_confidence=owner_min,
                     confirm_if=confirme, describe=dire))

    pose("tel_notifier", "Envoie une notification sur le téléphone d'Olivier.",
         params(["message"], message=s, titre=s), maison.notify)
    pose("tel_afficher", "Affiche une page web sur l'écran du téléphone d'Olivier.",
         params(["adresse"], adresse=s), maison.afficher,
         confirme=lambda a: True, dire=lambda a: f"afficher {a.get('adresse')} sur ton tél")
    pose("tel_parler", "Fait parler Valdar à voix haute dans le téléphone d'Olivier.",
         params(["texte"], texte=s), maison.parler)
    pose("musique", "Lance de la musique (sur une enceinte de la maison ou sur le tél).",
         params(["quoi"], quoi=s, enceinte=s), maison.musique,
         confirme=lambda a: True, dire=lambda a: f"lancer « {a.get('quoi')} »")
    pose("etat_maison", "Lit l'état d'un appareil de la maison (capteur, lampe, prise…).",
         params(["entite"], entite=s), maison.etat)
    if maison.cfg.services:
        pose("agir_maison",
             "Agit sur la maison (allumer, éteindre…), parmi ce qu'Olivier a autorisé.",
             params(["domaine", "service"], domaine=s, service=s,
                    entite={"type": "string", "description": "l'appareil visé"}),
             lambda domaine, service, entite="": maison.service(
                 domaine, service, {"entity_id": entite} if entite else None),
             confirme=lambda a: True,
             dire=lambda a: f"{a.get('domaine')}.{a.get('service')} sur {a.get('entite')}")


def build(cfg: Any, root: Path) -> Maison | None:
    """La maison, si elle est configurée. Sinon rien, et Valdar s'en passe."""
    if not cfg.enabled:
        return None
    token = read_token(cfg, root)
    if not token:
        return None
    return Maison(cfg, token)


def write_token(cfg: Any, root: Path, token: str) -> Path:
    """Range le jeton de Home Assistant là où il doit vivre : dans data/, lisible par lui."""
    p = Path(cfg.jeton)
    if not p.is_absolute():
        p = root / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token.strip(), "utf-8")
    with contextlib.suppress(OSError):
        p.chmod(0o600)
    return p


def describe(maison: Maison | None) -> str:
    if maison is None:
        return "pas configurée"
    if not maison.ready:
        return "configurée, mais sans jeton"
    try:
        data = maison._call("/api/")
    except HaError as exc:
        return f"injoignable ({exc})"
    msg = data.get("message") if isinstance(data, dict) else ""
    return f"joignable ({msg or 'ok'})" + (f", téléphone : {maison.cfg.telephone}"
                                           if maison.cfg.telephone else
                                           ", mais aucun téléphone nommé")


__all__ = ["HaError", "Maison", "add_maison_tools", "build", "describe", "read_token",
           "write_token"]
