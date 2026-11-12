"""Home Assistant : le téléphone d'Olivier et sa maison, pilotés par Valdar.

Aucune requête ne part : Home Assistant est bouchonné. Ce qui est vérifié, c'est ce que
Valdar envoie, et surtout ce qu'il refuse d'envoyer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from valdar.config.loader import DomotiqueConfig
from valdar.tools import domotique as dom
from valdar.tools.registry import Registry, risky


def _maison(tmp_path: Path, **kw: Any) -> tuple[dom.Maison, list[dict[str, Any]]]:
    envoyes: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _json
        envoyes.append({"url": str(req.url), "corps": _json.loads(req.content or b"{}"),
                        "auth": req.headers.get("authorization", "")})
        if req.url.path == "/api/states/sensor.salon":
            return httpx.Response(200, json={"state": "21.4", "attributes": {
                "friendly_name": "Salon", "unit_of_measurement": "°C"}})
        return httpx.Response(200, json=[])

    base = {"enabled": True, "url": "http://ha.local:8123",
            "telephone": "mobile_app_pixel_d_olivier"}
    base.update(kw)
    cfg = DomotiqueConfig(**base)
    return dom.Maison(cfg, "jeton-secret",
                      httpx.Client(transport=httpx.MockTransport(handler))), envoyes


def test_le_jeton_vient_de_lenvironnement_ou_de_data(tmp_path: Path, monkeypatch) -> None:
    cfg = DomotiqueConfig(jeton="data/ha.token")
    monkeypatch.setenv("VALDAR_HA_TOKEN", "  depuis-lenv  ")
    assert dom.read_token(cfg, tmp_path) == "depuis-lenv"
    monkeypatch.delenv("VALDAR_HA_TOKEN")
    assert dom.read_token(cfg, tmp_path) == ""
    p = dom.write_token(cfg, tmp_path, "  depuis-le-fichier\n")
    assert p == tmp_path / "data/ha.token"
    assert dom.read_token(cfg, tmp_path) == "depuis-le-fichier"


def test_le_jeton_nest_jamais_dans_le_depot() -> None:
    racine = Path(__file__).parents[1]
    cfg = (racine / "config/valdar.yaml").read_text("utf-8")
    assert "jeton: data/ha.token" in cfg          # le chemin, jamais la valeur
    assert "data/*" in (racine / ".gitignore").read_text("utf-8")


def test_notifier_et_afficher_une_page(tmp_path: Path) -> None:
    m, envoyes = _maison(tmp_path)
    assert "téléphone" in m.notify("le four est fini", titre="Atelier")
    assert envoyes[-1]["url"].endswith("/api/services/notify/mobile_app_pixel_d_olivier")
    assert envoyes[-1]["corps"] == {"message": "le four est fini", "title": "Atelier"}
    assert envoyes[-1]["auth"] == "Bearer jeton-secret"

    m.afficher("https://exemple.fr/plan")
    corps = envoyes[-1]["corps"]
    assert corps["message"] == "command_webview"
    assert corps["data"]["command"] == "https://exemple.fr/plan"


def test_parler_a_voix_haute_dans_le_telephone(tmp_path: Path) -> None:
    m, envoyes = _maison(tmp_path)
    m.parler("je suis là")
    assert envoyes[-1]["corps"]["message"] == "TTS"
    assert envoyes[-1]["corps"]["data"]["tts_text"] == "je suis là"


def test_la_musique_part_sur_lenceinte_sinon_sur_le_tel(tmp_path: Path) -> None:
    m, envoyes = _maison(tmp_path, enceinte="media_player.salon")
    m.musique("spotify:track:xyz")
    assert envoyes[-1]["url"].endswith("/api/services/media_player/play_media")
    assert envoyes[-1]["corps"]["entity_id"] == "media_player.salon"

    m2, env2 = _maison(tmp_path)                  # pas d'enceinte : le téléphone
    m2.musique("https://radio.fr/flux")
    assert env2[-1]["corps"]["message"] == "command_activity"


def test_il_refuse_ce_qui_nest_pas_autorise(tmp_path: Path) -> None:
    m, envoyes = _maison(tmp_path)
    with pytest.raises(dom.HaError):
        m.commande("command_bluetooth_set", {})          # pas dans la liste
    with pytest.raises(dom.HaError):
        m.service("lock", "unlock", {"entity_id": "lock.porte"})   # jamais les serrures
    with pytest.raises(dom.HaError):
        m.afficher("javascript:alert(1)")
    assert envoyes == []                                  # rien n'est parti


def test_sans_telephone_nomme_il_le_dit(tmp_path: Path) -> None:
    m, _ = _maison(tmp_path, telephone="")
    with pytest.raises(dom.HaError) as e:
        m.notify("coucou")
    assert "quel téléphone" in str(e.value)


def test_un_home_assistant_injoignable_nest_pas_un_drame(tmp_path: Path) -> None:
    def tombe(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("pas de réseau")

    cfg = DomotiqueConfig(enabled=True, telephone="mobile_app_x")
    m = dom.Maison(cfg, "x", httpx.Client(transport=httpx.MockTransport(tombe)))
    with pytest.raises(dom.HaError) as e:
        m.notify("coucou")
    assert "injoignable" in str(e.value)
    reg = Registry()
    dom.add_maison_tools(reg, m)
    assert "impossible" in reg.tools["tel_notifier"].fn(message="coucou")


def test_lire_letat_dun_capteur(tmp_path: Path) -> None:
    m, _ = _maison(tmp_path)
    assert m.etat("sensor.salon") == "Salon : 21.4 °C"


def test_les_outils_sont_reserves_a_olivier_et_demandent_pour_ce_qui_saffiche(
        tmp_path: Path) -> None:
    m, _ = _maison(tmp_path, services=["light.turn_on"])
    reg = Registry()
    dom.add_maison_tools(reg, m)
    for nom in ("tel_notifier", "tel_afficher", "tel_parler", "musique", "etat_maison",
                "agir_maison"):
        assert nom in reg.tools, nom
        assert reg.tools[nom].min_confidence == 0.75
    assert risky(reg.tools["tel_afficher"], {"adresse": "https://x.fr"})
    assert risky(reg.tools["musique"], {"quoi": "x"})
    assert risky(reg.tools["agir_maison"], {"domaine": "light", "service": "turn_on"})
    assert not risky(reg.tools["tel_notifier"], {"message": "coucou"})
    assert not risky(reg.tools["etat_maison"], {"entite": "sensor.salon"})


def test_sans_jeton_la_maison_nexiste_pas(tmp_path: Path) -> None:
    cfg = DomotiqueConfig(enabled=True, jeton="data/absent.token")
    assert dom.build(cfg, tmp_path) is None
    assert dom.describe(None) == "pas configurée"
