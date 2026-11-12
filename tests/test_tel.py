"""Le téléphone d'Olivier : appairage, jeton, écrit, vocal, vue.

Les requêtes passent par un vrai socket en https, avec le certificat que Valdar fabrique :
c'est le chemin que prendra le téléphone, pas une maquette.
"""
from __future__ import annotations

import io
import json
import ssl
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from valdar.interface.acces import Access, is_local, qr_lines
from valdar.interface.server import Server
from valdar.interface.session import Session
from valdar.interface.tel import TelError, to_wav, wav_to_mono16k
from valdar.interface.tls import ensure_cert, fingerprint
from valdar.llm.backend import ChatResult


# ------------------------------------------------------------------ le jeton
def test_le_jeton_est_tire_au_sort_garde_et_verifie(tmp_path: Path) -> None:
    a = Access(tmp_path / "tel.token")
    jeton = a.token
    assert len(jeton) >= 20
    assert a.ok(jeton) and not a.ok("") and not a.ok(jeton + "x")
    assert Access(tmp_path / "tel.token").token == jeton        # il survit au redémarrage
    neuf = a.renew()
    assert neuf != jeton and not a.ok(jeton)


def test_le_jeton_nest_pas_dans_le_depot(tmp_path: Path) -> None:
    a = Access(tmp_path / "tel.token")
    assert a.token and (tmp_path / "tel.token").is_file()
    gitignore = (Path(__file__).parents[1] / ".gitignore").read_text("utf-8")
    assert "data/*" in gitignore                 # le jeton vit dans data/


def test_qui_parle_depuis_le_telephone(tmp_path: Path) -> None:
    a = Access(tmp_path / "t", confidence=0.7)
    who = a.identity()
    assert who.person == "olivier" and who.role == "owner" and who.confidence == 0.7
    # 0,7 : il parle, il se souvient, mais le corps numérique (0,75) reste fermé
    assert who.confidence < 0.75


def test_la_machine_elle_meme_na_rien_a_prouver() -> None:
    assert is_local("127.0.0.1") and is_local("::1") and is_local("localhost")
    assert not is_local("192.168.1.42") and not is_local("100.64.0.1")


def test_le_qr_code_tient_dans_un_terminal() -> None:
    lignes = qr_lines("https://doxei-pc.tail.ts.net:8766/tel.html#j=" + "x" * 32)
    assert lignes and all(len(x) == len(lignes[0]) for x in lignes)
    assert len(lignes[0]) < 80                    # ça rentre dans une fenêtre normale


# ------------------------------------------------------------------ le certificat
def test_le_certificat_couvre_toutes_ses_adresses(tmp_path: Path) -> None:
    from cryptography import x509

    paire = ensure_cert(tmp_path, ["192.168.1.42", "100.64.0.9", "doxei-pc.tail.ts.net"])
    assert paire is not None
    cert = x509.load_pem_x509_certificate(paire[0].read_bytes())
    alt = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    ips = {str(x) for x in alt.get_values_for_type(x509.IPAddress)}
    noms = set(alt.get_values_for_type(x509.DNSName))
    assert {"127.0.0.1", "192.168.1.42", "100.64.0.9"} <= ips
    assert {"localhost", "doxei-pc.tail.ts.net"} <= noms
    assert len(fingerprint(paire[0])) > 40        # de quoi vérifier ce qu'on accepte


# ------------------------------------------------------------------ le son
def test_le_son_du_telephone_devient_du_16_khz() -> None:
    bruit = (np.sin(np.arange(48000) * 0.05) * 8000).astype("<i2")
    x = wav_to_mono16k(to_wav(bruit, 48000))
    assert abs(len(x) - 16000) < 20 and x.dtype == np.float32
    assert 0.1 < float(np.abs(x).max()) < 1.0


def test_un_son_illisible_le_dit_au_lieu_de_planter() -> None:
    with pytest.raises(TelError):
        wav_to_mono16k(b"pas du wav du tout")
    with pytest.raises(TelError):
        wav_to_mono16k(b"x" * 9_000_000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:              # 8 bits : refusé proprement
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(16000)
        w.writeframes(b"\x00" * 1000)
    with pytest.raises(TelError):
        wav_to_mono16k(buf.getvalue())


# ------------------------------------------------------------------ bout en bout
class FauxTTS:
    sample_rate = 16000

    def __init__(self) -> None:
        self.dit: list[str] = []

    def load(self) -> None:
        return None

    def synthesize(self, text: str) -> np.ndarray:
        self.dit.append(text)
        return (np.sin(np.arange(1600) * 0.1) * 5000).astype(np.int16)


@pytest.fixture()
def poste(runtime_factory, tmp_path):
    """Valdar qui tourne, avec son serveur https tel qu'il sera sur le PC d'Olivier."""
    rt = runtime_factory([ChatResult(content="Je suis là.")])
    rt.cfg.tel.enabled = True
    rt.cfg.tel.port = 0                      # un port libre, choisi par le système
    rt.cfg.tel.host = "127.0.0.1"
    s = Session(rt.cfg, rt, None)
    srv = Server(s, "127.0.0.1", 0, tel=rt.cfg.tel)
    srv.start()
    yield srv, s
    srv.stop()


def _port(srv: Server) -> int:
    return srv.tel_httpd.server_address[1]


def _get(srv: Server, chemin: str, jeton: str = "", binaire: bool = False) -> Any:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(f"https://127.0.0.1:{_port(srv)}{chemin}")
    if jeton:
        req.add_header("X-Valdar-Jeton", jeton)
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:   # noqa: S310
        data = r.read()
    return data if binaire else json.loads(data)


def _post(srv: Server, chemin: str, body: Any, jeton: str = "",
          ctype: str = "application/json") -> Any:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(f"https://127.0.0.1:{_port(srv)}{chemin}", data=raw,
                                 headers={"Content-Type": ctype})
    if jeton:
        req.add_header("X-Valdar-Jeton", jeton)
    with urllib.request.urlopen(req, context=ctx, timeout=15) as r:   # noqa: S310
        return json.loads(r.read())


def test_sans_jeton_le_reseau_nobtient_rien(poste) -> None:
    srv, _ = poste
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(srv, "/api/tel/etat")
    assert e.value.code == 401
    with pytest.raises(urllib.error.HTTPError):
        _get(srv, "/api/memoire", jeton="")
    # la coquille de l'appli, elle, se charge : c'est elle qui demandera le jeton
    page = _get(srv, "/tel.html", binaire=True)
    assert b"Valdar" in page and b"tel.js" in page


def test_avec_un_mauvais_jeton_non_plus(poste) -> None:
    srv, _ = poste
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(srv, "/api/tel/etat", jeton="pas-le-bon")
    assert e.value.code == 401


def test_le_telephone_na_pas_acces_aux_reglages_ni_a_la_memoire(poste) -> None:
    srv, _ = poste
    bon = srv.access.token
    assert _get(srv, "/api/tel/etat", jeton=bon)["nom"] == "Valdar"
    for interdit in ("/api/memoire", "/api/reglages", "/api/foyer", "/api/journal"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(srv, interdit, jeton=bon)
        assert e.value.code == 401, interdit


def test_ecrire_depuis_le_telephone_et_recevoir_sa_reponse(poste) -> None:
    srv, s = poste
    vus: list[tuple[str, dict]] = []
    s.bus.subscribe(lambda k, d: vus.append((k, d)))
    out = _post(srv, "/api/message", {"text": "tu es là ?", "voix": False},
                jeton=srv.access.token)
    assert out["ok"]
    for _ in range(100):
        if any(k == "reply" for k, _ in vus):
            break
        time.sleep(0.05)
    kinds = [k for k, _ in vus]
    assert "user" in kinds and "reply" in kinds
    dit = next(d for k, d in vus if k == "user")
    assert dit["source"] == "téléphone" and dit["person"] == "olivier"
    assert next(d for k, d in vus if k == "reply")["text"] == "Je suis là."


def test_sa_voix_part_vers_le_telephone_pas_dans_le_salon(poste) -> None:
    srv, s = poste
    srv.phone._tts = FauxTTS()
    clips: list[str] = []
    s.bus.subscribe(lambda k, d: clips.append(d["id"]) if k == "tel_audio" else None)
    _post(srv, "/api/message", {"text": "dis-moi un truc", "voix": True},
          jeton=srv.access.token)
    for _ in range(100):
        if clips:
            break
        time.sleep(0.05)
    assert clips, "aucun son produit pour le téléphone"
    wav = _get(srv, f"/api/tel/voix/{clips[0]}", jeton=srv.access.token, binaire=True)
    assert wav[:4] == b"RIFF" and len(wav) > 1000
    assert srv.phone._tts.dit                     # c'est bien sa voix qui a parlé


def test_sans_voix_chargee_il_le_dit_au_lieu_de_faire_semblant(poste) -> None:
    srv, _ = poste
    etat = _get(srv, "/api/tel/etat", jeton=srv.access.token)["voix"]
    assert etat["voix"] is False and etat["pourquoi"]
    assert etat["secours_navigateur"] is True     # le téléphone lira, lui


def test_parler_au_telephone_sans_oreilles_le_dit_clairement(poste) -> None:
    srv, _ = poste
    son = to_wav((np.sin(np.arange(16000) * 0.1) * 9000).astype("<i2"), 16000)
    out = _post(srv, "/api/tel/ecouter", son, jeton=srv.access.token, ctype="audio/wav")
    assert out["ok"] is False and out["error"]
    assert "oreilles" in out["error"] or "compris" in out["error"]


def test_un_silence_nest_pas_envoye_au_modele(poste) -> None:
    srv, _ = poste
    out = _post(srv, "/api/tel/ecouter", to_wav(np.zeros(16000, "<i2"), 16000),
                jeton=srv.access.token, ctype="audio/wav")
    assert out["ok"] is False and "entendu" in out["error"]


def test_la_camera_du_telephone_ne_bloque_jamais(poste) -> None:
    srv, _ = poste
    out = _post(srv, "/api/tel/voir", b"\xff\xd8\xff pas vraiment une image",
                jeton=srv.access.token, ctype="image/jpeg")
    assert out["ok"] is True
    assert "erreur" in out or "qui" in out        # il dit ce qui cloche, il ne tombe pas


def test_renouveler_le_jeton_sort_les_appareils(poste) -> None:
    srv, _ = poste
    vieux = srv.access.token
    assert _post(srv, "/api/tel/jeton", {"action": "renouveler"}, jeton=vieux)["ok"]
    with pytest.raises(urllib.error.HTTPError):
        _get(srv, "/api/tel/etat", jeton=vieux)
    assert _get(srv, "/api/tel/etat", jeton=srv.access.token)["nom"] == "Valdar"


def test_la_fenetre_du_pc_na_pas_besoin_de_jeton(poste) -> None:
    srv, _ = poste
    port = srv.httpd.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=10) as r:  # noqa: S310
        assert json.loads(r.read())["emotion"]


# ------------------------------------------------------------------ la page
def test_lappli_sinstalle_sur_lecran_daccueil() -> None:
    static = Path(__file__).parents[1] / "src/valdar/interface/static"
    m = json.loads((static / "manifeste.json").read_text("utf-8"))
    assert m["start_url"] == "tel.html" and m["display"] == "standalone"
    tailles = {i["sizes"] for i in m["icons"]}
    assert {"192x192", "512x512"} <= tailles
    for i in m["icons"]:
        assert (static / i["src"]).is_file(), i["src"]
    sw = (static / "sw.js").read_text("utf-8")
    assert "/api/" in sw and "return" in sw       # ses paroles ne sont jamais mises en cache
    page = (static / "tel.html").read_text("utf-8")
    assert 'name="viewport"' in page and "manifeste.json" in page


def test_le_visage_est_le_meme_des_deux_cotes() -> None:
    static = Path(__file__).parents[1] / "src/valdar/interface/static"
    svg = (static / "visage.svg").read_text("utf-8")
    assert 'id="face"' in svg and 'id="mouth"' in svg
    for page in ("index.html", "tel.html"):
        assert 'id="visage-hote"' in (static / page).read_text("utf-8"), page
    assert "chargerVisage" in (static / "face.js").read_text("utf-8")


# ------------------------------------------------------------------ assistant du téléphone
def test_valdar_se_fait_passer_pour_un_ollama(poste) -> None:
    """Home Assistant doit le voir comme un Ollama : c'est ce qui le met à la place de
    Gemini sur le téléphone."""
    srv, s = poste
    jeton = srv.access.token
    tags = _get(srv, "/api/tags", jeton=jeton)
    assert tags["models"] and tags["models"][0]["name"] == "valdar:latest"

    vus: list[tuple[str, dict]] = []
    s.bus.subscribe(lambda k, d: vus.append((k, d)))
    out = _post(srv, "/api/chat", {"model": "valdar:latest", "stream": False,
                                   "messages": [{"role": "user", "content": "quelle heure ?"}]},
                jeton=jeton)
    assert out["done"] and out["message"]["role"] == "assistant"
    assert out["message"]["content"] == "Je suis là."
    dit = next(d for k, d in vus if k == "user")
    assert dit["source"] == "assistant du téléphone"     # un vrai tour, pas un à-côté


def test_lassistant_du_telephone_demande_aussi_le_jeton(poste) -> None:
    srv, _ = poste
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(srv, "/api/chat", {"messages": [{"role": "user", "content": "coucou"}]})
    assert e.value.code == 401


def test_le_flux_a_la_mode_ollama_se_relit_ligne_par_ligne() -> None:
    from valdar.interface import ollama_api

    lignes = ollama_api.stream_lines("Salut, ça va ?")
    morceaux = [json.loads(x) for x in lignes]
    assert all(m["model"] == "valdar:latest" for m in morceaux)
    assert morceaux[-1]["done"] is True
    assert "".join(m["message"]["content"] for m in morceaux) == "Salut, ça va ?"


def test_il_prend_le_dernier_mot_dolivier_pas_tout_lhistorique() -> None:
    from valdar.interface import ollama_api

    body = {"messages": [{"role": "system", "content": "tu es un assistant"},
                         {"role": "user", "content": "vieux message"},
                         {"role": "assistant", "content": "ok"},
                         {"role": "user", "content": "la vraie question"}]}
    assert ollama_api.last_user_message(body) == "la vraie question"
