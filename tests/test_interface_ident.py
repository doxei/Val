"""Interface, foyer, reconnaissance (voix, visage), le chien sur la table, Kiwix."""
from __future__ import annotations

import json
import threading
import time
import urllib.request

import numpy as np
import pytest

from valdar.config.loader import IdentConfig, Identity, VisionConfig
from valdar.ident.store import Prints
from valdar.ident.voices import Enrollment, VoiceID
from valdar.relations.foyer import Foyer, slug
from valdar.vision.watch import Box, Watch, on_table

SR = 16000


# ------------------------------------------------------------------ foyer
def test_foyer_members_animals_and_identity(tmp_path):
    owner = Identity(person="olivier", name="Olivier", role="owner", confidence=0.8)
    f = Foyer(tmp_path / "foyer.json", owner)
    f.add_member("Océane", "ma femme")
    f.add_member("Héloïse", "ma fille", mineur=True)
    f.add_animal("Sony", "chien")
    assert slug("Héloïse") == "heloise"
    again = Foyer(tmp_path / "foyer.json", owner)          # relu depuis le disque
    assert [m["id"] for m in again.members] == ["olivier", "oceane", "heloise"]
    who = again.identity("heloise", 0.7)
    assert who.name == "Héloïse" and who.minor and who.role == "family"
    assert again.identity("olivier", 0.9).role == "owner"
    assert again.dog()["nom"] == "Sony"
    block = again.block()
    assert "Océane — pour Olivier : « ma femme »" in block and "Sony, le chien" in block
    assert not again.remove_member("olivier")              # le créateur reste
    assert again.remove_member("oceane")


# ------------------------------------------------------------------ voix
class FakeEmbedder:
    """Une « voix » = une fréquence ; l'empreinte = un vecteur propre à cette fréquence."""

    def embed(self, samples):
        x = np.asarray(samples, dtype=np.float32)
        spec = np.abs(np.fft.rfft(x[: SR]))[:2000]
        v = np.add.reduceat(spec, np.arange(0, 2000, 50)).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


def voice(freq, seconds=4.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(len(t))
            ).astype(np.float32)


def test_voice_enroll_identify_and_unsure(tmp_path):
    prints = Prints(tmp_path / "e.db")
    vid = VoiceID(IdentConfig(voice_threshold=0.8, voice_margin=0.05), prints, FakeEmbedder())
    assert vid.enroll("olivier", voice(140, 12)) >= 3
    assert vid.enroll("heloise", voice(300, 12)) >= 3
    g = vid.identify(voice(140, 3, seed=5))
    assert g.person == "olivier" and g.score > 0.9
    assert vid.identify(voice(300, 3, seed=6)).person == "heloise"
    assert vid.identify(voice(900, 3)).person is None        # inconnu : pas de devinette
    assert vid.identify(voice(140, 0.4)).person is None      # trop court
    c = vid.confidence(g)
    assert 0.55 <= c <= 0.85                                  # la voix seule < outils élevés
    assert prints.forget("olivier") > 0
    assert "olivier" not in prints.counts("voix")


def test_enrollment_accumulates_until_enough():
    e = Enrollment("oceane", 5.0)
    assert not e.add(np.zeros(2 * SR, np.float32))
    assert e.add(np.zeros(3 * SR, np.float32)) and e.progress == 1.0
    assert len(e.audio()) == 5 * SR


def test_prints_keep_only_vectors(tmp_path):
    import sqlite3

    p = Prints(tmp_path / "e.db")
    p.add("voix", "olivier", np.ones(8))
    con = sqlite3.connect(tmp_path / "e.db")
    cols = [r[1] for r in con.execute("PRAGMA table_info(prints)")]
    assert cols == ["id", "kind", "person", "vec", "t", "learned"]   # ni audio ni image


# ------------------------------------------------------------------ visages
class FakeFaces:
    def __init__(self, vec):
        self.vec = vec

    def faces(self, frame):
        return [((10, 10, 80, 80), 0.9, self.vec)] if self.vec is not None else []


def test_face_enroll_and_recognize(tmp_path):
    from valdar.ident.faces import FaceID

    prints = Prints(tmp_path / "e.db")
    a, b = np.eye(16, dtype=np.float32)[0], np.eye(16, dtype=np.float32)[1]
    fid = FaceID(IdentConfig(), prints, FakeFaces(a))
    assert fid.enroll_shot("gabrielle", np.zeros((4, 4, 3)))
    fid.engine = FakeFaces(b)
    fid.enroll_shot("bruno", np.zeros((4, 4, 3)))
    fid.engine = FakeFaces(a + 0.1 * b)
    assert fid.look(np.zeros((4, 4, 3)))[0].person == "gabrielle"


# ------------------------------------------------------------------ le chien
def test_dog_on_table_geometry():
    table = Box("dining table", 0.9, 100, 300, 500, 450)
    assert on_table(Box("dog", 0.9, 200, 180, 320, 330), table)      # pattes sur le plateau
    assert not on_table(Box("dog", 0.9, 200, 350, 320, 470), table)  # sous la table
    assert not on_table(Box("dog", 0.9, 600, 180, 700, 330), table)  # à côté


class FakeDetector:
    def __init__(self):
        self.boxes_out: list[Box] = []

    def boxes(self, frame):
        return self.boxes_out


def test_dog_rule_scolds_only_when_alone_with_cooldown():
    det = FakeDetector()
    clock = [0.0]
    scolded = []
    w = Watch(VisionConfig(table_frames=2, table_cooldown_seconds=60), lambda: np.zeros(1),
              det, None, on_dog_table=lambda: scolded.append(clock[0]), clock=lambda: clock[0])
    table = Box("dining table", 0.9, 100, 300, 500, 450)
    dog_up = Box("dog", 0.9, 200, 180, 320, 330)
    det.boxes_out = [table, dog_up, Box("person", 0.9, 0, 0, 50, 200)]
    w.step()
    w.step()
    assert scolded == []                       # quelqu'un est là : c'est à lui de gérer
    det.boxes_out = [table, dog_up]
    w.step()
    assert scolded == []                       # une seule image ne suffit pas
    clock[0] = 1
    w.step()
    assert scolded == [1]
    clock[0] = 30
    w.step()
    assert scolded == [1]                      # pas de harcèlement
    clock[0] = 70
    w.step()
    assert scolded == [1, 70]
    assert w.dog_seen == 70


# ------------------------------------------------------------------ session
@pytest.fixture()
def session(runtime_factory):
    from valdar.interface.session import Session

    rt = runtime_factory()
    s = Session(rt.cfg, rt, None)
    yield s
    s._stop.set()


def test_session_streams_and_publishes(session):
    seen = []
    session.bus.subscribe(lambda k, d: seen.append(k))
    session.answer("salut")
    assert seen[0] == "user" and seen[-1] == "reply"
    assert [m["kind"] for m in session.bus.log] == ["user", "reply"]


def test_who_spoke_uses_voice_then_face(session, tmp_path):
    session.foyer.add_member("Héloïse", "ma fille", True)
    session.prints = Prints(tmp_path / "e.db")
    session.voice_id = VoiceID(IdentConfig(voice_threshold=0.8), session.prints,
                               FakeEmbedder())
    session.voice_id.enroll("heloise", voice(300, 12))
    session.voice_id.enroll("olivier", voice(140, 12))
    who, info = session.who_spoke(voice(300, 3, seed=9))
    assert who.person == "heloise" and who.minor and who.confidence < 0.9
    session.watch = type("W", (), {"seen": {"heloise": time.time()}})()
    who, _ = session.who_spoke(voice(300, 3, seed=9))
    assert who.confidence == 0.95                          # voix et visage d'accord
    who, _ = session.who_spoke(voice(900, 3))
    assert who.person == "heloise" and who.confidence == 0.6   # voix inconnue, une en vue
    session.watch = None
    who, _ = session.who_spoke(voice(900, 3))
    assert who.person is None                              # personne : il ne devine pas


def test_dog_table_scold_uses_the_dogs_name(session):
    session.foyer.add_animal("Sony", "chien")
    session.on_dog_table()
    ev = [m for m in session.bus.log if m["kind"] == "event"][-1]
    assert ev["data"]["kind"] == "chien" and "Sony" in ev["data"]["text"]


def test_home_block_in_prompt(session):
    session.foyer.add_member("Océane", "ma femme")
    block = session._home_block("")[0]
    assert "TON FOYER" in block and "Océane" in block


# ------------------------------------------------------------------ réglages
def test_settings_apply_live_and_persist(session, tmp_path):
    from valdar.interface.settings import Settings

    st = Settings(session.cfg, tmp_path / "reglages.json")
    r = st.set("voice.character.high_shelf_db", "-4")
    assert r["value"] == -4.0 and session.cfg.voice.character.high_shelf_db == -4.0
    assert json.loads((tmp_path / "reglages.json").read_text())["voice"]["character"][
        "high_shelf_db"] == -4.0
    with pytest.raises(ValueError):
        st.set("ears.vad_threshold", 5)
    with pytest.raises(KeyError):
        st.set("permissions.owner_roles", ["tout le monde"])   # pas réglable d'ici
    assert st.set("vision.camera", "1")["value"] == 1
    assert {s["path"] for s in st.listing()} >= {"llm.model", "ident.voice_margin"}


def test_overrides_loaded_and_bad_ones_ignored(tmp_path):
    import shutil

    from valdar.config import load
    from valdar.config.loader import overrides_path

    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    src = load().root / "config"
    for f in ("valdar.yaml", "self_model.yaml"):
        shutil.copy(src / f, root / "config" / f)
    ov = overrides_path(root)
    ov.parent.mkdir()
    ov.write_text(json.dumps({"ears": {"vad_threshold": 0.35}}))
    import valdar.config.loader as L

    old = L.DEFAULT_CONFIG_PATH
    L.DEFAULT_CONFIG_PATH = root / "config" / "valdar.yaml"
    try:
        assert L.load().ears.vad_threshold == 0.35
        ov.write_text(json.dumps({"ears": {"vad_threshold": 9}}))
        assert L.load().ears.vad_threshold == 0.5          # invalide : configuration de base
    finally:
        L.DEFAULT_CONFIG_PATH = old


# ------------------------------------------------------------------ serveur
def test_server_serves_page_api_and_events(session):
    from valdar.interface.server import Server

    srv = Server(session, "127.0.0.1", 0)
    url = srv.start()
    try:
        html = urllib.request.urlopen(url).read().decode()
        assert "<title>Valdar</title>" in html and "face.js" in html
        st = json.loads(urllib.request.urlopen(url + "api/state").read())
        assert "emotion" in st and "pad" in st
        foyer = json.loads(urllib.request.urlopen(url + "api/foyer").read())
        assert foyer["membres"][0]["role"] == "owner"
        got: list[bytes] = []

        def listen():
            with urllib.request.urlopen(url + "api/events", timeout=5) as r:
                for line in r:
                    got.append(line)
                    if line.startswith(b"event: reply"):
                        return

        t = threading.Thread(target=listen, daemon=True)
        t.start()
        time.sleep(0.3)
        req = urllib.request.Request(url + "api/message", data=json.dumps(
            {"text": "salut"}).encode(), headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read())["ok"]
        t.join(5)
        assert any(line.startswith(b"event: reply") for line in got)
        bad = urllib.request.Request(url + "api/reglages", data=json.dumps(
            {"path": "nimporte", "value": 1}).encode())
        assert json.loads(urllib.request.urlopen(bad).read())["ok"] is False
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url + "../config/valdar.yaml")
    finally:
        srv.stop()


# ------------------------------------------------------------------ Kiwix
def test_kiwix_latest_and_clean():
    from valdar.knowledge.kiwix import clean, latest

    listing = ('<a href="wikipedia_fr_all_nopic_2024-11.zim">x</a>'
               '<a href="wikipedia_fr_all_nopic_2025-06.zim">x</a>'
               '<a href="wikipedia_fr_all_maxi_2025-09.zim">x</a>')
    assert latest(listing, "wikipedia_fr_all_nopic_") == "wikipedia_fr_all_nopic_2025-06.zim"
    assert latest(listing, "vikidia_") is None
    page = ("<html><script>x()</script><nav>menu</nav><h1>Chien</h1><p>Le chien"
            "<sup>[1]</sup> est un &ecirc;tre fid&egrave;le.</p></html>")
    assert clean(page) == "Chien\nLe chien est un être fidèle."


def test_kiwix_client_search_and_read():
    import httpx

    from valdar.knowledge.kiwix import Kiwix

    def handler(req):
        if req.url.path == "/catalog/v2/entries":
            return httpx.Response(200, text=(
                '<feed xmlns="http://www.w3.org/2005/Atom"><entry><name>vikidia_fr</name>'
                "<title>Vikidia</title><language>fra</language></entry></feed>"))
        if req.url.path == "/search":
            assert req.url.params["books.name"] == "vikidia_fr"
            return httpx.Response(200, text=(
                "<rss><channel><item><title>Chien</title><link>/content/vikidia_fr/A/Chien"
                "</link><description>Le &lt;b&gt;chien&lt;/b&gt; est…</description></item>"
                "</channel></rss>"))
        return httpx.Response(200, text="<p>Le chien aboie.</p>")

    k = Kiwix("http://kiwix", client=httpx.Client(transport=httpx.MockTransport(handler)))
    hits = k.search("chien")
    assert hits[0]["title"] == "Chien" and hits[0]["book"] == "vikidia_fr"
    assert k.read(hits[0]["path"]) == "Le chien aboie."


def test_runtime_has_offline_library_tools(session):
    names = set(session.rt.registry.tools)
    assert {"chercher_savoir", "lire_article"} <= names
    out = session.rt.registry.tools["chercher_savoir"].fn("chien")
    assert "injoignable" in out or "rien trouvé" in out
