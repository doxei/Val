"""Relations, modèle de l'autre, leçons par personne, « oublie-moi » (phase 7)."""
from valdar.config.loader import Identity
from valdar.llm.backend import ChatResult, ToolCall
from valdar.relations import Relations

T0 = 1_800_000_000.0
OLI = Identity(person="olivier", name="Olivier", role="owner", confidence=0.8)
ZOE = Identity(person="zoe", name="Zoé", role="guest", confidence=0.95)


def test_encounters_affection_and_topics(cfg, tmp_path):
    r = Relations(cfg, tmp_path / "p.db")
    assert r.observe(OLI, "salut mon pote", T0)["new_encounter"]
    assert not r.observe(OLI, "ma bouture de pothos pourrit", T0 + 60)["new_encounter"]
    out = r.observe(OLI, "merci, t'es génial", T0 + 3 * 3600)
    p = out["person"]
    assert out["new_encounter"] and p.encounters == 2 and p.messages == 3
    assert p.affection > 0 and p.topics == {"plantes": 1}
    assert r.observe(Identity(), "bonjour", T0)["person"] is None   # inconnu : rien gardé


def test_affection_fades_without_contact(cfg, tmp_path):
    r = Relations(cfg, tmp_path / "p.db")
    for i in range(30):
        r.observe(OLI, "merci mon pote", T0 + i)
    warm = r.get("olivier").affection
    r.observe(OLI, "bon", T0 + 120 * 86400)              # deux demi-vies plus tard
    assert abs(r.get("olivier").affection - warm / 4) < 0.02


def test_other_model_reads_the_recent_tone(cfg, tmp_path):
    r = Relations(cfg, tmp_path / "p.db")
    noon = 1_800_000_000.0 - (1_800_000_000 % 86400) + 12 * 3600
    for i in range(5):
        r.observe(OLI, "putain ça marche pas, ça m'énerve", noon + i * 30)
    words = r.state_words(r.get("olivier"), noon + 200)
    assert "agacé" in words or "à plat" in words
    assert r.state_words(r.get("olivier"), noon + 10 * 3600) == ""   # trop vieux : on ne sait pas


def test_lessons_are_per_person_never_shared(cfg, tmp_path):
    r = Relations(cfg, tmp_path / "p.db")
    r.observe(OLI, "salut", T0)
    r.observe(ZOE, "bonjour", T0)
    assert r.learn("olivier", "aime", "les vannes et les punchlines") == "noté, pour toi."
    assert r.learn("olivier", "aime", "les vannes et les punchlines") == "je le savais déjà."
    assert r.learn("zoe", "prefere", "qu'on la vouvoie").startswith("noté")
    assert r.learn("zoe", "n'importe", "x").startswith("genre inconnu")
    oli, zoe = r.block(OLI, T0), r.block(ZOE, T0)
    assert "vannes" in oli and "vouvoie" not in oli
    assert "vouvoie" in zoe and "vannes" not in zoe
    assert "ne généralise jamais" in oli


def test_presence_warms_or_puts_on_guard(cfg, make_heart, tmp_path):
    r = Relations(cfg, tmp_path / "p.db")
    for i in range(40):
        r.observe(OLI, "merci mon pote, t'es génial", T0 + i)
        r.observe(ZOE, "putain t'es nul, ça m'énerve", T0 + i)
    h = make_heart()
    assert r.feel_presence(h, r.get("olivier")) == "warmth"
    assert r.feel_presence(h, r.get("zoe")) == "concern"
    assert "prudent" in r.block(ZOE, T0)


def test_runtime_tracks_who_speaks_and_forgets_on_request(runtime_factory):
    oublie = [ChatResult(content="", tool_calls=[ToolCall("oublie_moi", {})]),
              ChatResult(content="c'est fait.")]
    rt = runtime_factory(script=[ChatResult(content="salut Zoé"),
                                 ChatResult(content="", tool_calls=[
                                     ToolCall("retenir_sur_toi",
                                              {"genre": "aime", "texte": "les chats"})]),
                                 ChatResult(content="noté"), *oublie])
    rt.handle("bonjour", ZOE)
    rt.handle("j'adore les chats", ZOE)
    assert [le["text"] for le in rt.relations.lessons("zoe")] == ["les chats"]
    assert rt.relations.lessons("olivier") == []            # rangé chez Zoé, pas chez Olivier
    assert "TA RELATION AVEC ZOÉ" in rt.llm.calls[-1]["system"]

    ask = rt.handle("oublie-moi", ZOE)
    assert "confirmes" in ask.text and rt.relations.get("zoe") is not None
    rt.handle("oui", OLI)                                    # pas la même personne : annulé
    assert rt.relations.get("zoe") is not None
    rt.llm.script[:0] = [ChatResult(content="", tool_calls=[ToolCall("oublie_moi", {})])]
    rt.handle("oublie-moi", ZOE)
    rt.llm.script[:0] = [ChatResult(content="c'est fait.")]
    done = rt.handle("oui", ZOE)
    assert "oublie_moi" in done.tools_used and rt.agent.history == []
    assert rt.relations.get("zoe") is None and rt.relations.lessons("zoe") == []
    with rt.memory._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM episodes WHERE person='zoe'").fetchone()[0] == 0
