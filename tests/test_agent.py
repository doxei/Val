"""Agent : dialogue, outils, permissions, effets sur le cœur (phase 2)."""
from pathlib import Path

from conftest import prompt_of

from valdar.config.loader import Identity
from valdar.llm.backend import ChatResult, LLMError, ToolCall
from valdar.tools.registry import DANGEROUS, ELEVATED, Tool, params

OWNER = Identity(person="olivier", name="Olivier", role="owner", confidence=0.99)
UNKNOWN = Identity()


def call(name, **args):
    return ChatResult(content="", tool_calls=[ToolCall(name=name, arguments=args)])


def test_simple_reply_and_heart_hears(runtime_factory):
    rt = runtime_factory([ChatResult(content="Salut toi !")])
    before = rt.heart.needs["contact"] = 0.6
    reply = rt.handle("bravo, t'es le meilleur")
    assert reply.text == "Salut toi !"
    assert ("praise", 0.6) in reply.stimuli
    assert rt.heart.needs["contact"] < before, "un message doit nourrir le besoin de contact"
    assert "praise" in rt.heart.recent


def test_tool_call_executes_and_feeds_back(runtime_factory):
    rt = runtime_factory([call("creer_stock", nom="vis M3x12", quantite=40),
                          ChatResult(content="C'est noté, 40 vis.")])
    reply = rt.handle("j'ai 40 vis M3x12")
    assert reply.tools_used == ["creer_stock"]
    assert reply.text == "C'est noté, 40 vis."
    assert rt.stock.find("M3x12")[0]["qty"] == 40
    second = rt.llm.calls[1]["messages"]
    assert second[-1]["role"] == "tool" and "vis M3x12" in second[-1]["content"]
    assert "success" in rt.heart.recent


def test_failing_tool_does_not_crash_and_hurts_a_little(runtime_factory):
    rt = runtime_factory([call("ajouter_stock", nom="x"),  # argument manquant
                          ChatResult(content="Oups.")])
    reply = rt.handle("enlève une vis")
    assert reply.text == "Oups."
    assert "failure" in rt.heart.recent


def test_llm_down_gives_clear_message_and_clean_history(runtime_factory):
    def boom(**_):
        raise LLMError("connexion refusée")
    rt = runtime_factory(boom)
    reply = rt.handle("tu es là ?")
    assert reply.error and "injoignable" in reply.text
    assert rt.agent.history == []


def test_unknown_person_never_sees_or_runs_elevated_tools(runtime_factory):
    rt = runtime_factory([call("secret"), ChatResult(content="non.")])
    ran = []
    rt.registry.add(Tool("secret", "outil élevé", params(), lambda: ran.append(1), ELEVATED))
    rt.handle("lance l'outil secret", who=UNKNOWN)
    names = [t["function"]["name"] for t in rt.llm.calls[0]["tools"]]
    assert "secret" not in names
    assert ran == []
    assert "refusé" in rt.llm.calls[1]["messages"][-1]["content"]


def test_console_identity_is_not_trusted_for_elevated(runtime_factory):
    rt = runtime_factory([ChatResult(content="ok")])
    rt.registry.add(Tool("secret", "outil élevé", params(), lambda: "x", ELEVATED))
    rt.handle("salut")
    names = [t["function"]["name"] for t in rt.llm.calls[0]["tools"]]
    assert "secret" not in names, "la console seule ne prouve pas que c'est Olivier"


def test_dangerous_tool_needs_confirmation(runtime_factory):
    ran = []
    rt = runtime_factory([call("danger"), ChatResult(content="Fait.")])
    rt.registry.add(Tool("danger", "outil dangereux", params(), lambda: ran.append(1) or "ok",
                         DANGEROUS))
    first = rt.handle("fais le truc dangereux", who=OWNER)
    assert "confirmes" in first.text and ran == []
    second = rt.handle("oui vas-y", who=OWNER)
    assert ran == [1]
    assert second.tools_used[0] == "danger"


def test_dangerous_tool_cancelled_by_anything_else(runtime_factory):
    ran = []
    rt = runtime_factory([call("danger")])
    rt.registry.add(Tool("danger", "outil dangereux", params(), lambda: ran.append(1),
                         DANGEROUS))
    rt.handle("fais le truc", who=OWNER)
    reply = rt.handle("non attends", who=OWNER)
    assert ran == [] and "annule" in reply.text


def test_confirmation_from_someone_else_is_refused(runtime_factory):
    ran = []
    rt = runtime_factory([call("danger")])
    rt.registry.add(Tool("danger", "outil dangereux", params(), lambda: ran.append(1),
                         DANGEROUS))
    rt.handle("fais le truc", who=OWNER)
    rt.handle("oui", who=Identity(person="invite", name="Invité", role="guest",
                                  confidence=0.99))
    assert ran == []


def test_each_reply_costs_energy(runtime_factory):
    rt = runtime_factory([ChatResult(content="a"), ChatResult(content="b")])
    e0 = rt.heart.variables["energy"]
    rt.handle("un")
    rt.handle("deux")
    assert rt.heart.variables["energy"] < e0


def test_spontaneous_speech_has_no_tools(runtime_factory):
    rt = runtime_factory([ChatResult(content="Hé, ça fait un bail !")])
    reply = rt.agent.spontaneous("tu as envie de parler.")
    assert reply.text.startswith("Hé")
    assert rt.llm.calls[0]["tools"] is None


def test_memory_tool_and_recall_in_prompt(runtime_factory):
    rt = runtime_factory([call("se_souvenir", action="note",
                               quoi="Olivier imprime en PLA+ sur une plaque PEI"),
                          ChatResult(content="Retenu."),
                          ChatResult(content="Du PLA+.")])
    rt.handle("retiens que j'imprime en PLA+ sur une plaque PEI")
    rt.handle("j'imprime avec quel filament déjà ?")
    assert "PLA+" in prompt_of(rt.llm.calls[-1])


def test_une_action_sur_le_pc_hors_atelier_demande_un_oui_net(runtime_factory, tmp_path):
    """Écrire dans son atelier : libre. Écrire ailleurs : « oui » net, et de lui."""
    from valdar.config.loader import CorpsConfig
    from valdar.corps.outils import Corps, add_corps_tools

    cc = CorpsConfig(atelier=str(tmp_path / "atelier"), lecture=[str(tmp_path)],
                     ecriture=[str(tmp_path)], interdits=[], proteges_ecriture=[])
    corps = Corps(cc, tmp_path, tmp_path / "data", web=None)
    dehors = str(tmp_path / "dehors.txt")

    rt = runtime_factory([call("ecrire_fichier", chemin=dehors, contenu="x"),
                          ChatResult(content="C'est écrit.")])
    add_corps_tools(rt.registry, corps, who=lambda: "olivier")
    first = rt.handle("écris x dans dehors.txt", who=OWNER)
    assert "confirmes" in first.text and dehors.split("/")[-1] in first.text
    assert rt.agent.pending["strict"] and not Path(dehors).exists()
    rt.handle("mouais pourquoi pas", who=OWNER)       # pas un oui net : annulé
    assert not Path(dehors).exists()

    rt2 = runtime_factory([call("ecrire_fichier", chemin=dehors, contenu="x"),
                           ChatResult(content="C'est écrit.")])
    add_corps_tools(rt2.registry, corps, who=lambda: "olivier")
    rt2.handle("écris x dans dehors.txt", who=OWNER)
    rt2.handle("oui", who=OWNER)
    assert Path(dehors).read_text() == "x"


def test_dans_son_atelier_il_ecrit_sans_demander(runtime_factory, tmp_path):
    from valdar.config.loader import CorpsConfig
    from valdar.corps.outils import Corps, add_corps_tools

    cc = CorpsConfig(atelier=str(tmp_path / "atelier"), lecture=[str(tmp_path)],
                     ecriture=[str(tmp_path)], interdits=[], proteges_ecriture=[])
    corps = Corps(cc, tmp_path, tmp_path / "data", web=None)
    dedans = str(corps.zones.atelier / "notes.txt")
    rt = runtime_factory([call("ecrire_fichier", chemin=dedans, contenu="ok"),
                          ChatResult(content="C'est noté.")])
    add_corps_tools(rt.registry, corps, who=lambda: "olivier")
    reply = rt.handle("note ça chez toi", who=OWNER)
    assert "confirmes" not in reply.text
    assert Path(dedans).read_text() == "ok"
    assert any(r["outil"] == "ecrire_fichier" for r in corps.journal.tail(5))


def test_le_corps_nest_pas_offert_a_un_inconnu(runtime_factory, tmp_path):
    """Un invité ne voit même pas ces outils dans la liste."""
    from valdar.config.loader import CorpsConfig
    from valdar.corps.outils import Corps, add_corps_tools

    cc = CorpsConfig(atelier=str(tmp_path / "atelier"), lecture=[str(tmp_path)],
                     ecriture=[str(tmp_path)], interdits=[], proteges_ecriture=[])
    corps = Corps(cc, tmp_path, tmp_path / "data", web=None)
    rt = runtime_factory([ChatResult(content="salut")])
    add_corps_tools(rt.registry, corps, who=lambda: "olivier")
    rt.handle("salut", who=Identity(person="inconnu", confidence=0.9))
    noms = [t["function"]["name"] for t in rt.llm.calls[0]["tools"]]
    for n in ("ecrire_fichier", "terminal", "regarder_ecran", "cliquer"):
        assert n not in noms, n
