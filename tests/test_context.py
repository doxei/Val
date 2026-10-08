"""Contexte de chaque tour : la mémoire à la demande, le monde quand il change (latence)."""
from conftest import prompt_of

from valdar.llm.backend import ChatResult, ToolCall

T0 = 1_790_000_000.0


def test_memory_and_facts_stay_out_unless_asked(runtime_factory):
    rt = runtime_factory([ChatResult(content="salut !")])
    rt.memory.log_turn("Olivier", "le plateau en verre a explosé ce matin", T0,
                       pad=(-0.8, 0.4, -0.5))
    rt.facts.remember("Olivier préfère le PETG noir", person="olivier")
    rt.handle("salut mon pote")
    p = prompt_of(rt.llm.calls[0])
    assert "explosé" not in p and "PETG" not in p        # rien n'est récité d'office
    assert "fouiller_memoire" in rt.llm.calls[0]["system"]  # mais il sait où chercher
    assert any(t["function"]["name"] == "fouiller_memoire" for t in rt.llm.calls[0]["tools"])


def test_fouiller_memoire_finds_and_relives(runtime_factory):
    rt = runtime_factory([
        ChatResult(content="", tool_calls=[ToolCall("fouiller_memoire",
                                                    {"sujet": "plateau en verre"})]),
        ChatResult(content="Ah oui, le plateau qui a éclaté…"),
    ])
    rt.memory.log_turn("Olivier", "le plateau en verre a explosé ce matin, je suis effondré",
                       T0, pad=(-0.8, 0.4, -0.5))
    rt.facts.remember("le plateau en verre est un Creality 310 mm", person="olivier")
    before = rt.heart.variables["cortisol"]
    reply = rt.handle("tu te souviens du plateau en verre ?")
    assert "fouiller_memoire" in reply.tools_used
    result = rt.agent.history[-2]["content"]          # ce que l'outil a rendu au modèle
    assert "explosé" in result and "Creality" in result
    assert rt.heart.variables["cortisol"] > before or any(
        p["target"] == "cortisol" for p in rt.heart.pending)   # le souvenir pince un peu


def test_world_full_first_then_hourly_or_on_change(runtime_factory):
    rt = runtime_factory()
    world = {"lines": ["heure : jeudi 12:00", "imprimante : prête", "prochain rappel : linge"]}
    rt.agent.world = lambda: list(world["lines"])
    a = rt.agent
    assert a._world_now(T0) == world["lines"]                 # premier tour : tout
    assert a._world_now(T0 + 60) == ["heure : jeudi 12:00"]   # ensuite : juste l'heure
    world["lines"][1] = "imprimante : impression en cours 3 %"
    assert len(a._world_now(T0 + 120)) == 3                   # ça a changé : tout de suite
    assert len(a._world_now(T0 + 180)) == 1
    every = rt.cfg.context.world_every_seconds
    assert len(a._world_now(T0 + 120 + every)) == 3           # une fois par heure


def test_auto_modes_still_available(runtime_factory):
    rt = runtime_factory([ChatResult(content="ok")])
    rt.cfg.context.auto_facts = True
    rt.facts.remember("Olivier préfère le PETG noir", person="olivier")
    rt.handle("quel filament je préfère déjà ?")
    assert "PETG" in prompt_of(rt.llm.calls[0])
