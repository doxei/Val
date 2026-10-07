"""Import du vécu : export claude.ai → souvenirs, faits, jeu d'entraînement."""
import json

from valdar.memory import Episodic, Facts
from valdar.migrate import VecuImport
from valdar.migrate.vecu import memory_lines


def _export(root):
    convs = [
        {"uuid": "b", "name": "Imprimante", "created_at": "2026-03-02T10:00:00Z",
         "chat_messages": [
             {"sender": "human", "created_at": "2026-03-02T10:00:00Z",
              "text": "merci, t'es génial, le warping a disparu"},
             {"sender": "assistant", "created_at": "2026-03-02T10:00:05Z",
              "content": [{"type": "text", "text": "Content que ça marche !"}]}]},
        {"uuid": "a", "name": "Plantes", "created_at": "2026-01-15T20:00:00Z",
         "chat_messages": [
             {"sender": "human", "created_at": "2026-01-15T20:00:00Z",
              "text": "mes boutures de pothos pourrissent"},
             {"sender": "assistant", "created_at": "2026-01-15T20:00:10Z",
              "text": "Trop d'eau stagnante, change l'eau chaque semaine."}]},
        {"uuid": "vide", "name": "", "created_at": "2026-01-01T00:00:00Z",
         "chat_messages": [{"sender": "human", "created_at": None, "text": "perdu"}]},
    ]
    (root / "conversations.json").write_text(json.dumps(convs), encoding="utf-8")
    mem = root / "memories-1" / "memories"
    mem.mkdir(parents=True)
    (mem / "m.json").write_text(json.dumps({
        "memory_files": [{"content": "# Olivier\n- [stated] Olivier imprime sur une CR-10S "
                                     "sous Klipper\n- trop court\n"}],
        "conversations_memory": "**Travail**\nOlivier est autodidacte et touche à tout. Ok."
    }), encoding="utf-8")


def test_memory_lines_strip_tags_and_short_bits():
    lines = memory_lines({"memory_files": [{"content": "- [inferred] Il aime les plantes "
                                                       "carnivores\n- court"}],
                          "conversations_memory": "**Titre**\nUne phrase assez longue ici. Non."})
    assert lines == ["Il aime les plantes carnivores", "Une phrase assez longue ici."]


def test_vecu_import_is_ordered_private_and_idempotent(cfg, tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    _export(root)
    mem = Episodic(tmp_path / "episodes.db", cfg.episodic)
    facts = Facts(tmp_path / "facts.db")
    imp = VecuImport(root, cfg, mem, facts, tmp_path / "training", person="olivier")
    assert [c["uuid"] for c in imp.conversations()] == ["vide", "a", "b"]

    seen = []
    report = imp.run(on_progress=lambda i, n: seen.append((i, n)))
    assert seen[-1] == (3, 3)
    assert report[0].startswith("vécu : 2 conversation(s)") and "4 échanges" in report[0]
    assert "2 nouveau(x) fait(s)" in report[1]
    assert mem.count() == {"claude": 4}
    with mem._conn() as con:
        rows = con.execute("SELECT person, title, started FROM episodes ORDER BY started"
                           ).fetchall()
        pad = con.execute("SELECT p FROM turns WHERE speaker='Olivier' AND text LIKE 'merci%'"
                          ).fetchone()[0]
    assert [r[1] for r in rows] == ["Plantes", "Imprimante"]
    assert {r[0] for r in rows} == {"olivier"}          # privé : souvenirs d'Olivier seulement
    assert pad > 0                                      # un remerciement laisse un affect positif

    lines = (tmp_path / "training" / "olivier_messages.jsonl").read_text(encoding="utf-8")
    msgs = [json.loads(x) for x in lines.splitlines()]
    assert len(msgs) == 2 and all("Content" not in m["text"] for m in msgs)  # jamais Claude

    again = imp.run()
    assert again[0].startswith("vécu : 0 conversation(s)") and "0 nouveau" in again[1]
    assert mem.count() == {"claude": 4}


def test_vecu_missing_folder(cfg, tmp_path):
    imp = VecuImport(tmp_path / "absent", cfg, Episodic(tmp_path / "e.db", cfg.episodic),
                     Facts(tmp_path / "f.db"), tmp_path / "t")
    assert imp.run()[0].startswith("dossier introuvable")
