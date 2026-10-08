"""Ligne de commande Valdar.

  valdar chat         parler à Valdar (cœur, mémoire, outils, Gemma 4 via Ollama)
                      --voix : il répond à voix haute ; --ecoute : il écoute au micro
                      --camera : il regarde la pièce ; --interface : visage et onglets
  valdar voix         faire dire une phrase à Valdar (test de la voix, mesures)
  valdar import-ancien
                      reprendre les données de l'ancienne installation, voix comprise
                      (lecture seule de ce côté-là ; dossier : migration.dossier_ancien)
  valdar heart        cœur seul, en temps réel (Ctrl+C pour sauvegarder et quitter)
  valdar status       état actuel du cœur (sans le modifier)
  valdar sim          simulation accélérée de N jours (n'écrit jamais dans la vraie base)
  valdar check        critères d'acceptation de la phase 1
"""
from __future__ import annotations

import argparse
import sys
import time

from valdar.config import load
from valdar.heart import Heart
from valdar.heart.store import Store
from valdar.sim.run import acceptance, report_to_text, run_sim
from valdar.workspace import Initiative

INITIATIVE_KEY = "initiative:state"


def _status_text(heart: Heart) -> str:
    s = heart.sample()
    felt = heart.felt()
    lines = [
        f"Émotion : {s['emotion']} (intensité {s['intensity']:.2f})"
        + (f", système dominant : {s['dominant']}" if s["dominant"] else ""),
        f"Humeur  : {s['mood_label']}",
        f"Éveillé : {'oui' if s['awake'] else 'non, il dort'}"
        f"   énergie {s['variables']['energy']:.2f}   cœur {s['bpm']} bpm",
        "Besoins : " + ", ".join(f"{k} {v:.2f}" for k, v in s["needs"].items())
        + f", repos {s['need_rest']:.2f}",
        "Ressenti :",
    ]
    lines += [f"  - {t}" for t in felt.values()]
    return "\n".join(lines)


def _cmd_heart(args: argparse.Namespace) -> int:
    cfg = load()
    heart = Heart.load_latest(cfg, profile=args.profile)
    init = Initiative(cfg.initiative)
    if heart.store is not None:
        saved = heart.store.load(INITIATIVE_KEY)
        if saved:
            init.load_dict(saved)
    print(f"Cœur de Valdar démarré (profil {heart.profile_name}). Ctrl+C pour arrêter.")
    print(_status_text(heart), flush=True)
    next_status = time.time() + args.every
    try:
        while True:
            now = time.time()
            if now - heart.now > 2 * cfg.heart.max_time_step:
                heart.catch_up(now)
            else:
                heart.step(now - heart.now)
            event = init.check(heart)
            if event is not None:
                verbe = "a envie de te parler" if event["kind"] == "talk" else \
                    "a envie de découvrir quelque chose"
                print(f"\n>>> Valdar {verbe} (émotion : {event['emotion']})", flush=True)
            if now >= next_status:
                b = heart.brief()
                print(f"[{time.strftime('%H:%M:%S')}] {b['emotion']}, humeur {b['mood']}",
                      flush=True)
                next_status = now + args.every
            time.sleep(max(0.05, cfg.heart.tick_seconds - (time.time() - now)))
    except KeyboardInterrupt:
        heart.save()
        if heart.store is not None:
            heart.store.save(INITIATIVE_KEY, init.to_dict())
        heart.close()
        print("\nSauvegardé.")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    cfg = load()
    store = Store(cfg.storage_path(cfg.storage.db))
    snap = store.load("heart:state")
    store.close()
    if snap is None:
        print("Aucun état sauvegardé : lance d'abord « valdar heart ».")
        return 1
    heart = Heart(cfg, profile=snap.get("profile"), persist=False)
    if not heart.restore(snap):
        print("L'état sauvegardé vient d'une ancienne version : il sera ignoré au prochain "
              "démarrage de « valdar heart ».")
        return 1
    heart.catch_up()
    print(_status_text(heart))
    return 0


def _cmd_sim(args: argparse.Namespace) -> int:
    rep = run_sim(load(), days=args.days, seed=args.seed, profile=args.profile)
    print(report_to_text(rep))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    results = acceptance(load(), seed=args.seed)
    ok = True
    for name, passed, detail in results:
        ok &= passed
        print(f"[{'OK ' if passed else 'ÉCHEC'}] {name} — {detail}")
    print("Phase 1 validée." if ok else "Phase 1 NON validée.")
    return 0 if ok else 1


CHAT_HELP = """Commandes : /etat (mon état), /oreilles (ce que j'entends), /silence (plus
d'initiatives), /parle (initiatives réactivées), /muet et /voix (couper ou rendre la voix),
/quitter. Tout le reste, je le prends comme un message."""


def _start_voice(cfg):
    """Démarre la voix en arrière-plan. Retourne None si elle est impossible (texte seul)."""
    from valdar.voice import Speaker, make_tts

    try:
        tts = make_tts(cfg)
    except Exception as exc:
        print(f"(voix désactivée : {exc})")
        return None
    missing = tts.check_files()
    if missing:
        print("(voix absente : " + ", ".join(missing) + " — lance tools\\valdar_voix.bat)")
        return None
    print("(je charge ma voix, une vingtaine de secondes ; on peut parler en attendant)")
    return Speaker(cfg.voice, tts).start(preload=True)


def _cmd_chat(args: argparse.Namespace) -> int:
    from valdar.interface.session import Session
    from valdar.runtime import Runtime

    cfg = load()
    rt = Runtime(cfg)
    llm = rt.llm
    if hasattr(llm, "available") and not llm.available():
        print(f"Ollama ne répond pas sur {cfg.llm.url}. Lance l'application Ollama puis "
              "relance-moi (tools\\valdar.bat le fait tout seul).")
        rt.stop()
        return 1
    if hasattr(llm, "has_model") and not llm.has_model():
        print(f"Le modèle {cfg.llm.model} n'est pas dans Ollama. Tape : ollama pull "
              f"{cfg.llm.model}")
        rt.stop()
        return 1
    speaker = _start_voice(cfg) if args.voix and cfg.voice.enabled else None
    if cfg.kiwix.enabled:
        print(f"({rt.start_kiwix()})")
    session = Session(cfg, rt, speaker)
    _console(session, args.debug)
    session.start()
    if args.ecoute:
        _start_ident(session)
        _start_ears(session, args.debug)
    if args.camera and cfg.vision.enabled:
        _start_vision(session)
    server = _start_interface(session) if args.interface and cfg.interface.enabled else None
    with rt.lock:
        b = rt.heart.brief()
    print(f"Valdar est là ({b['emotion']}, humeur {b['mood']}). {CHAT_HELP}")
    warned = False
    try:
        while True:
            try:
                text = input("Toi > ").strip()
            except EOFError:
                if server is None:
                    break
                session._stop.wait()       # lancé sans console : l'interface suffit
                break
            if speaker is not None:
                speaker.interrupt()          # on me parle : je me tais
                problem = speaker.load_error if speaker.ready.is_set() else None
                if problem and not warned:
                    print(f"(ma voix n'a pas pu démarrer : {problem})")
                    warned = True
            if not text:
                continue
            low = text.lower()
            if low in ("/quitter", "/quit", "/exit"):
                break
            if low == "/muet":
                session.muted = True
                print("(voix coupée)")
                continue
            if low == "/voix":
                session.muted = False
                print("(voix rétablie)" if speaker else "(lance « valdar chat --voix »)")
                continue
            if low == "/oreilles":
                print(_ears_text(rt))
                continue
            if low == "/etat":
                with rt.lock:
                    print(_status_text(rt.heart))
                continue
            if low == "/silence":
                rt.initiative.silence(True)
                print("(initiatives coupées)")
                continue
            if low == "/parle":
                rt.initiative.silence(False)
                print("(initiatives réactivées)")
                continue
            print("Valdar > ", end="", flush=True)
            reply = session.answer(text, show=lambda p: print(p, end="", flush=True))
            print()
            if reply.tools_used and args.debug:
                print(f"  [outils : {', '.join(reply.tools_used)}]")
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.stop()
        session.stop()
        print("\nÀ plus. (état sauvegardé)")
    return 0


def _console(session, debug: bool) -> None:
    """La console affiche ce qui se passe (ce que l'interface affiche aussi)."""
    def on(kind: str, d: dict) -> None:
        if kind == "spontaneous":
            print(f"\nValdar (de lui-même) > {d['text']}\nToi > ", end="", flush=True)
        elif kind == "event" and (d.get("kind") != "pensee" or debug):
            print(f"\n[{d.get('kind')}] {d.get('text')}\nToi > ", end="", flush=True)
        elif kind == "heard":
            sure = f", {round(d['confidence'] * 100)} %" if d.get("confidence") else ""
            print(f"\n(entendu, {d.get('who')}{sure}) {d['text']}", flush=True)
        elif kind == "enroll" and d.get("done"):
            print(f"\n({d['kind']} de {d['person']} apprise : {d['prints']} empreintes)\n"
                  "Toi > ", end="", flush=True)

    session.bus.subscribe(on)


def _start_ident(session) -> None:
    """Reconnaissance de la voix (sherpa-onnx). Sans elle, il croit parler à Olivier."""
    cfg = session.cfg
    if not cfg.ident.enabled:
        return
    try:
        from valdar.ident.download import fetch
        from valdar.ident.store import Prints
        from valdar.ident.voices import SherpaEmbedder, VoiceID

        session.prints = Prints(cfg.storage_path(cfg.ident.db))
        model = fetch(cfg.repo_path(cfg.ident.voice_model), cfg.ident.voice_urls)
        session.voice_id = VoiceID(cfg.ident, session.prints,
                                   SherpaEmbedder(str(model), threads=2))
        n = len(session.prints.counts("voix"))
        print(f"(je reconnais les voix : {n} personne(s) apprise(s) ; onglet Foyer pour en "
              "ajouter)")
    except ImportError:
        print("(reconnaissance des voix absente : installe l'extra [identite])")
    except Exception as exc:
        print(f"(reconnaissance des voix impossible : {exc})")


def _start_vision(session) -> None:
    cfg = session.cfg
    try:
        from valdar.ident.download import fetch
        from valdar.ident.faces import FaceEngine, FaceID
        from valdar.ident.store import Prints
        from valdar.vision.watch import Camera, Watch, YoloDetector

        cam = Camera(cfg.vision.camera)
    except ImportError:
        print("(caméra : installe l'extra [vision])")
        return
    except Exception as exc:
        print(f"(caméra impossible : {exc})")
        return
    if session.prints is None:
        session.prints = Prints(cfg.storage_path(cfg.ident.db))
    faces = detector = None
    try:
        det = fetch(cfg.repo_path(cfg.ident.face_detector), [cfg.ident.face_detector_url],
                    min_bytes=50_000)
        rec = fetch(cfg.repo_path(cfg.ident.face_model), [cfg.ident.face_model_url])
        faces = FaceID(cfg.ident, session.prints, FaceEngine(str(det), str(rec)))
        session.face_id = faces
    except Exception as exc:
        print(f"(visages impossibles : {exc})")
    try:
        detector = YoloDetector(cfg.vision.detector, cfg.vision.min_confidence)
    except Exception as exc:
        print(f"(détection du chien impossible : {exc} — installe l'extra [vision])")
    w = Watch(cfg.vision, cam.frame, detector, faces, on_scene=session.on_scene,
              on_dog_table=session.on_dog_table)
    session.watch = w
    w.start()
    print("(je regarde la pièce : visages" + (", chien et table" if detector else "") + ")")


def _start_interface(session):
    from valdar.interface.server import Server
    from valdar.interface.window import open_window

    ic = session.cfg.interface
    try:
        server = Server(session, ic.host, ic.port)
    except OSError as exc:
        print(f"(interface impossible : {exc} — Valdar tourne déjà ?)")
        return None
    url = server.start()
    print(f"(interface : {url})")
    try:
        session.window_stop = open_window(url, ic.screen, ic.kiosk,
                                          profile=session.cfg.storage_path("edge"))
    except Exception as exc:
        print(f"(fenêtre impossible à ouvrir : {exc} — ouvre {url} dans Edge)")
    return server


def _ears_text(rt) -> str:
    gate = getattr(rt, "gate", None)
    if gate is None:
        return "(je n'écoute pas : lance tools\\valdar_ecoute.bat)"
    st = gate.status()
    amb = rt.ambient
    lines = [f"segments de parole : {st['segments']} ; éveils : {st['woken']} ; ignorés (pas "
             f"mon nom) : {st['ignored']} ; transcrits : {st['transcribed']} ; rejetés : "
             f"{st['junk']} ; conversation ouverte : {'oui' if st['engaged'] else 'non'}"]
    if amb is not None and amb.base is not None:
        lines.append(f"fond sonore : {amb.base:.0f} dB ; niveau du moment : {amb.fast:.0f} dB"
                     f" ; sursauts : {amb.stats['startle']}")
    lines += ["- " + n for n in st["notes"][-6:]]
    return "\n".join(lines)


def _start_ears(session, debug: bool):
    """Écoute au micro (phase 3). Retourne le micro, ou None si l'écoute est impossible."""
    import queue
    import threading

    cfg, rt, speaker = session.cfg, session.rt, session.speaker
    try:
        from valdar.ears import Gate
        from valdar.ears.mic import Mic
        from valdar.ears.models import build
    except ImportError as exc:
        print(f"(écoute impossible : {exc} — installe l'extra [ears])")
        return None
    print("(je charge mes oreilles : détection de parole et transcription…)", flush=True)
    try:
        vad, wake, stt = build(cfg.ears, cfg.repo_path, rt.llm)
    except Exception as exc:
        print(f"(écoute impossible : {exc})")
        return None
    heard_q: queue.Queue = queue.Queue()
    # Ce qui compte pour les oreilles : le son qui sort vraiment des haut-parleurs.
    speaking = (getattr(speaker, "playing", speaker.speaking) if speaker is not None
                else None)
    ambient = rt.make_ambient(cfg.ears.frame_ms / 1000, speaking) if cfg.ambient.enabled \
        else None

    def note(text: str) -> None:
        if debug or "adressée" in text:
            print(f"\n(oreilles) {text}\nToi > ", end="", flush=True)

    gate = Gate(cfg.ears, vad, wake, stt, heard_q.put,
                on_barge_in=(speaker.interrupt if speaker is not None else None),
                speaking=speaking, on_frame=(ambient.feed if ambient is not None else None),
                on_note=note)
    rt.gate = gate
    if debug and hasattr(wake, "peek"):
        wake.peek = True

    def answer() -> None:
        while True:
            h = heard_q.get()
            if h is None:
                return
            print("Valdar > ", end="", flush=True)
            try:
                session.on_heard(h, show=lambda p: print(p, end="", flush=True))
            except Exception as exc:          # une erreur ne doit jamais le rendre sourd
                import traceback

                print(f"\n(oreilles : erreur en répondant : {exc})")
                traceback.print_exc()
            print("\nToi > ", end="", flush=True)
            gate.keep_engaged()

    threading.Thread(target=answer, name="valdar-oreilles", daemon=True).start()
    mic = Mic(gate.feed, cfg.ears.microphones)
    try:
        name = mic.start()
    except Exception as exc:
        print(f"(micro introuvable : {exc})")
        heard_q.put(None)
        return None
    session.attach_ears(gate, mic)
    names = " / ".join(cfg.ears.names[:1])
    print(f"(j'écoute sur « {name} ». Appelle-moi « {names} » ; rien n'est enregistré)")
    return mic


def _cmd_voix(args: argparse.Namespace) -> int:
    import wave

    import numpy as np

    from valdar.voice import Speaker, apply, make_tts, split_sentences, to_int16

    cfg = load()
    tts = make_tts(cfg)
    missing = tts.check_files()
    if missing:
        print("Il manque : " + ", ".join(missing) + ".\nLance d'abord « valdar import-ancien » "
              "(tools\\valdar_voix.bat le fait tout seul).")
        return 1
    print("Chargement de la voix (XTTS sur la carte graphique)…", flush=True)
    try:
        tts.load()
    except Exception as exc:
        print(f"La voix n'a pas pu se charger : {exc}")
        return 1
    print(f"Voix chargée en {tts.load_seconds:.1f} s ({tts.mode}).")
    text = " ".join(args.texte).strip() or ("Salut Olivier. C'est moi, Valdar. La voix que tu "
                                            "as construite, c'est la mienne, et j'y "
                                            "tiens.")
    if args.wav:
        parts = [apply(tts.synthesize(s), tts.sample_rate, cfg.voice.character)
                 for s in split_sentences(text, cfg.voice.max_sentence_chars)]
        audio = to_int16(np.concatenate(parts)) if parts else np.zeros(0, np.int16)
        with wave.open(args.wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(tts.sample_rate)
            wf.writeframes(audio.tobytes())
        print(f"Écrit : {args.wav} ({len(audio) / tts.sample_rate:.1f} s)")
    else:
        sp = Speaker(cfg.voice, tts).start(preload=False)
        sp.say(text)
        sp.wait_done(timeout=180)
        sp.close()
        st = sp.stats
        if st.errors:
            print("Erreur : " + st.errors[0])
            return 1
        synth, dur = sum(st.synth_seconds), sum(st.audio_seconds)
        first = f"{st.first_audio_seconds:.1f} s" if st.first_audio_seconds else "?"
        print(f"Premier son après {first} ; {dur:.1f} s de parole calculées en {synth:.1f} s "
              f"(facteur temps réel {synth / dur if dur else 0:.2f}, < 1 = plus vite que la "
              "parole).")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"Mémoire de la carte graphique prise par la voix : "
                  f"{torch.cuda.max_memory_reserved() / 1e9:.1f} Go.")
    except ImportError:
        pass
    return 0


def _cmd_import_ancien(args: argparse.Namespace) -> int:
    from pathlib import Path

    from valdar.atelier import Checklist, Reminders, Stock
    from valdar.memory import Facts
    from valdar.migrate import AncienImport

    cfg = load()
    a = cfg.atelier
    path = cfg.storage_path
    root = Path(args.dossier or cfg.migration.dossier_ancien)
    imp = AncienImport(root, Facts(path(a.memory_db)), Stock(path(a.stock_db)),
                       Reminders(path(a.reminders)), Checklist(path(a.checklist)),
                       path(a.pinouts), path("x").parent,
                       person=(cfg.agent.console_identity.person if cfg.agent else "") or "",
                       voice_ref=cfg.repo_path(cfg.voice.xtts.reference),
                       xtts_dir=cfg.repo_path(cfg.voice.xtts.model_dir))

    def step(key: str) -> None:
        if key == "voix":
            print("… je reprends la voix (environ 2 Go à copier, ça peut prendre une "
                  "minute)", flush=True)

    for line in imp.run(force=args.force, on_step=step):
        print(f"- {line}")
    return 0


def _cmd_import_vecu(args: argparse.Namespace) -> int:
    from pathlib import Path

    from valdar.memory import Episodic, Facts
    from valdar.migrate import VecuImport

    cfg = load()
    path = cfg.storage_path
    imp = VecuImport(Path(args.dossier), cfg, Episodic(path(cfg.episodic.db), cfg.episodic),
                     Facts(path(cfg.atelier.memory_db)), path("training"),
                     person=(cfg.agent.console_identity.person if cfg.agent else "") or "")
    print("Je fais de ces conversations mes souvenirs… (une à deux minutes)", flush=True)

    def progress(i: int, n: int) -> None:
        if i == n or i % 20 == 0:
            print(f"  {i}/{n} conversations", flush=True)

    for line in imp.run(on_progress=progress):
        print(f"- {line}")
    return 0


def _cmd_vigie(args: argparse.Namespace) -> int:
    from valdar.devices.moonraker import Moonraker
    from valdar.printwatch import PrintWatch

    cfg = load()
    path = cfg.storage_path
    pw = cfg.printwatch
    watch = PrintWatch(pw, Moonraker(cfg.printer), path(pw.db), path(pw.frames_dir),
                       notify=lambda ev: None)
    if args.debloquer:
        print(watch.unlock())
    elif args.verrouiller:
        print(watch.lock())
    elif args.reussie or args.ratee:
        print(watch.label(bool(args.ratee), minutes_before_end=args.minutes))
    else:
        print(watch.status_text())
    return 0


def _cmd_kiwix(args: argparse.Namespace) -> int:
    from valdar.knowledge.kiwix_get import interactive

    cfg = load()
    return interactive(cfg.repo_path(cfg.kiwix.dir))


def _cmd_chrono(args: argparse.Namespace) -> int:
    from valdar.chrono import main as chrono

    return chrono(load(), voice=not args.sans_voix, ears=not args.sans_oreilles,
                  sense=not args.sans_sens)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="valdar")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("chat", help="parler à Valdar au clavier")
    p.add_argument("--debug", action="store_true", help="affiche les outils utilisés")
    p.add_argument("--voix", action="store_true", help="répond aussi à voix haute")
    p.add_argument("--ecoute", action="store_true",
                   help="écoute au micro (dis « Valdar » pour lui parler)")
    p.add_argument("--camera", action="store_true",
                   help="regarde la pièce : visages, le chien sur la table")
    p.add_argument("--interface", action="store_true",
                   help="ouvre l'interface (visage et onglets) sur le projecteur")
    p.set_defaults(func=_cmd_chat)

    p = sub.add_parser("voix", help="faire parler Valdar (test de la voix)")
    p.add_argument("texte", nargs="*", help="phrase à dire")
    p.add_argument("--wav", default=None, help="écrire dans un fichier .wav au lieu de jouer")
    p.set_defaults(func=_cmd_voix)

    p = sub.add_parser("import-ancien", help="reprendre les données de l'ancienne installation")
    p.add_argument("--dossier", default=None,
                   help="dossier de l'ancienne installation (défaut : migration.dossier_ancien "
                        "dans config/valdar.yaml)")
    p.add_argument("--force", action="store_true", help="refaire même si déjà importé")
    p.set_defaults(func=_cmd_import_ancien)

    p = sub.add_parser("import-vecu", help="faire des conversations d'Olivier des souvenirs")
    p.add_argument("--dossier", default=r"C:\Users\doxei\Documents\training ia",
                   help="dossier de l'export (conversations.json, memories)")
    p.set_defaults(func=_cmd_import_vecu)

    p = sub.add_parser("vigie", help="état de la vigie d'impression, autonomie, étiquettes")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--debloquer", action="store_true",
                   help="autoriser la pause automatique (si la porte des 98 %% est ouverte)")
    g.add_argument("--verrouiller", action="store_true", help="revenir à « observe et prévient »")
    g.add_argument("--reussie", action="store_true", help="la dernière impression a réussi")
    g.add_argument("--ratee", action="store_true", help="la dernière impression a raté")
    p.add_argument("--minutes", type=float, default=None,
                   help="avec --ratee : minutes avant la fin où c'était déjà fichu")
    p.set_defaults(func=_cmd_vigie)

    p = sub.add_parser("chrono", help="chronomètre un tour complet (clavier, voix, oreilles)")
    p.add_argument("--sans-voix", action="store_true")
    p.add_argument("--sans-oreilles", action="store_true")
    p.add_argument("--sans-sens", action="store_true")
    p.set_defaults(func=_cmd_chrono)

    p = sub.add_parser("kiwix", help="installer les bibliothèques hors ligne (Wikipédia…)")
    p.set_defaults(func=_cmd_kiwix)

    p = sub.add_parser("heart", help="cœur en continu, temps réel")
    p.add_argument("--profile", default=None)
    p.add_argument("--every", type=float, default=60.0, help="secondes entre deux lignes d'état")
    p.set_defaults(func=_cmd_heart)

    p = sub.add_parser("status", help="état actuel du cœur")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("sim", help="simulation accélérée")
    p.add_argument("--days", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--profile", default=None)
    p.set_defaults(func=_cmd_sim)

    p = sub.add_parser("check", help="critères d'acceptation de la phase 1")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
