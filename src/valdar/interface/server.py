"""Serveur de l'interface : une page locale, des données en JSON, le direct en SSE.

Bibliothèque standard seulement (rien à installer de plus). Écoute sur 127.0.0.1 : rien
n'est accessible depuis le réseau.
"""
from __future__ import annotations

import contextlib
import json
import mimetypes
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from valdar.interface.session import Session
from valdar.interface.settings import Settings

STATIC = Path(__file__).resolve().parent / "static"


class Api:
    """Les réponses de l'interface, séparées du transport (testables sans réseau)."""

    def __init__(self, session: Session):
        self.s = session
        self.rt = session.rt
        self.settings = Settings(session.cfg)

    # ------------------------------------------------------------ lecture
    def state(self) -> dict[str, Any]:
        return self.s.state()

    def history(self) -> dict[str, Any]:
        return {"log": self.s.bus.log[-120:]}

    def foyer(self) -> dict[str, Any]:
        rel = self.rt.relations
        vc = self.s.prints.counts("voix") if self.s.prints else {}
        fc = self.s.prints.counts("visage") if self.s.prints else {}
        members = []
        for m in self.s.foyer.members:
            p = rel.get(m["id"])
            members.append({**m, "voix": vc.get(m["id"], 0), "visage": fc.get(m["id"], 0),
                            "rencontres": p.encounters if p else 0,
                            "affection": round(p.affection, 2) if p else 0.0,
                            "confiance": round(p.trust, 2) if p else 0.0,
                            "sujets": sorted((p.topics if p else {}).items(),
                                             key=lambda kv: -kv[1])[:5],
                            "lecons": [x["text"] for x in rel.lessons(m["id"])][-6:]})
        w = self.s.watch
        return {"membres": members, "animaux": self.s.foyer.animals,
                "vus": sorted(w.seen.items(), key=lambda kv: -kv[1]) if w else [],
                "camera": None if w is None else {"error": w.error,
                                                  "scene": w.last.to_dict() if w.last
                                                  else None},
                "voix_ok": self.s.voice_id is not None, "visage_ok": self.s.face_id is not None}

    def memoire(self, q: str = "", statut: str = "") -> dict[str, Any]:
        f = self.rt.facts
        rows = f.recall(q, limit=200) if q.strip() else f._rows("archived=0")
        if statut:
            rows = [r for r in rows if r["statut"] == statut]
        rows.sort(key=lambda r: -(r["last_used"] or 0))
        return {"croyances": rows[:300], "total": f.count()}

    def pensees(self) -> dict[str, Any]:
        th = self.rt.thoughts
        with th._conn() as con:
            rows = con.execute("SELECT t, reflection, curiosity FROM thoughts "
                               "ORDER BY t DESC LIMIT 30").fetchall()
        ideas = [{"id": i.id, "t": i.t, "texte": i.text, "pourquoi": i.why,
                  "statut": i.status, "score": round(i.score, 2)}
                 for i in th.ideas(limit=40, status=None)]
        night = self.rt.night.last
        return {"pensees": [{"t": r[0], "texte": r[1], "curiosite": r[2]} for r in rows],
                "idees": ideas, "nuit": None if night is None else night.text()}

    def atelier(self) -> dict[str, Any]:
        rt = self.rt
        printer = None
        if rt.printer is not None:
            from valdar.tools.standard import printer_line

            try:
                printer = printer_line(rt.printer)
            except Exception as exc:
                printer = f"imprimante injoignable : {exc}"
        return {"checklist": rt.checklist.render() if rt.checklist.current.active() else "",
                "stock": rt.stock.find("")[:200], "bas": rt.stock.low(),
                "rappels": rt.reminders.upcoming(20), "imprimante": printer,
                "vigie": rt.printwatch.status_text() if rt.printwatch else None}

    def connaissances(self, q: str = "") -> dict[str, Any]:
        k = self.rt.knowledge
        hits = [{"titre": h.title, "doc": h.doc, "texte": h.text[:1200],
                 "score": round(h.score, 3)} for h in k.search(q, k=12)] if q else []
        with k._conn() as con:
            docs = [{"doc": d, "passages": n} for d, n in
                    con.execute("SELECT doc, n FROM docs ORDER BY doc")]
        return {"docs": docs, "resultats": hits, "total": k.count(),
                "kiwix": getattr(self.rt, "kiwix_status", lambda: None)()}

    def outils(self) -> dict[str, Any]:
        return {"outils": [{"nom": t.name, "description": t.description, "niveau": t.tier,
                            "famille": t.family}
                           for t in self.rt.registry.tools.values()]}

    def reglages(self) -> dict[str, Any]:
        models: list[str] = []
        try:
            import httpx

            r = httpx.get(self.s.cfg.llm.url.rstrip("/") + "/api/tags", timeout=3.0)
            models = sorted(m["name"] for m in r.json().get("models", []))
        except Exception:
            pass
        return {"reglages": self.settings.listing(), "modeles": models}

    def n8n(self) -> dict[str, Any]:
        c = self.s.cfg.n8n
        return {"enabled": c.enabled, "url": c.url, "workflows": list(c.workflows)}

    def journal(self) -> dict[str, Any]:
        mem = self.rt.memory
        now = time.time()
        turns = mem.day_turns(now - 7 * 86400, now + 1, include_quarantined=True)[-300:]
        q = mem.quarantined_turns()
        return {"tours": [{**t, "quarantaine": q.get(t["id"])} for t in turns]}

    # ------------------------------------------------------------ actions
    def message(self, body: dict[str, Any]) -> dict[str, Any]:
        text = str(body.get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "message vide"}
        threading.Thread(target=self.s.answer, args=(text,),
                         kwargs={"source": "interface"}, daemon=True).start()
        return {"ok": True}

    def control(self, body: dict[str, Any]) -> dict[str, Any]:
        a = body.get("action")
        s = self.s
        if a == "muet":
            s.muted = True
            if s.speaker:
                s.speaker.interrupt()
        elif a == "voix":
            s.muted = False
        elif a == "tais-toi" and s.speaker:
            s.speaker.interrupt()
        elif a == "silence":
            self.rt.initiative.silence(True)
        elif a == "parle":
            self.rt.initiative.silence(False)
        elif a == "annuler":
            s.cancel_enroll()
        else:
            return {"ok": False, "error": f"action inconnue : {a}"}
        return {"ok": True}

    def foyer_action(self, body: dict[str, Any]) -> dict[str, Any]:
        a = body.get("action")
        f = self.s.foyer
        try:
            if a == "ajouter":
                m = f.add_member(str(body.get("nom", "")), str(body.get("lien", "")),
                                 bool(body.get("mineur")))
                return {"ok": True, "membre": m}
            if a == "retirer":
                pid = str(body.get("id"))
                self.s.forget_prints(pid)
                return {"ok": f.remove_member(pid)}
            if a == "animal":
                return {"ok": True, "animal": f.add_animal(str(body.get("nom", "")),
                                                           str(body.get("espece", "chien")))}
            if a == "retirer_animal":
                return {"ok": f.remove_animal(str(body.get("nom", "")))}
            if a == "voix":
                return {"ok": True, "message": self.s.enroll_voice(str(body.get("id")))}
            if a == "visage":
                return {"ok": True, "message": self.s.enroll_face(str(body.get("id")))}
            if a == "oublier_empreintes":
                return {"ok": True, "effacees": self.s.forget_prints(str(body.get("id")))}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": False, "error": f"action inconnue : {a}"}

    def memoire_action(self, body: dict[str, Any]) -> dict[str, Any]:
        f = self.rt.facts
        fid = int(body.get("id", 0))
        a = body.get("action")
        if a == "oublier":
            return {"ok": f.archive(fid)}
        if a == "lever":
            return {"ok": f.lift(fid)}
        if a == "quarantaine":
            return {"ok": f.quarantine(fid, str(body.get("motif") or "mis de côté par "
                                                "Olivier"))}
        if a == "ajouter":
            return {"ok": True, "message": f.remember(str(body.get("texte", "")),
                                                      person=self.s.owner.person or "",
                                                      origine="dit")}
        return {"ok": False, "error": f"action inconnue : {a}"}

    def reglage(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, **self.settings.set(str(body.get("path")), body.get("value"))}
        except (KeyError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    def idee(self, body: dict[str, Any]) -> dict[str, Any]:
        st = str(body.get("statut") or "")
        if st not in ("acceptee", "refusee", "nouvelle"):
            return {"ok": False, "error": "statut inconnu"}
        self.rt.thoughts.mark(int(body.get("id", 0)), st)
        return {"ok": True}


GETS = {"/api/state": "state", "/api/historique": "history", "/api/foyer": "foyer",
        "/api/memoire": "memoire", "/api/pensees": "pensees", "/api/atelier": "atelier",
        "/api/connaissances": "connaissances", "/api/outils": "outils",
        "/api/reglages": "reglages", "/api/journal": "journal", "/api/n8n": "n8n"}
POSTS = {"/api/message": "message", "/api/control": "control", "/api/foyer": "foyer_action",
         "/api/memoire": "memoire_action", "/api/reglages": "reglage", "/api/idee": "idee"}


def make_handler(api: Api) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Valdar/1"

        def log_message(self, fmt: str, *args: Any) -> None:   # silencieux
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any, code: int = 200) -> None:
            self._send(code, json.dumps(data, ensure_ascii=False, default=str).encode(),
                       "application/json; charset=utf-8")

        def do_GET(self) -> None:
            u = urlparse(self.path)
            if u.path == "/api/events":
                return self._events()
            if u.path in GETS:
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                try:
                    return self._json(getattr(api, GETS[u.path])(**q))
                except TypeError:
                    return self._json(getattr(api, GETS[u.path])())
                except Exception as exc:
                    return self._json({"error": str(exc)}, 500)
            name = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
            f = (STATIC / name).resolve()
            if STATIC not in f.parents or not f.is_file():
                return self._send(404, b"introuvable", "text/plain")
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            self._send(200, f.read_bytes(), ctype)

        def do_POST(self) -> None:
            u = urlparse(self.path)
            if u.path not in POSTS:
                return self._json({"error": "introuvable"}, 404)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"error": "JSON invalide"}, 400)
            try:
                self._json(getattr(api, POSTS[u.path])(body))
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 500)

        def _events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q: queue.Queue = queue.Queue(maxsize=500)

            def on(kind: str, data: dict[str, Any]) -> None:
                with contextlib.suppress(queue.Full):    # client trop lent : on saute
                    q.put_nowait((kind, data))

            off = api.s.bus.subscribe(on)
            try:
                self.wfile.write(b": bonjour\n\n")
                self.wfile.flush()
                while True:
                    try:
                        kind, data = q.get(timeout=15)
                        payload = json.dumps(data, ensure_ascii=False, default=str)
                        self.wfile.write(f"event: {kind}\ndata: {payload}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                off()

    return Handler


class Server:
    def __init__(self, session: Session, host: str = "127.0.0.1", port: int = 8765):
        self.api = Api(session)
        self.httpd = ThreadingHTTPServer((host, port), make_handler(self.api))
        self.httpd.daemon_threads = True
        self.url = f"http://{host}:{self.httpd.server_address[1]}/"
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="valdar-web",
                                        daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
