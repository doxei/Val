"""Le moteur du cortex : Gemma chargé en mémoire par Valdar (llama.cpp), avec deux gestes
qu'Ollama ne permet pas :

- **injecter** : ajouter un vecteur au flux résiduel de chaque couche, à chaque jeton
  (`llama_set_adapter_cvec`) ;
- **lire** : relever les activations de chaque couche pendant que le modèle lit un texte
  (rappel `cb_eval`), pour apprendre les directions des sentiments.

Tout passe par `llama-cpp-python`. Ce module ne s'importe que si le cortex est utilisé.
"""
from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from valdar.llm.backend import GenParams, LLMError

GGML_TYPE_Q8_0 = 8


def _dll_dirs() -> None:
    """Windows : la roue CUDA de llama-cpp-python a besoin des bibliothèques CUDA 12
    (cudart, cublas). Celles de PyTorch (déjà là pour la voix) ou des paquets nvidia-*
    font l'affaire : on les ajoute au chemin de recherche."""
    if os.name != "nt":
        return
    dirs: list[Path] = []
    for mod, sub in (("torch", ["lib"]), ("nvidia", ["cuda_runtime/bin", "cublas/bin"])):
        with contextlib.suppress(Exception):
            spec = importlib.util.find_spec(mod)
            for loc in (spec.submodule_search_locations if spec else None) or []:
                dirs += [Path(loc) / s for s in sub]
    for d in dirs:
        if d.is_dir():
            with contextlib.suppress(OSError):
                os.add_dll_directory(str(d))
            os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"


class Engine:
    """Un modèle GGUF chargé, prêt à générer, à être piloté et à être lu."""

    def __init__(self, cc: Any, root: Path, capture: bool = False, n_ctx: int = 8192,
                 verbose: bool = False):
        self.cc = cc
        self.path = (root / cc.model_path) if not Path(cc.model_path).is_absolute() else Path(
            cc.model_path)
        self.capture = capture
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.lock = threading.RLock()
        self.llm: Any = None
        self._lc: Any = None
        self._cb: Any = None
        self._gg: Any = None
        self._got: dict[int, np.ndarray] = {}
        self._want_rows = 0
        self.steered = False

    # ----------------------------------------------------------- chargement
    def load(self) -> None:
        if self.llm is not None:
            return
        if not self.path.is_file():
            raise LLMError(f"modèle du cortex introuvable : {self.path} "
                           "(lance tools\\valdar_cortex.bat)")
        _dll_dirs()
        try:
            import llama_cpp
            import llama_cpp.llama_cpp as lc
        except Exception as exc:        # paquet absent ou bibliothèque CUDA manquante
            raise LLMError(f"llama-cpp-python inutilisable : {exc}") from exc
        self._lc = lc
        kw: dict[str, Any] = dict(
            model_path=str(self.path), n_ctx=self.n_ctx, n_batch=self.cc.n_batch,
            n_gpu_layers=self.cc.gpu_layers, n_threads=self.cc.threads,
            flash_attn=self.cc.flash_attn, verbose=self.verbose, seed=-1)
        if self.cc.kv_q8 and self.cc.flash_attn:
            kw.update(type_k=GGML_TYPE_Q8_0, type_v=GGML_TYPE_Q8_0)
        orig = lc.llama_context_default_params
        if self.capture:
            self._bind_ggml(llama_cpp)
            self._cb = lc.ggml_backend_sched_eval_callback(self._on_tensor)

            def patched() -> Any:
                p = orig()
                p.cb_eval = self._cb
                return p
            lc.llama_context_default_params = patched
        try:
            try:
                self.llm = llama_cpp.Llama(**kw)
            except Exception:
                if "type_k" not in kw:
                    raise
                kw.pop("type_k")        # ce modèle refuse le cache en 8 bits : cache normal
                kw.pop("type_v")
                self.llm = llama_cpp.Llama(**kw)
        except Exception as exc:
            raise LLMError(f"le modèle du cortex ne se charge pas : {exc}") from exc
        finally:
            lc.llama_context_default_params = orig
        meta = self.llm.metadata
        arch = meta.get("general.architecture", "")
        self.name = meta.get("general.name") or self.path.stem
        self.n_embd = int(self.llm.n_embd())
        self.n_layers = int(meta.get(f"{arch}.block_count", 0))
        self.template = meta.get("tokenizer.chat_template", "")
        if self.n_layers < 2:
            raise LLMError("modèle du cortex : nombre de couches inconnu")
        self.bos = self._piece(self.llm.token_bos()) or "<bos>"
        self.stop_ids = {self.llm.token_eos()}
        for mark in ("<turn|>", "<|tool_response>", "<|turn>", "<end_of_turn>"):
            ids = self.llm.tokenize(mark.encode(), add_bos=False, special=True)
            if len(ids) == 1:
                self.stop_ids.add(ids[0])

    def _piece(self, tok: int) -> str:
        return self.llm.detokenize([tok], special=True).decode("utf-8", errors="ignore")

    def close(self) -> None:
        with self.lock:
            if self.llm is not None:
                with contextlib.suppress(Exception):
                    self.llm.close()
                self.llm = None

    # ----------------------------------------------------------- injection
    def steer(self, buf: np.ndarray | None, first: int = 1, last: int | None = None) -> None:
        """Pose (ou retire, avec None) l'injection : `buf` est [couche − 1, dimension]."""
        self.load()
        ctx = self.llm._ctx.ctx
        with self.lock:
            if buf is None:
                if self.steered:
                    self._lc.llama_set_adapter_cvec(ctx, None, 0, self.n_embd, -1, -1)
                self.steered = False
                return
            if buf.shape != (self.n_layers - 1, self.n_embd):
                raise LLMError(f"injection de mauvaise taille : {buf.shape}, attendu "
                               f"{(self.n_layers - 1, self.n_embd)} (vecteurs d'un autre modèle ?)")
            flat = np.ascontiguousarray(buf, dtype=np.float32).ravel()
            ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            last = self.n_layers - 1 if last is None else last
            rc = self._lc.llama_set_adapter_cvec(ctx, ptr, flat.size, self.n_embd, first, last)
            if rc != 0:
                raise LLMError("l'injection a été refusée par le modèle")
            self.steered = True

    # ----------------------------------------------------------- génération
    def tokens(self, text: str) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)

    def generate(self, prompt: str, params: GenParams,
                 on_text: Callable[[str], None] | None = None) -> tuple[str, dict[str, Any]]:
        """Écrit la suite de `prompt`. Renvoie (texte brut avec ses marques, mesures)."""
        self.load()
        with self.lock:
            toks = self.tokens(prompt)
            if len(toks) + 16 >= self.n_ctx:
                raise LLMError(f"prompt trop long : {len(toks)} jetons pour {self.n_ctx}")
            room = min(params.max_tokens, self.n_ctx - len(toks) - 8)
            t0 = time.perf_counter()
            reused = _common_prefix(self.llm.n_tokens, toks, self.llm)
            out: list[int] = []
            sent = 0
            text = ""
            first_at: float | None = None
            try:
                stream: Iterator[int] = self.llm.generate(
                    toks, temp=params.temperature, top_p=params.top_p,
                    repeat_penalty=params.repeat_penalty, reset=True)
                for tok in stream:
                    if first_at is None:
                        first_at = time.perf_counter()
                    if tok in self.stop_ids:
                        break
                    out.append(tok)
                    text = self.llm.detokenize(out, special=True).decode("utf-8",
                                                                         errors="ignore")
                    if on_text is not None and len(text) > sent and not text.endswith("�"):
                        on_text(text[sent:])
                        sent = len(text)
                    if len(out) >= room:
                        break
            except LLMError:
                raise
            except Exception as exc:
                raise LLMError(f"le cortex a calé pendant la génération : {exc}") from exc
            if on_text is not None and len(text) > sent:
                on_text(text[sent:])
            t1 = time.perf_counter()
            first_at = first_at or t1
            usage = {
                "prompt_eval_count": len(toks) - reused,
                "prompt_tokens_total": len(toks),
                "eval_count": len(out),
                "total_duration": int((t1 - t0) * 1e9),
                "prompt_eval_duration": int((first_at - t0) * 1e9),
                "eval_duration": int((t1 - first_at) * 1e9),
                "load_duration": 0,
            }
            return text, usage

    # ----------------------------------------------------------- lecture
    def _bind_ggml(self, llama_cpp: Any) -> None:
        import llama_cpp._ctypes_extensions as ext

        base = Path(llama_cpp.__file__).parent / "lib"
        last: Exception | None = None
        for name in ("ggml-base", "ggml"):
            try:
                gg = ext.load_shared_library(name, base)
                gg.ggml_get_name.restype = ctypes.c_char_p
                gg.ggml_get_name.argtypes = [ctypes.c_void_p]
                gg.ggml_nbytes.restype = ctypes.c_size_t
                gg.ggml_nbytes.argtypes = [ctypes.c_void_p]
                gg.ggml_backend_tensor_get.restype = None
                gg.ggml_backend_tensor_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                       ctypes.c_size_t, ctypes.c_size_t]
                self._gg = gg
                return
            except Exception as exc:
                last = exc
        raise LLMError(f"lecture des couches impossible (bibliothèque ggml) : {last}")

    def _on_tensor(self, t: Any, ask: bool, _user: Any) -> bool:
        try:
            name = self._gg.ggml_get_name(t) or b""
            if not name.startswith(b"l_out-"):
                return not ask
            if ask:
                return True
            n = self._gg.ggml_nbytes(t)
            arr = np.empty(n // 4, dtype=np.float32)
            self._gg.ggml_backend_tensor_get(t, arr.ctypes.data, 0, n)
            rows = arr.reshape(-1, self.n_embd)
            if rows.shape[0] == self._want_rows:
                self._got[int(name[6:])] = rows
        except Exception:
            pass
        return True

    def read(self, prompt: str, tail: str) -> np.ndarray:
        """Activations moyennes sur les jetons de `tail` (la fin du texte), à chaque couche
        1..n−1 : tableau [couche − 1, dimension]. Demande un moteur ouvert avec capture."""
        if not self.capture:
            raise LLMError("moteur ouvert sans lecture des couches")
        self.load()
        with self.lock:
            head = self.tokens(prompt)
            body = self.tokens(tail)
            toks = head + body
            if len(toks) > self.cc.n_batch:
                raise LLMError("texte trop long pour une lecture en une passe")
            self._got = {}
            self._want_rows = len(toks)
            self.llm.reset()
            self.llm.eval(toks)
            out = np.zeros((self.n_layers - 1, self.n_embd), dtype=np.float32)
            missing = []
            for layer in range(1, self.n_layers):
                rows = self._got.get(layer)
                if rows is None:
                    missing.append(layer)
                    continue
                out[layer - 1] = rows[len(head):].mean(axis=0)
            # la dernière couche ne calcule que le dernier jeton : on la laisse à zéro
            if [m for m in missing if m < self.n_layers - 1]:
                raise LLMError(f"couches illisibles : {missing[:6]}")
            self._got = {}
            return out


def _common_prefix(before: int, toks: list[int], llm: Any) -> int:
    """Jetons du prompt déjà en cache avant cet appel (estimation pour les mesures)."""
    with contextlib.suppress(Exception):
        ids = llm._input_ids[:min(before, len(toks))]
        n = 0
        for a, b in zip(ids, toks, strict=False):
            if int(a) != b:
                break
            n += 1
        return n
    return 0
