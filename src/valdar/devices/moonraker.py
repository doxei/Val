"""Client Moonraker (Klipper) : lecture d'état, fichiers, pause et arrêt d'urgence.

Démarrer, annuler, températures et macros arrivent en phase 5, avec l'identification
et les garde-fous de PrinterAgent.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from valdar.config.loader import PrinterConfig


class PrinterError(RuntimeError):
    pass


class Moonraker:
    def __init__(self, cfg: PrinterConfig, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self._transport = transport
        self._cache: tuple[float, dict[str, Any] | None] = (0.0, None)

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.cfg.moonraker_url.rstrip("/"),
                            timeout=httpx.Timeout(self.cfg.timeout_seconds, connect=2.0),
                            headers={"Accept": "application/json"}, transport=self._transport)

    def _get(self, path: str) -> dict[str, Any]:
        try:
            with self._client() as c:
                r = c.get(path)
        except httpx.HTTPError as exc:
            raise PrinterError(f"imprimante injoignable ({self.cfg.moonraker_url})") from exc
        if r.status_code >= 400:
            raise PrinterError(f"{path} → {r.status_code}")
        return r.json().get("result", {})

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> None:
        try:
            with self._client() as c:
                r = c.post(path, json=payload or {})
        except httpx.HTTPError as exc:
            raise PrinterError(f"imprimante injoignable ({self.cfg.moonraker_url})") from exc
        if r.status_code >= 400:
            raise PrinterError(f"{path} → {r.status_code} : {r.text[:200]}")

    def status(self) -> dict[str, Any]:
        info = self._get("/server/info")
        st = self._get(
            "/printer/objects/query?print_stats&virtual_sdcard&display_status"
            "&heater_bed&extruder&gcode_move").get("status", {})
        ps, sd = st.get("print_stats", {}), st.get("virtual_sdcard", {})
        bed, ext = st.get("heater_bed", {}), st.get("extruder", {})
        return {
            "klippy": info.get("klippy_state"),
            "state": ps.get("state"),
            "message": ps.get("message") or st.get("display_status", {}).get("message") or "",
            "file": (sd.get("file_path") or ps.get("filename") or "").rsplit("/", 1)[-1],
            "progress": round(100 * (sd.get("progress") or 0), 1),
            "duration_s": int(ps.get("print_duration") or 0),
            "bed": (round(bed.get("temperature") or 0, 1), bed.get("target") or 0),
            "nozzle": (round(ext.get("temperature") or 0, 1), ext.get("target") or 0),
        }

    def cached_status(self) -> dict[str, Any] | None:
        """État mis en cache (cache_seconds) ; None si l'imprimante est injoignable."""
        at, val = self._cache
        if time.time() - at < self.cfg.cache_seconds:
            return val
        try:
            val = self.status()
        except PrinterError:
            val = None
        self._cache = (time.time(), val)
        return val

    def status_text(self) -> str:
        try:
            s = self.status()
        except PrinterError as exc:
            return str(exc)
        return describe(s)

    def files(self) -> list[dict[str, Any]]:
        data = self._get("/server/files/list?root=gcodes")
        if isinstance(data, dict):
            data = data.get("files", [])
        return [f for f in data or [] if isinstance(f, dict) and f.get("path")]

    def webcams(self) -> list[dict[str, Any]]:
        """Caméras déclarées dans Moonraker (crowsnest) : nom et URL d'instantané."""
        try:
            data = self._get("/server/webcams/list")
        except PrinterError:
            return []
        cams = data.get("webcams", []) if isinstance(data, dict) else []
        return [{"name": c.get("name") or "camera", "snapshot_url": c.get("snapshot_url"),
                 "rotate": int(c.get("rotation") or 0)}
                for c in cams if isinstance(c, dict) and c.get("snapshot_url")
                and c.get("enabled", True)]

    def pause(self) -> None:
        self._post("/printer/print/pause")

    def emergency_stop(self) -> None:
        self._post("/printer/emergency_stop")


def describe(s: dict[str, Any]) -> str:
    lines = [f"état : {s['state']} (klipper {s['klippy']})"]
    if s.get("file"):
        lines.append(f"fichier : {s['file']}")
    if s.get("state") in ("printing", "paused"):
        m, sec = divmod(s.get("duration_s", 0), 60)
        lines.append(f"progression : {s['progress']} % depuis {m} min {sec:02d} s")
    lines.append(f"plateau : {s['bed'][0]} °C (consigne {s['bed'][1]:.0f})")
    lines.append(f"buse : {s['nozzle'][0]} °C (consigne {s['nozzle'][1]:.0f})")
    if s.get("message"):
        lines.append(f"message : {s['message']}")
    return "\n".join(lines)
