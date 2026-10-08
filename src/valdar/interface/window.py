"""Ouvrir l'interface sur le projecteur (Windows) : Edge en mode application, toujours devant.

Comme l'ancienne interface : Edge `--app` (une fenêtre sans barre d'adresse), placée sur
l'écran voulu (2 = le projecteur), puis gardée au premier plan.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

TITLE = "Valdar"


def monitors() -> list[tuple[int, int, int, int]]:
    """Rectangles des écrans (x, y, largeur, hauteur), l'écran principal d'abord."""
    if os.name != "nt":
        return []
    import ctypes
    import ctypes.wintypes

    rects: list[tuple[int, int, int, int]] = []
    proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.wintypes.HMONITOR, ctypes.wintypes.HDC,
                               ctypes.POINTER(ctypes.wintypes.RECT), ctypes.wintypes.LPARAM)

    @proto
    def cb(_h, _dc, rect, _d):           # type: ignore[no-untyped-def]
        r = rect.contents
        rects.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(None, None, cb, 0)
    rects.sort(key=lambda r: (r[0] != 0 or r[1] != 0, r[0]))   # principal (0,0) d'abord
    return rects


def _edge() -> Path | None:
    for base in (r"C:\Program Files (x86)\Microsoft\Edge\Application",
                 r"C:\Program Files\Microsoft\Edge\Application"):
        p = Path(base) / "msedge.exe"
        if p.is_file():
            return p
    return None


def _keep_on_top(stop: threading.Event) -> None:
    """Retrouve la fenêtre « Valdar » et la garde au-dessus (même après un clic ailleurs)."""
    import ctypes
    import ctypes.wintypes

    user32 = ctypes.windll.user32
    found: list[int] = []
    proto = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

    @proto
    def each(hwnd, _l):                 # type: ignore[no-untyped-def]
        n = user32.GetWindowTextLengthW(hwnd)
        if n and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if buf.value.startswith(TITLE):
                found.append(hwnd)
        return True

    HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = -1, 0x2, 0x1, 0x10
    while not stop.is_set():
        found.clear()
        user32.EnumWindows(each, 0)
        for hwnd in found:
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
        stop.wait(5.0)


def open_window(url: str, screen: int = 2, kiosk: bool = False) -> threading.Event | None:
    """Ouvre l'interface. Renvoie un drapeau à lever pour arrêter le « toujours devant »."""
    if os.name != "nt":
        import webbrowser

        webbrowser.open(url)
        return None
    rects = monitors()
    rect = rects[screen - 1] if 0 < screen <= len(rects) else (rects[0] if rects else None)
    edge = _edge()
    if edge is None:
        os.startfile(url)        # type: ignore[attr-defined]
        return None
    cmd = [str(edge), f"--app={url}", "--no-first-run", "--no-default-browser-check",
           "--autoplay-policy=no-user-gesture-required"]
    if rect:
        x, y, w, h = rect
        cmd += [f"--window-position={x},{y}", f"--window-size={w},{h}"]
    if kiosk:
        cmd[1:1] = ["--kiosk", "--edge-kiosk-type=fullscreen"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(cmd, close_fds=True, creationflags=flags)
    if len(rects) < 2 or screen < 2:
        return None          # un seul écran : ne jamais bloquer le bureau d'Olivier
    stop = threading.Event()

    def later() -> None:
        time.sleep(3)
        _keep_on_top(stop)

    threading.Thread(target=later, name="valdar-devant", daemon=True).start()
    return stop
