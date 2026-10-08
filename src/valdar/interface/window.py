"""Ouvrir l'interface sur le projecteur (Windows) et l'y tenir.

Edge en mode application (`--app`, sans barre d'adresse), dans **son propre profil** (sinon
Edge, déjà ouvert, avale la nouvelle fenêtre et ignore position et taille). Ensuite Valdar
prend la main sur la fenêtre :
- sans bordure ni barre de titre (elle ne se déplace pas à la souris) ;
- exactement aux dimensions de l'écran 2 (le projecteur) ;
- toujours au premier plan ;
- remise en place toutes les secondes si quelque chose la bouge.

Un seul écran : la fenêtre s'ouvre normalement, sans rien bloquer.
"""
from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

TITLE = "Valdar"
EDGE_CLASS = "Chrome_WidgetWin_1"        # classe des fenêtres d'Edge (et de Chrome)


@dataclass
class Monitor:
    x: int
    y: int
    w: int
    h: int
    primary: bool


def _dpi_aware() -> None:
    """Coordonnées réelles des écrans (sinon Windows les « met à l'échelle » et la fenêtre
    tombe à côté sur un projecteur réglé à 125 ou 150 %)."""
    import ctypes

    for call in (lambda: ctypes.windll.user32.SetProcessDpiAwarenessContext(-4),
                 lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),
                 lambda: ctypes.windll.user32.SetProcessDPIAware()):
        try:
            if call():
                return
        except Exception:
            continue


def monitors() -> list[Monitor]:
    """Les écrans, le principal d'abord, puis les autres de gauche à droite."""
    if os.name != "nt":
        return []
    import ctypes
    import ctypes.wintypes as wt

    _dpi_aware()

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wt.DWORD), ("rcMonitor", wt.RECT), ("rcWork", wt.RECT),
                    ("dwFlags", wt.DWORD)]

    found: list[Monitor] = []
    proto = ctypes.WINFUNCTYPE(ctypes.c_int, wt.HMONITOR, wt.HDC, ctypes.POINTER(wt.RECT),
                               wt.LPARAM)

    @proto
    def cb(hmon, _dc, _rect, _d):        # type: ignore[no-untyped-def]
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        ctypes.windll.user32.GetMonitorInfoW(hmon, ctypes.byref(info))
        r = info.rcMonitor
        found.append(Monitor(r.left, r.top, r.right - r.left, r.bottom - r.top,
                             bool(info.dwFlags & 1)))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(None, None, cb, 0)
    found.sort(key=lambda m: (not m.primary, m.x, m.y))
    return found


def pick(mons: list[Monitor], screen: int) -> Monitor | None:
    if not mons:
        return None
    return mons[screen - 1] if 0 < screen <= len(mons) else mons[-1]


def _edge() -> Path | None:
    for base in (r"C:\Program Files (x86)\Microsoft\Edge\Application",
                 r"C:\Program Files\Microsoft\Edge\Application"):
        p = Path(base) / "msedge.exe"
        if p.is_file():
            return p
    return None


def _find_window() -> int | None:
    """La fenêtre d'Edge dont le titre commence par « Valdar » (pas la console)."""
    import ctypes
    import ctypes.wintypes as wt

    user32 = ctypes.windll.user32
    hits: list[int] = []
    proto = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)

    @proto
    def each(hwnd, _l):                  # type: ignore[no-untyped-def]
        if not user32.IsWindowVisible(hwnd):
            return True
        cls = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, cls, 64)
        if cls.value != EDGE_CLASS:
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if buf.value.startswith(TITLE):
            hits.append(hwnd)
        return True

    user32.EnumWindows(each, 0)
    return hits[0] if hits else None


def _pin(hwnd: int, m: Monitor) -> None:
    """Sans bordure, à la taille exacte de l'écran, au premier plan."""
    import ctypes
    import ctypes.wintypes as wt

    user32 = ctypes.windll.user32
    user32.GetWindowLongW.restype = ctypes.c_long
    GWL_STYLE, WS_CAPTION, WS_THICKFRAME = -16, 0x00C00000, 0x00040000
    WS_SYSMENU, WS_MIN, WS_MAX = 0x00080000, 0x00020000, 0x00010000
    style = user32.GetWindowLongW(hwnd, GWL_STYLE)
    bare = style & ~(WS_CAPTION | WS_THICKFRAME | WS_SYSMENU | WS_MIN | WS_MAX)
    if bare != style:
        user32.SetWindowLongW(hwnd, GWL_STYLE, bare)
    r = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    HWND_TOPMOST, SWP_NOACTIVATE, SWP_FRAMECHANGED, SWP_SHOWWINDOW = -1, 0x10, 0x20, 0x40
    moved = (r.left, r.top, r.right - r.left, r.bottom - r.top) != (m.x, m.y, m.w, m.h)
    flags = SWP_NOACTIVATE | SWP_SHOWWINDOW | (SWP_FRAMECHANGED if bare != style or moved
                                               else 0)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, m.x, m.y, m.w, m.h, flags)


def _hold(m: Monitor, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            hwnd = _find_window()
            if hwnd:
                _pin(hwnd, m)
        except Exception:
            pass
        stop.wait(1.0)


def open_window(url: str, screen: int = 2, kiosk: bool = False,
                profile: Path | None = None) -> threading.Event | None:
    """Ouvre l'interface. Renvoie un drapeau à lever pour la libérer (à l'arrêt)."""
    if os.name != "nt":
        import webbrowser

        webbrowser.open(url)
        return None
    mons = monitors()
    target = pick(mons, screen)
    edge = _edge()
    if edge is None:
        os.startfile(url)        # type: ignore[attr-defined]
        return None
    cmd = [str(edge), f"--app={url}", "--no-first-run", "--no-default-browser-check",
           "--autoplay-policy=no-user-gesture-required", "--disable-features=Translate"]
    if profile is not None:
        profile.mkdir(parents=True, exist_ok=True)
        cmd.append(f"--user-data-dir={profile}")
    if target is not None:
        cmd += [f"--window-position={target.x},{target.y}",
                f"--window-size={target.w},{target.h}"]
    if kiosk:
        cmd.append("--start-fullscreen")
    subprocess.Popen(cmd, close_fds=True, creationflags=getattr(subprocess,
                                                                "CREATE_NO_WINDOW", 0))
    if target is None or target.primary or len(mons) < 2:
        return None              # pas de projecteur : ne jamais bloquer l'écran principal
    stop = threading.Event()
    threading.Thread(target=_hold, args=(target, stop), name="valdar-ecran2",
                     daemon=True).start()
    return stop
