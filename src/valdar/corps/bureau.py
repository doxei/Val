"""L'écran, la souris et le clavier du PC, vus et utilisés par Valdar.

Il voit de deux façons à la fois :
- une **capture d'écran** (réduite), avec un numéro posé sur chaque bouton ou champ ;
- la **liste des fenêtres et des contrôles** de la fenêtre active, lue par l'accessibilité de
  Windows (UI Automation) : nom, type, position. C'est ce qui lui permet de viser juste
  (« clique sur 12 », ou « clique sur Enregistrer »).

Windows : mss (capture), pywinauto (accessibilité, souris, clavier), Pillow.
"""
from __future__ import annotations

import contextlib
import io
import os
import time
from dataclasses import dataclass, field
from typing import Any

INTERACTIVE = {"Button", "Edit", "MenuItem", "ListItem", "TabItem", "Hyperlink", "CheckBox",
               "RadioButton", "ComboBox", "TreeItem", "SplitButton", "MenuBar", "Document",
               "DataItem", "Spinner", "Slider"}


class BodyError(RuntimeError):
    pass


@dataclass
class Control:
    id: int
    name: str
    kind: str
    rect: tuple[int, int, int, int]      # gauche, haut, droite, bas (pixels de l'écran)
    enabled: bool = True
    password: bool = False
    ref: Any = field(default=None, repr=False)

    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.rect
        return (x1 + x2) // 2, (y1 + y2) // 2

    def line(self) -> str:
        flags = ("" if self.enabled else ", grisé") + (", mot de passe" if self.password else "")
        return f"[{self.id}] {self.kind} « {self.name or '—'} »{flags}"


@dataclass
class Window:
    title: str
    rect: tuple[int, int, int, int]
    active: bool = False
    ref: Any = field(default=None, repr=False)


class Desktop:
    """Interface : la vraie (Windows) et une fausse pour les tests."""

    def screen_size(self) -> tuple[int, int]:
        raise NotImplementedError

    def screenshot(self) -> Any:            # image PIL, taille réelle de l'écran
        raise NotImplementedError

    def windows(self) -> list[Window]:
        raise NotImplementedError

    def controls(self, limit: int = 80) -> list[Control]:
        raise NotImplementedError

    def focus(self, window: Window) -> None:
        raise NotImplementedError

    def click(self, x: int, y: int, double: bool = False, right: bool = False) -> None:
        raise NotImplementedError

    def click_control(self, c: Control, double: bool = False, right: bool = False) -> None:
        x, y = c.center()
        self.click(x, y, double=double, right=right)

    def focused_is_password(self) -> bool:
        raise NotImplementedError

    def type_text(self, text: str) -> None:
        raise NotImplementedError

    def keys(self, combo: str) -> None:
        raise NotImplementedError

    def scroll(self, x: int, y: int, clicks: int) -> None:
        raise NotImplementedError

    def open(self, target: str) -> None:
        raise NotImplementedError


# ------------------------------------------------------------------ Windows
_SPECIAL = set("{}+^%~()[]")
_KEYS = {"ctrl": "^", "control": "^", "alt": "%", "shift": "+"}
_NAMED = {"enter": "{ENTER}", "entrée": "{ENTER}", "return": "{ENTER}", "tab": "{TAB}",
          "escape": "{ESC}", "esc": "{ESC}", "échap": "{ESC}", "up": "{UP}", "down": "{DOWN}",
          "left": "{LEFT}", "right": "{RIGHT}", "home": "{HOME}", "end": "{END}",
          "pageup": "{PGUP}", "pagedown": "{PGDN}", "delete": "{DELETE}", "suppr": "{DELETE}",
          "backspace": "{BACKSPACE}", "space": "{SPACE}", "espace": "{SPACE}",
          "win": "{VK_LWIN}", "windows": "{VK_LWIN}", "f5": "{F5}", "f2": "{F2}",
          "f11": "{F11}", "f4": "{F4}"}


def escape_text(text: str) -> str:
    """Texte → séquence pywinauto (les caractères spéciaux entre accolades)."""
    out = []
    for ch in text:
        if ch in _SPECIAL:
            out.append("{" + ch + "}")
        elif ch == "\n":
            out.append("{ENTER}")
        elif ch == "\t":
            out.append("{TAB}")
        else:
            out.append(ch)
    return "".join(out)


def combo_keys(combo: str) -> str:
    """« ctrl+shift+s » → « ^+s » (pywinauto)."""
    parts = [p.strip().lower() for p in combo.replace(" ", "").split("+") if p.strip()]
    if not parts:
        raise BodyError("combinaison vide")
    mods, key = parts[:-1], parts[-1]
    prefix = ""
    for m in mods:
        if m in ("win", "windows"):
            return "{VK_LWIN down}" + (_NAMED.get(key) or escape_text(key)) + "{VK_LWIN up}"
        if m not in _KEYS:
            raise BodyError(f"touche de modification inconnue : {m}")
        prefix += _KEYS[m]
    body = _NAMED.get(key) or (key if len(key) == 1 else None)
    if body is None:
        if key.startswith("f") and key[1:].isdigit():
            body = "{" + key.upper() + "}"
        else:
            raise BodyError(f"touche inconnue : {key}")
    if prefix and len(body) == 1 and body in _SPECIAL:
        body = "{" + body + "}"
    return prefix + (body if not prefix or len(body) > 1 else body.lower())


class WindowsDesktop(Desktop):
    def __init__(self) -> None:
        if os.name != "nt":
            raise BodyError("le corps numérique ne marche que sous Windows")
        try:
            import mss  # noqa: F401
            import pywinauto  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception as exc:
            raise BodyError(f"modules manquants ({exc}) : lance tools\\valdar_corps.bat") \
                from exc
        from pywinauto import Desktop as UIA

        self._uia = UIA(backend="uia")

    def screen_size(self) -> tuple[int, int]:
        import mss

        with mss.mss() as m:
            mon = m.monitors[1]
            return mon["width"], mon["height"]

    def screenshot(self) -> Any:
        import mss
        from PIL import Image

        with mss.mss() as m:
            mon = m.monitors[1]
            shot = m.grab(mon)
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def _foreground(self) -> Any:
        import win32gui

        hwnd = win32gui.GetForegroundWindow()
        return self._uia.window(handle=hwnd) if hwnd else None

    def windows(self) -> list[Window]:
        fg = self._foreground()
        fg_handle = fg.handle if fg is not None else None
        out = []
        for w in self._uia.windows():
            try:
                title = w.window_text()
                if not title or not w.is_visible():
                    continue
                r = w.rectangle()
                out.append(Window(title, (r.left, r.top, r.right, r.bottom),
                                  active=w.handle == fg_handle, ref=w))
            except Exception:
                continue
        return out

    def controls(self, limit: int = 80) -> list[Control]:
        fg = self._foreground()
        if fg is None:
            return []
        out: list[Control] = []
        deadline = time.time() + 3.0
        try:
            items = fg.descendants()
        except Exception:
            return []
        for e in items:
            if time.time() > deadline or len(out) >= limit:
                break
            try:
                info = e.element_info
                kind = info.control_type or ""
                if kind not in INTERACTIVE or not e.is_visible():
                    continue
                r = info.rectangle
                if r.right - r.left < 4 or r.bottom - r.top < 4:
                    continue
                pwd = False
                with contextlib.suppress(Exception):
                    pwd = bool(info.element.CurrentIsPassword)
                out.append(Control(len(out) + 1, (info.name or "").strip()[:80], kind,
                                   (r.left, r.top, r.right, r.bottom), e.is_enabled(), pwd,
                                   ref=e))
            except Exception:
                continue
        return out

    def focus(self, window: Window) -> None:
        window.ref.set_focus()

    def click(self, x: int, y: int, double: bool = False, right: bool = False) -> None:
        from pywinauto import mouse

        if right:
            mouse.right_click(coords=(x, y))
        elif double:
            mouse.double_click(coords=(x, y))
        else:
            mouse.click(coords=(x, y))

    def focused_is_password(self) -> bool:
        try:
            from pywinauto.uia_defines import IUIA

            el = IUIA().iuia.GetFocusedElement()
            return bool(el.CurrentIsPassword)
        except Exception:
            return False

    def type_text(self, text: str) -> None:
        from pywinauto import keyboard

        keyboard.send_keys(escape_text(text), with_spaces=True, with_tabs=True,
                           with_newlines=True, pause=0.01)

    def keys(self, combo: str) -> None:
        from pywinauto import keyboard

        keyboard.send_keys(combo_keys(combo), pause=0.02)

    def scroll(self, x: int, y: int, clicks: int) -> None:
        from pywinauto import mouse

        mouse.scroll(coords=(x, y), wheel_dist=clicks)

    def open(self, target: str) -> None:
        os.startfile(target)  # type: ignore[attr-defined]


# ------------------------------------------------------------------ la vue
def annotate(img: Any, controls: list[Control], width: int) -> tuple[bytes, float]:
    """Capture réduite à `width`, un numéro sur chaque contrôle. Renvoie (PNG, échelle)."""
    from PIL import Image, ImageDraw

    scale = min(1.0, width / img.width)
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    for c in controls:
        x1, y1, x2, y2 = (int(v * scale) for v in c.rect)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=1)
        label = str(c.id)
        draw.rectangle([x1, y1, x1 + 7 * len(label) + 4, y1 + 12], fill=(255, 64, 64))
        draw.text((x1 + 2, y1), label, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue(), scale
