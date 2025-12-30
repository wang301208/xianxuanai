"""UI automation primitives for translating motor intentions into OS input events.

This module is **disabled by default** and is designed with conservative safety
controls (rate limiting + foreground/app allow-list checks) so that agents can
only emit synthetic mouse/keyboard input when explicitly enabled via config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


class UIAutomationError(RuntimeError):
    """Base error for UI automation failures."""


class UIAutomationUnavailable(UIAutomationError):
    """Raised when UI automation is disabled or a backend is unavailable."""


class UIAutomationBlocked(UIAutomationError):
    """Raised when a safety policy blocks a requested UI action."""


class UIAutomationRateLimited(UIAutomationError):
    """Raised when UI automation is rate limited."""


@dataclass(frozen=True)
class UIWindowInfo:
    """Best-effort snapshot of the currently active window."""

    title: str | None = None
    process_name: str | None = None
    pid: int | None = None
    rect: Tuple[int, int, int, int] | None = None  # (left, top, right, bottom) in screen coords


@dataclass
class UIAutomationConfig:
    """Configuration for :class:`UIAutomationController`."""

    enabled: bool = False
    backend: str = "auto"  # auto | pyautogui | windows_native | xdotool
    dry_run: bool = True

    # Throttling
    min_interval_s: float = 0.25
    max_actions_per_minute: int = 120
    throttle_strategy: str = "block"  # block | sleep

    # Scope restrictions
    require_foreground: bool = True
    require_allowlist_when_live: bool = True
    allowed_window_title_substrings: Tuple[str, ...] = ()
    allowed_process_names: Tuple[str, ...] = ()
    require_within_active_window: bool = False
    allowed_screen_region: Tuple[int, int, int, int] | None = None  # (l, t, r, b)

    # Default timing for typing/clicks
    key_interval_s: float = 0.0
    click_interval_s: float = 0.0

    # Optional mapping for higher-level intentions -> UI actions (used by motor control).
    command_map: Dict[str, Any] = field(default_factory=dict)


def _normalize_process_name(name: str) -> str:
    lowered = name.strip().lower()
    return lowered[:-4] if lowered.endswith(".exe") else lowered


def _contains_any(haystack: str, needles: Sequence[str]) -> bool:
    lowered = haystack.lower()
    return any(needle.lower() in lowered for needle in needles if needle)


class _RateLimiter:
    def __init__(self, *, min_interval_s: float, max_actions_per_minute: int) -> None:
        self._min_interval_s = max(0.0, float(min_interval_s))
        self._max_per_min = max(0, int(max_actions_per_minute))
        self._last_action_t = 0.0
        self._window_start_t = time.monotonic()
        self._window_count = 0

    def claim(self, *, throttle_strategy: str) -> None:
        now = time.monotonic()

        # Per-minute cap
        if self._max_per_min > 0:
            if now - self._window_start_t >= 60.0:
                self._window_start_t = now
                self._window_count = 0
            if self._window_count >= self._max_per_min:
                raise UIAutomationRateLimited("ui_rate_limited_per_minute")

        # Min interval
        elapsed = now - self._last_action_t
        if elapsed < self._min_interval_s:
            if throttle_strategy == "sleep":
                time.sleep(max(0.0, self._min_interval_s - elapsed))
            else:
                raise UIAutomationRateLimited("ui_rate_limited_min_interval")

        self._last_action_t = time.monotonic()
        self._window_count += 1


class UIBackend:
    """Backend interface for emitting UI events."""

    name: str = "base"

    def move_mouse(self, x: int, y: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def click(self, *, button: str = "left", clicks: int = 1, interval_s: float = 0.0) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def type_text(self, text: str, *, interval_s: float = 0.0) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def press_key(self, key: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def hotkey(self, keys: Sequence[str]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def scroll(self, amount: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def activate_window(self, *, title_contains: str) -> bool:  # pragma: no cover - optional
        return False


class NoOpBackend(UIBackend):
    """A backend that records actions but performs no real UI interaction."""

    name = "noop"

    def __init__(self) -> None:
        self.actions: List[Dict[str, Any]] = []

    def _record(self, action: Dict[str, Any]) -> None:
        self.actions.append({"t": time.time(), **action})

    def move_mouse(self, x: int, y: int) -> None:
        self._record({"action": "move_mouse", "x": int(x), "y": int(y)})

    def click(self, *, button: str = "left", clicks: int = 1, interval_s: float = 0.0) -> None:
        self._record({"action": "click", "button": button, "clicks": int(clicks), "interval_s": float(interval_s)})

    def type_text(self, text: str, *, interval_s: float = 0.0) -> None:
        self._record({"action": "type_text", "text": text, "interval_s": float(interval_s)})

    def press_key(self, key: str) -> None:
        self._record({"action": "press_key", "key": key})

    def hotkey(self, keys: Sequence[str]) -> None:
        self._record({"action": "hotkey", "keys": list(keys)})

    def scroll(self, amount: int) -> None:
        self._record({"action": "scroll", "amount": int(amount)})

    def activate_window(self, *, title_contains: str) -> bool:
        self._record({"action": "activate_window", "title_contains": title_contains})
        return True


class PyAutoGUIBackend(UIBackend):
    name = "pyautogui"

    def __init__(self) -> None:
        try:
            import pyautogui  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise UIAutomationUnavailable("pyautogui_not_available") from exc
        self._pyautogui = pyautogui

    def move_mouse(self, x: int, y: int) -> None:
        self._pyautogui.moveTo(int(x), int(y))

    def click(self, *, button: str = "left", clicks: int = 1, interval_s: float = 0.0) -> None:
        self._pyautogui.click(button=button, clicks=int(clicks), interval=float(interval_s))

    def type_text(self, text: str, *, interval_s: float = 0.0) -> None:
        self._pyautogui.typewrite(str(text), interval=float(interval_s))

    def press_key(self, key: str) -> None:
        self._pyautogui.press(str(key))

    def hotkey(self, keys: Sequence[str]) -> None:
        self._pyautogui.hotkey(*[str(k) for k in keys])

    def scroll(self, amount: int) -> None:
        self._pyautogui.scroll(int(amount))


class WindowsNativeBackend(UIBackend):
    name = "windows_native"

    def __init__(self) -> None:
        if sys.platform != "win32":  # pragma: no cover - platform specific
            raise UIAutomationUnavailable("windows_native_backend_requires_win32")

        import ctypes
        from ctypes import wintypes

        ULONG_PTR = wintypes.ULONG_PTR

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            ]

        class INPUTUNION(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _anonymous_ = ("union",)
            _fields_ = [("type", wintypes.DWORD), ("union", INPUTUNION)]

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._INPUT = INPUT
        self._KEYBDINPUT = KEYBDINPUT

        self._INPUT_KEYBOARD = 1
        self._KEYEVENTF_KEYUP = 0x0002
        self._KEYEVENTF_UNICODE = 0x0004
        self._KEYEVENTF_EXTENDEDKEY = 0x0001

        self._user32.SetCursorPos.argtypes = [wintypes.INT, wintypes.INT]
        self._user32.SetCursorPos.restype = wintypes.BOOL
        self._user32.mouse_event.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ULONG_PTR]
        self._user32.mouse_event.restype = None
        self._user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        self._user32.SendInput.restype = wintypes.UINT

    # ------------------------------ mouse ------------------------------
    def move_mouse(self, x: int, y: int) -> None:
        ok = bool(self._user32.SetCursorPos(int(x), int(y)))
        if not ok:
            raise UIAutomationUnavailable(f"setcursorpos_failed:{self._ctypes.get_last_error()}")

    def _mouse_event(self, flags: int, data: int = 0) -> None:
        self._user32.mouse_event(int(flags), 0, 0, int(data), 0)

    def click(self, *, button: str = "left", clicks: int = 1, interval_s: float = 0.0) -> None:
        button_key = str(button or "left").lower()
        if button_key in ("left", "l"):
            down, up = 0x0002, 0x0004
        elif button_key in ("right", "r"):
            down, up = 0x0008, 0x0010
        elif button_key in ("middle", "m"):
            down, up = 0x0020, 0x0040
        else:
            raise UIAutomationBlocked(f"unsupported_mouse_button:{button}")

        total = max(1, int(clicks))
        for idx in range(total):
            self._mouse_event(down)
            self._mouse_event(up)
            if interval_s and idx < total - 1:
                time.sleep(max(0.0, float(interval_s)))

    def scroll(self, amount: int) -> None:
        delta = int(amount)
        if abs(delta) < 50:
            delta *= 120
        self._mouse_event(0x0800, data=delta)

    # ----------------------------- keyboard ----------------------------
    def _send_inputs(self, inputs: Sequence[Any]) -> None:
        arr = (self._INPUT * len(inputs))(*inputs)
        sent = int(self._user32.SendInput(len(inputs), arr, self._ctypes.sizeof(self._INPUT)))
        if sent != len(inputs):
            raise UIAutomationUnavailable(f"sendinput_failed:{self._ctypes.get_last_error()}")

    def type_text(self, text: str, *, interval_s: float = 0.0) -> None:
        for idx, ch in enumerate(str(text)):
            ki_down = self._KEYBDINPUT(
                wVk=0,
                wScan=self._wintypes.WORD(ord(ch)),
                dwFlags=self._KEYEVENTF_UNICODE,
                time=0,
                dwExtraInfo=0,
            )
            ki_up = self._KEYBDINPUT(
                wVk=0,
                wScan=self._wintypes.WORD(ord(ch)),
                dwFlags=self._KEYEVENTF_UNICODE | self._KEYEVENTF_KEYUP,
                time=0,
                dwExtraInfo=0,
            )
            self._send_inputs([self._INPUT(type=self._INPUT_KEYBOARD, ki=ki_down), self._INPUT(type=self._INPUT_KEYBOARD, ki=ki_up)])
            if interval_s and idx < len(text) - 1:
                time.sleep(max(0.0, float(interval_s)))

    def _vk_from_key(self, key: str) -> Tuple[int, int]:
        key_norm = str(key).strip().lower()
        ext = self._KEYEVENTF_EXTENDEDKEY
        vk_map: Dict[str, Tuple[int, int]] = {
            "backspace": (0x08, 0),
            "tab": (0x09, 0),
            "enter": (0x0D, 0),
            "return": (0x0D, 0),
            "esc": (0x1B, 0),
            "escape": (0x1B, 0),
            "space": (0x20, 0),
            "left": (0x25, ext),
            "up": (0x26, ext),
            "right": (0x27, ext),
            "down": (0x28, ext),
            "home": (0x24, ext),
            "end": (0x23, ext),
            "pageup": (0x21, ext),
            "pagedown": (0x22, ext),
            "insert": (0x2D, ext),
            "delete": (0x2E, ext),
            "ctrl": (0x11, 0),
            "control": (0x11, 0),
            "shift": (0x10, 0),
            "alt": (0x12, 0),
            "menu": (0x12, 0),
            "win": (0x5B, ext),
            "meta": (0x5B, ext),
        }
        if key_norm in vk_map:
            return vk_map[key_norm]
        if key_norm.startswith("f") and key_norm[1:].isdigit():
            idx = int(key_norm[1:])
            if 1 <= idx <= 24:
                return (0x6F + idx, 0)  # VK_F1 == 0x70
        if len(key_norm) == 1:
            ch = key_norm.upper()
            if "A" <= ch <= "Z" or "0" <= ch <= "9":
                return (ord(ch), 0)
        raise UIAutomationBlocked(f"unsupported_key:{key}")

    def _vk_input(self, vk: int, flags: int) -> Any:
        ki = self._KEYBDINPUT(
            wVk=self._wintypes.WORD(vk),
            wScan=0,
            dwFlags=self._wintypes.DWORD(flags),
            time=0,
            dwExtraInfo=0,
        )
        return self._INPUT(type=self._INPUT_KEYBOARD, ki=ki)

    def press_key(self, key: str) -> None:
        vk, flags = self._vk_from_key(key)
        self._send_inputs([self._vk_input(vk, flags), self._vk_input(vk, flags | self._KEYEVENTF_KEYUP)])

    def hotkey(self, keys: Sequence[str]) -> None:
        if not keys:
            raise UIAutomationBlocked("hotkey_requires_keys")
        vk_flags = [self._vk_from_key(k) for k in keys]
        downs = [self._vk_input(vk, flags) for vk, flags in vk_flags]
        ups = [self._vk_input(vk, flags | self._KEYEVENTF_KEYUP) for vk, flags in reversed(vk_flags)]
        self._send_inputs(downs + ups)

    # ------------------------------ window -----------------------------
    def activate_window(self, *, title_contains: str) -> bool:
        return _windows_activate_window_by_title(title_contains) is not None


class XDoToolBackend(UIBackend):
    name = "xdotool"

    def __init__(self) -> None:
        if shutil.which("xdotool") is None:  # pragma: no cover - depends on system command
            raise UIAutomationUnavailable("xdotool_not_found")

    def _run(self, args: List[str]) -> None:
        completed = subprocess.run(["xdotool", *args], capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise UIAutomationUnavailable(f"xdotool_failed:{(completed.stderr or '').strip()}")

    def move_mouse(self, x: int, y: int) -> None:
        self._run(["mousemove", str(int(x)), str(int(y))])

    def click(self, *, button: str = "left", clicks: int = 1, interval_s: float = 0.0) -> None:
        mapping = {"left": "1", "middle": "2", "right": "3"}
        btn = mapping.get(str(button).lower())
        if btn is None:
            raise UIAutomationBlocked(f"unsupported_mouse_button:{button}")
        total = max(1, int(clicks))
        for idx in range(total):
            self._run(["click", btn])
            if interval_s and idx < total - 1:
                time.sleep(max(0.0, float(interval_s)))

    def type_text(self, text: str, *, interval_s: float = 0.0) -> None:
        delay_ms = max(0, int(float(interval_s) * 1000))
        args = ["type"]
        if delay_ms:
            args.extend(["--delay", str(delay_ms)])
        args.append(str(text))
        self._run(args)

    def press_key(self, key: str) -> None:
        self._run(["key", str(key)])

    def hotkey(self, keys: Sequence[str]) -> None:
        if not keys:
            raise UIAutomationBlocked("hotkey_requires_keys")
        combo = "+".join(str(k) for k in keys)
        self._run(["key", combo])

    def scroll(self, amount: int) -> None:
        clicks = int(abs(amount))
        if clicks <= 0:
            return
        btn = "4" if amount > 0 else "5"
        for _ in range(clicks):
            self._run(["click", btn])


def _windows_cursor_position() -> Optional[Tuple[int, int]]:
    if sys.platform != "win32":  # pragma: no cover - platform specific
        return None

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    pt = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(pt)):
        return None
    return int(pt.x), int(pt.y)


def _windows_active_window_info() -> Optional[UIWindowInfo]:
    if sys.platform != "win32":  # pragma: no cover - platform specific
        return None

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)

    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None

    title = None
    length = int(user32.GetWindowTextLengthW(hwnd))
    if length > 0:
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value

    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    pid_int = int(pid.value) if pid.value else None

    rect_tuple = None
    rect = wintypes.RECT()
    if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        rect_tuple = (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))

    process_name = None
    if pid_int:
        try:  # pragma: no cover - optional dependency path
            import psutil  # type: ignore

            process_name = psutil.Process(pid_int).name()
        except Exception:
            process_name = None

    return UIWindowInfo(title=title, process_name=process_name, pid=pid_int, rect=rect_tuple)


def _windows_activate_window_by_title(title_contains: str) -> Optional[UIWindowInfo]:
    if sys.platform != "win32":  # pragma: no cover - platform specific
        return None

    query = str(title_contains or "").strip()
    if not query:
        return None

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)

    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    matches: List[wintypes.HWND] = []

    def callback(hwnd: wintypes.HWND, lparam: wintypes.LPARAM) -> wintypes.BOOL:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = int(user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value or ""
        if query.lower() in title.lower():
            matches.append(hwnd)
            return False
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    if not matches:
        return None

    hwnd = matches[0]
    try:
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
    except Exception:
        return None

    return _windows_active_window_info()


def _resolve_backend(config: UIAutomationConfig) -> UIBackend:
    backend = (config.backend or "auto").strip().lower()

    if config.dry_run:
        return NoOpBackend()

    if backend == "auto":
        try:
            return PyAutoGUIBackend()
        except UIAutomationUnavailable:
            pass
        if sys.platform == "win32":
            return WindowsNativeBackend()
        if shutil.which("xdotool") is not None:
            return XDoToolBackend()
        raise UIAutomationUnavailable("no_ui_backend_available")

    if backend == "pyautogui":
        return PyAutoGUIBackend()
    if backend in ("windows", "windows_native", "win32"):
        return WindowsNativeBackend()
    if backend == "xdotool":
        return XDoToolBackend()

    raise UIAutomationUnavailable(f"unknown_ui_backend:{backend}")


class UIAutomationController:
    """Safety-gated UI automation façade used by motor control systems."""

    def __init__(self, config: UIAutomationConfig | Dict[str, Any] | None = None) -> None:
        if isinstance(config, dict):
            cfg = UIAutomationConfig(**config)
        else:
            cfg = config or UIAutomationConfig()

        # Normalize allowlists (accept lists in JSON/YAML configs).
        cfg.allowed_window_title_substrings = tuple(cfg.allowed_window_title_substrings or ())
        cfg.allowed_process_names = tuple(cfg.allowed_process_names or ())
        cfg.throttle_strategy = str(cfg.throttle_strategy or "block").lower()
        if cfg.throttle_strategy not in ("block", "sleep"):
            cfg.throttle_strategy = "block"
        self.config = cfg

        self._limiter = _RateLimiter(
            min_interval_s=self.config.min_interval_s,
            max_actions_per_minute=self.config.max_actions_per_minute,
        )
        self._backend = _resolve_backend(self.config)

    @property
    def backend_name(self) -> str:
        return getattr(self._backend, "name", "unknown")

    def active_window(self) -> Optional[UIWindowInfo]:
        return _windows_active_window_info() if sys.platform == "win32" else None

    # ------------------------------------------------------------------
    # Public action helpers (requested API)
    # ------------------------------------------------------------------
    def move_mouse(self, x: int, y: int, *, relative_to: str = "screen") -> None:
        self._execute_with_policy({"action": "move_mouse", "x": x, "y": y, "relative_to": relative_to})

    def click(
        self,
        button: str = "left",
        *,
        clicks: int = 1,
        interval_s: float | None = None,
        x: int | None = None,
        y: int | None = None,
        relative_to: str = "screen",
    ) -> None:
        payload: Dict[str, Any] = {
            "action": "click",
            "button": button,
            "clicks": clicks,
            "interval_s": self.config.click_interval_s if interval_s is None else float(interval_s),
            "relative_to": relative_to,
        }
        if x is not None and y is not None:
            payload["x"] = int(x)
            payload["y"] = int(y)
        self._execute_with_policy(payload)

    def type_text(self, text: str, *, interval_s: float | None = None) -> None:
        self._execute_with_policy(
            {
                "action": "type_text",
                "text": text,
                "interval_s": self.config.key_interval_s if interval_s is None else float(interval_s),
            }
        )

    def press_key(self, key: str) -> None:
        self._execute_with_policy({"action": "press_key", "key": key})

    def hotkey(self, keys: Sequence[str]) -> None:
        self._execute_with_policy({"action": "hotkey", "keys": list(keys)})

    def scroll(self, amount: int) -> None:
        self._execute_with_policy({"action": "scroll", "amount": int(amount)})

    def activate_window(self, *, title_contains: str) -> bool:
        result = self._execute_with_policy({"action": "activate_window", "title_contains": title_contains})
        return bool(result.get("ok"))

    # ------------------------------------------------------------------
    def execute_actions(self, actions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a batch of actions, returning per-action results."""

        results: List[Dict[str, Any]] = []
        for action in actions:
            try:
                results.append(self._execute_with_policy(action))
            except UIAutomationError as exc:
                results.append({"ok": False, "error": str(exc), "action": dict(action)})
        return results

    # ------------------------------------------------------------------
    def _ensure_enabled(self) -> None:
        if not self.config.enabled:
            raise UIAutomationUnavailable("ui_automation_disabled")
        if os.getenv("BSS_UI_AUTOMATION_DISABLE") in ("1", "true", "TRUE"):
            raise UIAutomationUnavailable("ui_automation_disabled_by_env")
        if self.config.dry_run:
            return
        if self.config.require_allowlist_when_live and not (
            self.config.allowed_window_title_substrings or self.config.allowed_process_names
        ):
            raise UIAutomationBlocked("ui_allowlist_required_when_live")

    def _assert_scope(
        self,
        *,
        x: int | None = None,
        y: int | None = None,
    ) -> Optional[UIWindowInfo]:
        window = self.active_window() if self.config.require_foreground else None

        if self.config.require_foreground and window is None:
            raise UIAutomationBlocked("foreground_window_unavailable")

        if window is not None:
            if self.config.allowed_window_title_substrings:
                if not window.title:
                    raise UIAutomationBlocked("active_window_title_unavailable")
                if not _contains_any(window.title, self.config.allowed_window_title_substrings):
                    raise UIAutomationBlocked("active_window_title_not_allowed")

            if self.config.allowed_process_names:
                if not window.process_name:
                    raise UIAutomationBlocked("active_process_unavailable")
                proc = _normalize_process_name(window.process_name)
                allowed = {_normalize_process_name(p) for p in self.config.allowed_process_names if p}
                if proc not in allowed:
                    raise UIAutomationBlocked("active_process_not_allowed")

        if x is not None and y is not None:
            if self.config.allowed_screen_region is not None:
                l, t, r, b = self.config.allowed_screen_region
                if not (l <= x <= r and t <= y <= b):
                    raise UIAutomationBlocked("target_outside_allowed_screen_region")

            if self.config.require_within_active_window and window and window.rect:
                l, t, r, b = window.rect
                if not (l <= x <= r and t <= y <= b):
                    raise UIAutomationBlocked("target_outside_active_window")

        return window

    def _resolve_coordinates(self, payload: Dict[str, Any]) -> Tuple[int | None, int | None]:
        if "x" not in payload or "y" not in payload:
            return None, None
        x = int(payload.get("x"))
        y = int(payload.get("y"))
        relative_to = str(payload.get("relative_to") or "screen").lower()
        if relative_to in ("active_window", "window", "foreground"):
            window = self.active_window()
            if window is None or window.rect is None:
                raise UIAutomationBlocked("active_window_rect_unavailable")
            l, t, _, _ = window.rect
            x += int(l)
            y += int(t)
        return x, y

    def _execute_with_policy(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_enabled()

        action = str(payload.get("action") or "").strip().lower()
        if not action:
            raise UIAutomationBlocked("missing_action")

        self._limiter.claim(throttle_strategy=self.config.throttle_strategy)

        x, y = self._resolve_coordinates(payload)

        # For pointer actions without explicit coords, apply coordinate policy using the current cursor.
        policy_x, policy_y = x, y
        if action in ("click", "scroll") and (policy_x is None or policy_y is None) and (
            self.config.allowed_screen_region is not None or self.config.require_within_active_window
        ):
            cursor = _windows_cursor_position()
            if cursor is None:
                raise UIAutomationBlocked("cursor_position_unavailable_for_scope_check")
            policy_x, policy_y = cursor

        window = self._assert_scope(x=policy_x, y=policy_y)

        if action == "move_mouse":
            if x is None or y is None:
                raise UIAutomationBlocked("move_mouse_requires_coordinates")
            self._backend.move_mouse(x, y)
            return {"ok": True, "action": "move_mouse", "x": x, "y": y, "backend": self.backend_name}

        if action == "click":
            if x is not None and y is not None:
                self._backend.move_mouse(x, y)
            self._backend.click(
                button=str(payload.get("button") or "left"),
                clicks=int(payload.get("clicks", 1)),
                interval_s=float(payload.get("interval_s", self.config.click_interval_s)),
            )
            return {"ok": True, "action": "click", "x": x, "y": y, "backend": self.backend_name}

        if action == "type_text":
            self._backend.type_text(
                str(payload.get("text") or ""),
                interval_s=float(payload.get("interval_s", self.config.key_interval_s)),
            )
            return {
                "ok": True,
                "action": "type_text",
                "backend": self.backend_name,
                "window_title": getattr(window, "title", None),
                "process_name": getattr(window, "process_name", None),
            }

        if action == "press_key":
            self._backend.press_key(str(payload.get("key") or ""))
            return {"ok": True, "action": "press_key", "backend": self.backend_name}

        if action == "hotkey":
            keys = payload.get("keys") or payload.get("key")
            if isinstance(keys, str):
                seq = [keys]
            elif isinstance(keys, list):
                seq = [str(k) for k in keys]
            else:
                raise UIAutomationBlocked("hotkey_requires_keys")
            self._backend.hotkey(seq)
            return {"ok": True, "action": "hotkey", "keys": seq, "backend": self.backend_name}

        if action == "scroll":
            self._backend.scroll(int(payload.get("amount", 0)))
            return {"ok": True, "action": "scroll", "backend": self.backend_name}

        if action == "activate_window":
            title_contains = str(payload.get("title_contains") or payload.get("title") or "").strip()
            if not title_contains:
                raise UIAutomationBlocked("activate_window_requires_title_contains")
            ok = bool(self._backend.activate_window(title_contains=title_contains))
            return {"ok": ok, "action": "activate_window", "title_contains": title_contains, "backend": self.backend_name}

        raise UIAutomationBlocked(f"unsupported_action:{action}")


__all__ = [
    "UIAutomationError",
    "UIAutomationUnavailable",
    "UIAutomationBlocked",
    "UIAutomationRateLimited",
    "UIWindowInfo",
    "UIAutomationConfig",
    "UIAutomationController",
]
