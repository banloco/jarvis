"""The "Iron Man" look of the window: colors and the animated arc reactor.

The reactor is redrawn about 30 times a second on a Tkinter Canvas. Tkinter has no
transparency, so the glow is faked with concentric circles whose color fades from the
background to the reactor color.
"""
import math
import random
import time
import tkinter as tk

# --- Colors ---
BG = "#05080f"          # midnight blue background
PANEL = "#0a1220"       # panels
CYAN = "#00d4ff"        # main color
CYAN_DIM = "#0e4a5c"    # subtle lines and borders
WHITE = "#e8fbff"
TEXT_DIM = "#5f8a99"
USER = "#9fe8c9"        # your messages
ERROR = "#ff4d5e"
FONT = "Consolas"       # monospace, control-room style

# Reactor color depending on what Jarvis is doing
STATE_COLORS = {"boot": CYAN, "idle": CYAN, "listening": "#7ff3ff", "thinking": CYAN,
                "speaking": "#5fe6ff", "error": ERROR}


def make_icon(size=64):
    """Jarvis's icon (a small arc reactor) as a PIL image, for the tray and the shortcuts.
    Drawn big and then scaled down, which gives smooth edges."""
    from PIL import Image, ImageDraw
    big = size * 4
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = big / 2
    ring = lambda r, **kw: d.ellipse((c - r, c - r, c + r, c + r), **kw)
    ring(big * 0.48, fill=BG)                                   # round midnight blue background
    ring(big * 0.44, outline=CYAN, width=int(big * 0.03))      # outer ring
    for k in range(10):                                         # coils
        d.arc((c - big * 0.33, c - big * 0.33, c + big * 0.33, c + big * 0.33),
              start=k * 36 + 6, end=k * 36 + 30, fill=CYAN, width=int(big * 0.09))
    ring(big * 0.2, fill=CYAN)                                  # core
    ring(big * 0.12, fill=WHITE)
    return img.resize((size, size), Image.LANCZOS)


def blend(color_a, color_b, t):
    """Mix two "#rrggbb" colors: t = 0 -> a, t = 1 -> b."""
    a = [int(color_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(color_b[i:i + 2], 16) for i in (1, 3, 5)]
    t = max(0.0, min(1.0, t))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


class ArcReactor(tk.Canvas):
    """Animated arc reactor. get_state() is called on every frame and should return
    "idle", "listening", "thinking", "speaking" or "error"."""

    FPS = 30       # frames per second while Jarvis listens, thinks or talks
    IDLE_FPS = 12  # at rest the animation is slow, no need to redraw that often

    def __init__(self, master, size=260, get_state=lambda: "idle"):
        super().__init__(master, width=size, height=size, bg=BG, highlightthickness=0)
        self.size, self.c = size, size / 2
        self.get_state = get_state
        self.start = time.monotonic()
        self.rotation = 0.0         # angle of the outer rings (degrees)
        self.sweep = 0.0            # angle of the "thinking" arc
        self.level = 0.0            # smoothed core brightness (0 to 1)
        self._tick()

    # --- Animation loop ---

    def _tick(self):
        now = time.monotonic() - self.start
        state = "boot" if now < 1.5 else self.get_state()
        fps = self.IDLE_FPS if state == "idle" else self.FPS
        self._update_motion(state, now, fps)
        self._draw(state, now)
        self.after(1000 // fps, self._tick)

    def _update_motion(self, state, now, fps):
        """Speed and brightness for the current state. Speeds are in degrees per second,
        so the rotation looks the same whatever the frame rate."""
        speed = {"idle": 12, "listening": 36, "thinking": 90, "speaking": 24}.get(state, 12)
        self.rotation = (self.rotation + speed / fps) % 360
        if state == "thinking":
            self.sweep = (self.sweep + 270 / fps) % 360

        if state == "speaking":      # the core "vibrates" like a voice (stacked sines + a bit of noise)
            target = 0.55 + 0.25 * math.sin(now * 17) * math.sin(now * 5.3) + random.uniform(0, 0.2)
        elif state == "listening":   # very bright, breathing slightly
            target = 0.9 + 0.1 * math.sin(now * 6)
        elif state == "thinking":
            target = 0.6 + 0.15 * math.sin(now * 9)
        elif state == "error":
            target = 0.4 + 0.4 * (math.sin(now * 8) > 0)
        else:                        # idle: slow breathing
            target = 0.45 + 0.15 * math.sin(now * 1.6)
        self.level += (target - self.level) * 0.35  # smoothing: no sudden jumps

    # --- Drawing ---

    def _ring(self, r, **kw):
        c = self.c
        return self.create_oval(c - r, c - r, c + r, c + r, **kw)

    def _arc(self, r, start, extent, **kw):
        c = self.c
        return self.create_arc(c - r, c - r, c + r, c + r, start=start, extent=extent, style="arc", **kw)

    def _draw(self, state, now):
        self.delete("all")
        color = STATE_COLORS.get(state, CYAN)
        R = self.size * 0.46
        boot = min(1.0, now / 1.5)            # 0 -> 1 during the startup sequence
        R *= 0.3 + 0.7 * boot ** 0.5
        glow = self.level * boot

        # Glow: circles from biggest to smallest, each one closer to the reactor color
        for i in range(12):
            r = R * (1.0 - i * 0.035)
            self._ring(r, fill=blend(BG, color, glow * 0.02 * i), outline="")

        # Outer ring: 12 segments turning slowly
        for k in range(12):
            self._arc(R * 0.95, self.rotation + k * 30, 20, outline=blend(BG, color, 0.55 * boot), width=3)
        # Thin tick marks, turning the other way
        for k in range(48):
            a = math.radians(-self.rotation * 0.7 + k * 7.5)
            r1, r2 = R * 0.83, R * (0.87 if k % 4 else 0.9)
            self.create_line(self.c + r1 * math.cos(a), self.c + r1 * math.sin(a),
                             self.c + r2 * math.cos(a), self.c + r2 * math.sin(a),
                             fill=blend(BG, color, 0.35 * boot))
        self._ring(R * 0.8, outline=blend(BG, color, 0.4 * boot), width=1)

        # Thinking arc: a quarter circle sweeping around
        if state == "thinking":
            self._arc(R * 0.74, self.sweep, 90, outline=WHITE, width=3)
            self._arc(R * 0.74, self.sweep + 180, 40, outline=blend(BG, color, 0.8), width=3)

        # Coils: 10 thick blocks around the core (like the Mark I reactor)
        for k in range(10):
            self._arc(R * 0.58, k * 36 + 6, 24, outline=blend(CYAN_DIM, color, 0.3 + 0.7 * glow), width=14)
        self._ring(R * 0.47, outline=blend(BG, color, 0.7 * boot), width=2)

        # Core: concentric discs, whiter towards the center
        for i in range(8):
            r = R * (0.38 - i * 0.04)
            base = blend(BG, color, (0.35 + 0.65 * glow) * boot)  # never fully off, even at rest
            self._ring(r, fill=blend(base, WHITE, i / 8 * (0.3 + 0.7 * glow)), outline="")
