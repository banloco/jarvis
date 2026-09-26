"""Interface holographique façon Iron Man, pilotée avec la main devant la webcam.

Ce que tu vois : ton image webcam assombrie et, par-dessus, des panneaux lumineux
(météo, état du PC, rappels, musique, volume, applis, dernière réponse de Jarvis)
autour d'un réacteur arc.

Comment la piloter :
- le curseur lumineux suit le point entre ton POUCE et ton INDEX ;
- PINCER (pouce contre index) = cliquer sur un bouton ;
- pincer un panneau (en dehors d'un bouton) puis bouger la main = le déplacer ;
- pincer la barre de volume puis glisser = régler le volume ;
- pincer le réacteur au centre = parler à Jarvis (si le HUD est ouvert depuis sa fenêtre) ;
- Échap, Q ou le bouton FERMER = quitter. La souris marche aussi (clic gauche = pincer).

Lancer seul :  python jarvis_holo.py            (plein écran)
               python jarvis_holo.py --fenetre  (dans une fenêtre)
Ou depuis Jarvis : bouton INTERFACE HOLO, ou menu de son icône près de l'horloge.

Comment c'est construit :
1. HandTracker (jarvis_gestures.py) donne l'image et les 21 points de la main.
2. Le curseur est lissé par un « filtre 1 euro » : il reste immobile quand la main
   tremble un peu, et suit sans retard quand elle bouge vite.
3. Le pincement a deux seuils (hystérésis) : on pince sous PINCH_ON, on relâche au-dessus
   de PINCH_OFF. Entre les deux, rien ne change : pas de clics en rafale quand on hésite.
4. Chaque image est dessinée en couches (classe Painter) : fonds translucides, traits
   lumineux avec un halo flou, textes, puis le curseur par-dessus tout.
"""
import functools
import math
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import psutil
from PIL import Image, ImageDraw, ImageFont
from jarvis_gestures import HandTracker, vision
from jarvis_reminders import reminders
from jarvis_tools import (HOME_CITY, meteo, controle_musique, couper_son, speakers,
                          ouvrir_application, ouvrir_site)

W, H = 1280, 720           # taille de l'image dessinée (agrandie ensuite à la taille de l'écran)
WINDOW = "J.A.R.V.I.S - interface holographique"
TOP_BAR = 62               # hauteur de la barre du haut (les panneaux restent en dessous)

# Main
CURSOR_GAIN = 1.35         # amplifie les mouvements : pas besoin d'aller au bord de l'image
PINCH_ON = 0.28            # pouce-index plus proches que 28 % de la paume = pincement...
PINCH_OFF = 0.42           # ... qui ne se relâche qu'au-delà de 42 % (hystérésis)
FILTER_MIN_CUTOFF = 1.0    # filtre 1 euro : plus petit = curseur plus stable mais plus mou
FILTER_BETA = 0.008        # filtre 1 euro : plus grand = moins de retard quand la main va vite

# Apparence
VIDEO_BRIGHTNESS = 0.45    # luminosité de l'image webcam en fond
FILL_ALPHA = 0.55          # opacité des fonds de panneaux
GLOW = 1.4                 # intensité du halo autour des traits lumineux
TOAST_SECONDS = 4.0        # durée d'affichage du résultat d'une action

# Données affichées
DATA_REFRESH = 1.0         # état du PC, volume, rappels : relus chaque seconde
WEATHER_REFRESH = 15 * 60  # météo : relue tous les quarts d'heure

# Couleurs (OpenCV les écrit dans l'ordre Bleu, Vert, Rouge) : la palette de jarvis_hud.py
CYAN = (255, 212, 0)
CYAN_DIM = (92, 74, 14)
WHITE = (255, 251, 232)
TEXT_DIM = (153, 138, 95)
PANEL = (32, 18, 10)
PANEL_HI = (60, 38, 16)
ORANGE = (40, 170, 255)    # ce qui est attrapé / pincé
ERROR = (94, 77, 255)
STATE_COLORS = {"listening": (255, 243, 127), "speaking": (255, 230, 95), "error": ERROR}

FONT_FILE = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "bahnschrift.ttf"
DAYS = ["LUN.", "MAR.", "MER.", "JEU.", "VEN.", "SAM.", "DIM."]


# --- Textes ---
# OpenCV ne sait pas écrire les accents : les textes sont dessinés avec PIL (police Bahnschrift),
# une seule fois, puis gardés en mémoire (cache) sous forme de masque réutilisé à chaque image.

@functools.lru_cache(maxsize=16)
def font(size, bold=False):
    try:
        f = ImageFont.truetype(str(FONT_FILE), size)
        if bold:
            f.set_variation_by_name("Bold")  # Bahnschrift contient plusieurs graisses
    except OSError:
        f = ImageFont.load_default(size)
    return f


def clean(text):
    """Retire les émojis et symboles absents de la police (ils s'afficheraient en carrés)."""
    return "".join(c for c in str(text) if ord(c) < 0x2190 or 0x2200 <= ord(c) < 0x2300).strip()


@functools.lru_cache(maxsize=512)
def text_mask(text, size, bold=False):
    """Le texte dessiné en niveaux de gris (0 = vide, 1 = plein), prêt à être « tamponné »."""
    f = font(size, bold)
    ascent, descent = f.getmetrics()
    accent = max(0, -f.getbbox("ÉÈÂ", anchor="la")[1])  # les accents des majuscules dépassent en haut
    img = Image.new("L", (max(1, int(f.getlength(text)) + 2), accent + ascent + descent))
    ImageDraw.Draw(img).text((0, accent), text, font=f, fill=255)
    return np.asarray(img, dtype=np.float32)[:, :, None] / 255


def text_width(text, size, bold=False):
    return text_mask(clean(text), size, bold).shape[1]


def fit(text, size, width, bold=False):
    """Raccourcit le texte (avec « … ») pour qu'il tienne dans `width` pixels."""
    text = clean(text)
    if text_width(text, size, bold) <= width:
        return text
    while text and text_width(text + "…", size, bold) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def wrap(text, size, width, max_lines, bold=False):
    """Découpe le texte en lignes de `width` pixels au plus (« … » s'il en reste)."""
    lines, current = [], ""
    for word in clean(text).split():
        trial = f"{current} {word}".strip()
        if current and text_width(trial, size, bold) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = fit(lines[-1] + " …", size, width, bold)
    return lines


def blit_text(img, text, x, y, size, color, bold=False, anchor="lt"):
    """Écrit le texte sur l'image. anchor : horizontal (l, c, r) + vertical (t, c, b)."""
    mask = text_mask(clean(text), size, bold)
    h, w = mask.shape[:2]
    x = int(x - {"l": 0, "c": w // 2, "r": w}[anchor[0]])
    y = int(y - {"t": 0, "c": h // 2, "b": h}[anchor[1]])
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x0 >= x1 or y0 >= y1:
        return  # entièrement hors de l'image
    a = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


# --- Dessin en couches ---

def ipt(p):
    return int(round(p[0])), int(round(p[1]))


class Painter:
    """Note tout ce qu'il faut dessiner, dans n'importe quel ordre, puis compose l'image :
    1. les fonds (translucides : l'image webcam reste visible au travers) ;
    2. les traits lumineux, recopiés sur un calque flouté qui leur fait un halo ;
    3. les textes ;
    4. le curseur et ses effets, par-dessus tout."""

    def __init__(self):
        self.fills, self.shapes, self.texts, self.top = [], [], [], []

    def fill_rect(self, x, y, w, h, color):
        self.fills.append(lambda img: cv2.rectangle(img, ipt((x, y)), ipt((x + w, y + h)), color, -1))

    def fill_circle(self, c, r, color):
        self.fills.append(lambda img: cv2.circle(img, ipt(c), int(r), color, -1, cv2.LINE_AA))

    def _shape(self, draw, glow, top):
        (self.top if top else self.shapes).append((draw, glow))

    def line(self, a, b, color, th=1, glow=True, top=False):
        self._shape(lambda img: cv2.line(img, ipt(a), ipt(b), color, th, cv2.LINE_AA), glow, top)

    def rect(self, x, y, w, h, color, th=1, glow=True):
        self._shape(lambda img: cv2.rectangle(img, ipt((x, y)), ipt((x + w, y + h)), color, th,
                                              cv2.LINE_AA), glow, False)

    def circle(self, c, r, color, th=1, glow=True, top=False):
        self._shape(lambda img: cv2.circle(img, ipt(c), int(r), color, th, cv2.LINE_AA), glow, top)

    def arc(self, c, r, start, end, color, th=1, glow=True):
        """Portion de cercle, angles en degrés (0 = à droite, sens des aiguilles d'une montre)."""
        self._shape(lambda img: cv2.ellipse(img, ipt(c), (int(r), int(r)), 0, start, end, color, th,
                                            cv2.LINE_AA), glow, False)

    def poly(self, points, color, glow=True):
        pts = np.array([ipt(p) for p in points], dtype=np.int32)
        self._shape(lambda img: cv2.fillPoly(img, [pts], color, cv2.LINE_AA), glow, False)

    def text(self, text, x, y, size, color, bold=False, anchor="lt"):
        self.texts.append((text, x, y, size, color, bold, anchor))

    def compose(self, canvas):
        overlay = canvas.copy()
        for draw in self.fills:
            draw(overlay)
        canvas = cv2.addWeighted(overlay, FILL_ALPHA, canvas, 1 - FILL_ALPHA, 0)
        glow = np.zeros_like(canvas)
        for draw, glows in self.shapes:
            draw(canvas)
            if glows:
                draw(glow)
        # Halo : le calque des traits est réduit, flouté puis ré-agrandi (4x plus rapide qu'en grand)
        small = cv2.GaussianBlur(cv2.resize(glow, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 2.5)
        canvas = cv2.addWeighted(canvas, 1.0, cv2.resize(small, (W, H)), GLOW, 0)
        for args in self.texts:
            blit_text(canvas, *args)
        for draw, _ in self.top:
            draw(canvas)
        return canvas


# --- Éléments interactifs ---

class Button:
    """Bouton : un texte ou une icône. Coordonnées relatives au panneau qui le contient."""

    def __init__(self, x, y, w, h, label="", action=None, icon=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label, self.action, self.icon = label, action, icon
        self.flash_until = 0.0  # le bouton s'allume brièvement quand on le « clique »

    def contains(self, rx, ry):
        return self.x <= rx < self.x + self.w and self.y <= ry < self.y + self.h

    def draw(self, p, ox, oy, hud, now):
        x, y, w, h = ox + self.x, oy + self.y, self.w, self.h
        hovered = hud.hover_widget is self
        flash = now < self.flash_until
        p.fill_rect(x, y, w, h, CYAN if flash else PANEL_HI if hovered else PANEL)
        p.rect(x, y, w, h, WHITE if hovered or flash else CYAN, 1, glow=hovered or flash)
        color = WHITE if hovered or flash else CYAN
        if self.icon:
            self.icon(p, x + w / 2, y + h / 2, color)
        else:
            p.text(self.label, x + w / 2, y + h / 2, 15, color, bold=True, anchor="cc")


class Slider:
    """Barre de réglage de 0 à 100 : on la pince puis on glisse la main."""

    def __init__(self, x, y, w, h, get, set_):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.get, self.set = get, set_

    def contains(self, rx, ry):  # zone de prise plus haute que la barre : plus facile à viser
        return self.x - 10 <= rx < self.x + self.w + 10 and self.y - 16 <= ry < self.y + self.h + 16

    def value_at(self, rx):
        return round(min(max((rx - self.x) / self.w, 0), 1) * 100)

    def draw(self, p, ox, oy, hud, now):
        x, y, w, h = ox + self.x, oy + self.y, self.w, self.h
        active = hud.hover_widget is self
        filled = w * self.get() / 100
        p.fill_rect(x, y, w, h, PANEL)
        p.fill_rect(x, y, filled, h, CYAN)
        p.rect(x, y, w, h, CYAN_DIM, 1, glow=False)
        p.line((x + filled, y - 7), (x + filled, y + h + 7), ORANGE if active and hud.pinched else WHITE, 3)


class Panel:
    """Panneau lumineux : titre, contenu dessiné par `content(p, x, y, w, h)`, boutons."""
    TITLE_H = 34

    def __init__(self, title, x, y, w, h, content=None, widgets=(), movable=True, frame=True):
        self.title, self.x, self.y, self.w, self.h = title, x, y, w, h
        self.content, self.widgets = content, list(widgets)
        self.movable, self.frame = movable, frame
        self.on_press = None  # action quand on pince le panneau lui-même (au lieu de le déplacer)

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def widget_at(self, px, py):
        return next((w for w in self.widgets if w.contains(px - self.x, py - self.y)), None)

    def move_to(self, x, y):
        self.x = min(max(x, 0), W - self.w)
        self.y = min(max(y, TOP_BAR), H - self.h)

    def draw(self, p, hud, now):
        x, y, w, h = self.x, self.y, self.w, self.h
        if self.frame:
            edge = ORANGE if hud.dragging is self else WHITE if hud.hover_owner is self else CYAN
            p.fill_rect(x, y, w, h, PANEL)
            p.rect(x, y, w, h, CYAN_DIM, 1, glow=False)
            for cx, cy, dx, dy in [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
                p.line((cx, cy), (cx + 18 * dx, cy), edge, 2)  # coins lumineux
                p.line((cx, cy), (cx, cy + 18 * dy), edge, 2)
            if self.title:
                p.text(self.title, x + 14, y + 8, 15, edge, bold=True)
                p.line((x + 14, y + 32), (x + w - 14, y + 32), CYAN_DIM, 1, glow=False)
        if self.content:
            self.content(p, x, y, w, h)
        for widget in self.widgets:
            widget.draw(p, x, y, hud, now)


class Reactor:
    """Le réacteur arc au centre. Le pincer = parler à Jarvis."""
    movable, widgets = False, []

    def __init__(self, hud, x, y, r):
        self.hud, self.x, self.y, self.r = hud, x, y, r
        self.on_press = hud.talk

    def contains(self, px, py):
        return math.hypot(px - self.x, py - self.y) <= self.r + 10

    def widget_at(self, px, py):
        return None

    def draw(self, p, hud, now):
        c, r = (self.x, self.y), self.r
        state = hud.get_state() if hud.get_state else "idle"
        color = STATE_COLORS.get(state, CYAN)
        speed = {"thinking": 4.0, "listening": 2.5, "speaking": 2.0}.get(state, 0.7)
        pulse = 0.5 + 0.5 * math.sin(now * (7 if state in ("listening", "speaking") else 2))
        turn = (now * 40 * speed) % 360
        p.fill_circle(c, r, PANEL)
        p.circle(c, r, color, 2)
        p.circle(c, r - 10, CYAN_DIM, 1, glow=False)
        for k in range(3):   # arcs qui tournent
            p.arc(c, r - 22, turn + k * 120, turn + k * 120 + 70, color, 3)
        for k in range(4):   # arcs extérieurs, en sens inverse
            p.arc(c, r + 14, -turn * 0.6 + k * 90, -turn * 0.6 + k * 90 + 40, CYAN_DIM, 2, glow=False)
        for k in range(10):  # bobines
            p.arc(c, r - 48, k * 36 + 6, k * 36 + 30, color, 9)
        core = r * 0.26 + 5 * pulse
        p.fill_circle(c, core + 8, color)
        p.circle(c, core, WHITE, -1)
        if hud.hover_owner is self:
            p.circle(c, r + 26, WHITE, 1)


# Icônes des boutons de musique, dessinées avec des formes simples
def icon_prev(p, cx, cy, c):
    p.poly([(cx + 10, cy - 11), (cx + 10, cy + 11), (cx - 6, cy)], c)
    p.rect(cx - 12, cy - 11, 3, 22, c, -1)


def icon_next(p, cx, cy, c):
    p.poly([(cx - 10, cy - 11), (cx - 10, cy + 11), (cx + 6, cy)], c)
    p.rect(cx + 9, cy - 11, 3, 22, c, -1)


def icon_play_pause(p, cx, cy, c):
    p.poly([(cx - 14, cy - 11), (cx - 14, cy + 11), (cx + 1, cy)], c)
    p.rect(cx + 5, cy - 11, 3, 22, c, -1)
    p.rect(cx + 12, cy - 11, 3, 22, c, -1)


# --- Lissage du curseur ---

class OneEuroFilter:
    """Filtre « 1 euro » (Casiez, Roussel et Vogel, 2012) : un lissage qui s'adapte à la vitesse.
    Main presque immobile -> lissage fort (le tremblement disparaît) ;
    main rapide -> lissage faible (le curseur ne traîne pas derrière)."""

    def __init__(self, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.value = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, value, now):
        value = np.asarray(value, dtype=float)
        if self.value is None:
            self.value, self.speed, self.time = value, np.zeros_like(value), now
            return value
        dt = max(now - self.time, 1e-3)
        self.time = now
        a = self._alpha(self.d_cutoff, dt)
        self.speed = a * (value - self.value) / dt + (1 - a) * self.speed  # vitesse lissée
        a = self._alpha(self.min_cutoff + self.beta * np.linalg.norm(self.speed), dt)
        self.value = a * value + (1 - a) * self.value
        return self.value


# --- L'interface ---

class HoloHUD:
    """Interface holographique. Tout (webcam, calcul, dessin, fenêtre) tourne dans UN thread :
    OpenCV exige que la fenêtre soit gérée par le thread qui l'a créée.

    Liens facultatifs avec la fenêtre de Jarvis (tous appelés depuis le thread du HUD) :
    on_talk()   : faire écouter Jarvis ;     get_state() : état du réacteur (idle, listening...) ;
    get_status(): texte de statut ;           get_reply() : dernière réponse de Jarvis ;
    on_close()  : le HUD est fermé ;          on_error(message) : problème (webcam...)."""

    def __init__(self, on_talk=None, get_state=None, get_status=None, get_reply=None,
                 on_close=None, on_error=None, fullscreen=True):
        self.on_talk, self.get_state, self.get_status, self.get_reply = on_talk, get_state, get_status, get_reply
        self.on_close, self.on_error, self.fullscreen = on_close, on_error, fullscreen
        self.running = False
        self.data = {"cpu": 0.0, "ram": 0.0, "disk": 0.0, "battery": None, "volume": 0, "muted": False,
                     "reminders": [], "weather": None}
        self.volume_touched = 0.0    # dernier réglage à la main (la relecture attend un peu)
        self.volume_control = None   # commande pycaw du thread du HUD (créée au premier réglage)
        self.crop = None             # comment l'image webcam est recadrée à la taille du HUD
        self.filter = OneEuroFilter()
        self.cursor, self.ratio, self.pinched = None, 1.0, False
        self.hover_owner = self.hover_widget = None
        self.drag = None             # fonction appelée pendant qu'on garde la pince (déplacer, glisser)
        self.dragging = None         # panneau en cours de déplacement
        self.mouse = None            # (x, y, bouton enfoncé) : la souris remplace la main si besoin
        self.ripples = []            # ondes affichées à chaque pincement : (x, y, instant)
        self.toast_text, self.toast_time = "", -TOAST_SECONDS
        self.fps = 0.0
        self.reactor = Reactor(self, W // 2, 282, 105)
        self.panels = self.build_panels()

    # --- Panneaux ---

    def build_panels(self):
        act = self.act
        apps = [("NAVIGATEUR", ouvrir_site, "google.com"), ("YOUTUBE", ouvrir_site, "youtube.com"),
                ("FICHIERS", ouvrir_application, "explorateur"), ("SPOTIFY", ouvrir_application, "spotify"),
                ("VS CODE", ouvrir_application, "vs code"), ("CALCULATRICE", ouvrir_application, "calculatrice")]
        return [
            Panel("", 0, 0, W, TOP_BAR, self.draw_top_bar, movable=False, frame=False,
                  widgets=[Button(W - 150, 14, 120, 34, "FERMER", self.close)]),
            Panel("MÉTÉO", 30, 84, 300, 150, self.draw_weather),
            Panel("SYSTÈME", 30, 250, 300, 180, self.draw_system),
            Panel("RAPPELS", 30, 446, 300, 170, self.draw_reminders),
            Panel("JARVIS", 400, 476, 480, 140, self.draw_reply),
            Panel("MUSIQUE", 950, 84, 300, 120, widgets=[
                Button(16, 46, 76, 56, action=lambda: act(controle_musique, "precedent"), icon=icon_prev),
                Button(112, 46, 76, 56, action=lambda: act(controle_musique, "pause"), icon=icon_play_pause),
                Button(208, 46, 76, 56, action=lambda: act(controle_musique, "suivant"), icon=icon_next)]),
            Panel("VOLUME", 950, 220, 300, 120, self.draw_volume, widgets=[
                Slider(16, 66, 196, 20, lambda: self.data["volume"], self.set_volume),
                Button(226, 52, 58, 46, "MUET", lambda: act(couper_son))]),
            Panel("APPLIS", 950, 356, 300, 260, widgets=[
                Button(16 + (i % 2) * 140, 48 + (i // 2) * 68, 128, 56, label,
                       lambda fn=fn, arg=arg: act(fn, arg))  # fn=fn : fige la valeur de CE tour de boucle
                for i, (label, fn, arg) in enumerate(apps)]),
        ]

    def draw_top_bar(self, p, x, y, w, h):
        now = datetime.now()
        p.text("J.A.R.V.I.S", 30, 14, 28, CYAN, bold=True)
        p.text("INTERFACE HOLOGRAPHIQUE", 36 + text_width("J.A.R.V.I.S", 28, True), 26, 13, TEXT_DIM)
        p.text(now.strftime("%H:%M:%S"), W - 176, 12, 24, CYAN, bold=True, anchor="rt")
        p.text(f"{DAYS[now.weekday()]} {now:%d.%m.%Y}", W - 176, 40, 12, TEXT_DIM, anchor="rt")
        p.text(f"{self.fps:.0f} IMG/S", W - 330, 26, 12, TEXT_DIM, anchor="rt")
        p.line((24, TOP_BAR), (W - 24, TOP_BAR), CYAN_DIM, 1)

    def draw_weather(self, p, x, y, w, h):
        text = self.data["weather"]
        if text is None:
            p.text("Chargement...", x + 16, y + 50, 15, TEXT_DIM)
            return
        if text.startswith("Météo indisponible"):
            for i, line in enumerate(wrap(text, 14, w - 32, 4)):
                p.text(line, x + 16, y + 46 + 22 * i, 14, ERROR)
            return
        # Texte de wttr.in : « Lieu : Ciel, +29°C (ressenti +33°C), vent ↑11km/h, humidité 70% »
        place, _, rest = text.partition(" : ")
        temp = re.search(r"([+-]?\d+)\s*°C", rest)
        if temp:
            rest = rest.replace(temp.group(0), "", 1)
        parts = [s.strip(" ()") for s in re.split(r",|\(", rest) if s.strip(" ()")]
        left = x + 16
        if temp:
            p.text(temp.group(1).lstrip("+") + "°", left, y + 40, 48, WHITE, bold=True)
            left += text_width(temp.group(1).lstrip("+") + "°", 48, True) + 14
        p.text(fit(place.split(",")[0].upper(), 15, x + w - 16 - left, True), left, y + 50, 15, CYAN, bold=True)
        if parts:
            p.text(fit(parts[0], 14, x + w - 16 - left), left, y + 72, 14, WHITE)
        for i, line in enumerate(wrap(" · ".join(parts[1:]), 13, w - 32, 2)):
            p.text(line, x + 16, y + 106 + 18 * i, 13, TEXT_DIM)

    def draw_system(self, p, x, y, w, h):
        d = self.data
        rows = [("PROCESSEUR", d["cpu"], d["cpu"] > 85), ("MÉMOIRE", d["ram"], d["ram"] > 85),
                ("DISQUE C", d["disk"], d["disk"] > 90)]
        if d["battery"]:
            level, plugged = d["battery"]
            rows.append(("EN CHARGE" if plugged else "BATTERIE", level, level < 20 and not plugged))
        for i, (label, value, alert) in enumerate(rows):
            ry = y + 48 + 32 * i
            color = ERROR if alert else CYAN
            p.text(label, x + 16, ry, 13, TEXT_DIM)
            bx, bw = x + 112, w - 180
            p.fill_rect(bx, ry + 3, bw, 10, PANEL)
            p.fill_rect(bx, ry + 3, bw * value / 100, 10, color)
            p.rect(bx, ry + 3, bw, 10, CYAN_DIM, 1, glow=False)
            p.text(f"{value:.0f} %", x + w - 16, ry, 14, WHITE, bold=True, anchor="rt")

    def draw_reminders(self, p, x, y, w, h):
        items = self.data["reminders"]
        if not items:
            p.text("Aucun rappel prévu", x + 16, y + 50, 14, TEXT_DIM)
            return
        today = datetime.now().date()
        for i, item in enumerate(items[:4]):
            due = datetime.fromisoformat(item["due"])
            when = f"{due:%H:%M}" if due.date() == today else f"{due:%d/%m %H:%M}"
            ry = y + 48 + 28 * i
            p.text(when, x + 16, ry, 14, CYAN, bold=True)
            left = x + 26 + text_width(when, 14, True)
            p.text(fit(item["message"], 14, x + w - 16 - left), left, ry, 14, WHITE)

    def draw_reply(self, p, x, y, w, h):
        if not self.get_reply:
            text = "Ouvre cette interface depuis la fenêtre de Jarvis (bouton INTERFACE HOLO) pour lui parler."
        else:
            text = self.get_reply() or "Pince le réacteur pour me parler."
        for i, line in enumerate(wrap(text, 15, w - 32, 4)):
            p.text(line, x + 16, y + 44 + 22 * i, 15, WHITE)

    def draw_volume(self, p, x, y, w, h):
        label = "MUET" if self.data["muted"] else f"{self.data['volume']} %"
        p.text(label, x + w - 14, y + 8, 15, ORANGE if self.data["muted"] else WHITE, bold=True, anchor="rt")

    # --- Actions ---

    def toast(self, text):
        """Affiche un message en bas de l'écran pendant quelques secondes."""
        self.toast_text, self.toast_time = str(text), time.monotonic()

    def act(self, fn, *args):
        """Lance un outil de Jarvis dans un thread (ouvrir une appli peut prendre du temps :
        l'image ne doit pas se figer), puis affiche ce qu'il répond."""
        def run():
            try:
                self.toast(fn(*args))
            except Exception as e:
                self.toast(f"Erreur : {e}")
        threading.Thread(target=run, daemon=True).start()

    def set_volume(self, level):
        """Appelée en continu pendant qu'on glisse sur la barre de volume."""
        self.volume_touched = time.monotonic()
        if level == self.data["volume"] and not self.data["muted"]:
            return
        self.data["volume"], self.data["muted"] = level, False
        if self.volume_control is None:
            self.volume_control = speakers() or False  # False : indisponible, inutile de réessayer
        if self.volume_control:
            self.volume_control.SetMasterVolumeLevelScalar(level / 100, None)
            self.volume_control.SetMute(0, None)

    def talk(self):
        if self.on_talk:
            self.on_talk()
        else:
            self.toast("Pour parler à Jarvis, ouvre cette interface depuis sa fenêtre.")

    def close(self):
        self.running = False

    # --- Données (thread séparé : lire la météo prend plusieurs secondes) ---

    def data_loop(self):
        control = speakers()  # pycaw passe par COM : une commande par thread
        next_weather = 0.0
        psutil.cpu_percent(None)  # la 1re mesure sert de point de départ
        while self.running:
            d = self.data
            d["cpu"] = psutil.cpu_percent(None)  # utilisation depuis la mesure précédente
            d["ram"] = psutil.virtual_memory().percent
            d["disk"] = psutil.disk_usage("C:\\").percent
            battery = psutil.sensors_battery()  # None sur un PC fixe
            d["battery"] = (battery.percent, battery.power_plugged) if battery else None
            if control and time.monotonic() - self.volume_touched > 1.5:  # pas pendant un réglage
                try:
                    d["volume"] = round(control.GetMasterVolumeLevelScalar() * 100)
                    d["muted"] = bool(control.GetMute())
                except Exception:
                    pass
            d["reminders"] = reminders.pending()[:4]
            if time.monotonic() >= next_weather:
                next_weather = time.monotonic() + WEATHER_REFRESH
                threading.Thread(target=lambda: d.update(weather=meteo(HOME_CITY)), daemon=True).start()
            time.sleep(DATA_REFRESH)

    # --- Main et souris ---

    def set_crop(self, frame):
        """L'image webcam (souvent 4:3) est agrandie puis rognée en haut et en bas pour remplir
        le HUD (16:9). On retient le calcul pour placer la main au bon endroit."""
        fh, fw = frame.shape[:2]
        if self.crop is None or self.crop[3:] != (fw, fh):
            scale = max(W / fw, H / fh)
            self.crop = (scale, (fw * scale - W) / 2, (fh * scale - H) / 2, fw, fh)

    def to_canvas(self, nx, ny):
        """Point MediaPipe (fractions de l'image webcam) -> pixels du HUD."""
        scale, ox, oy, fw, fh = self.crop
        return nx * fw * scale - ox, ny * fh * scale - oy

    def hand_pointer(self, hand):
        """Position du curseur (entre pouce et index, amplifiée) et écart pouce-index / paume."""
        points, aspect = hand
        dist = lambda a, b: math.hypot((a.x - b.x) * aspect, a.y - b.y)  # vraies proportions
        ratio = dist(points[4], points[8]) / max(dist(points[0], points[9]), 1e-6)
        # Le milieu pouce-index bouge peu quand on pince : le clic ne décale pas le curseur
        x, y = self.to_canvas((points[4].x + points[8].x) / 2, (points[4].y + points[8].y) / 2)
        x = min(max(W / 2 + (x - W / 2) * CURSOR_GAIN, 0), W - 1)
        y = min(max(H / 2 + (y - H / 2) * CURSOR_GAIN, 0), H - 1)
        return (x, y), ratio

    def on_mouse(self, event, x, y, flags, param):
        down = bool(flags & cv2.EVENT_FLAG_LBUTTON) or event == cv2.EVENT_LBUTTONDOWN
        if event == cv2.EVENT_LBUTTONUP:
            down = False
        self.mouse = (x, y, down)

    def track(self, hand, now):
        """Met à jour le curseur à partir de la main (sinon de la souris)."""
        if hand is not None:
            (x, y), self.ratio = self.hand_pointer(hand)
            x, y = self.filter((x, y), now)
            pinched = self.ratio < (PINCH_OFF if self.pinched else PINCH_ON)  # hystérésis
            self.pointer(x, y, pinched, now)
        elif self.mouse:
            x, y, down = self.mouse
            self.ratio = 0.0 if down else 1.0
            self.pointer(x, y, down, now)
        else:  # main perdue de vue : on lâche tout
            self.filter.reset()
            self.cursor = self.hover_owner = self.hover_widget = None
            self.release()

    def hit(self, x, y):
        """(élément, bouton) sous le curseur : d'abord les panneaux du dessus."""
        for owner in list(reversed(self.panels)) + [self.reactor]:
            if owner.contains(x, y):
                return owner, owner.widget_at(x, y)
        return None, None

    def pointer(self, x, y, pinched, now):
        """Cœur de l'interaction : position du curseur + pince fermée ou non."""
        self.cursor = (x, y)
        if not self.drag:  # pendant un déplacement, on garde l'élément attrapé
            self.hover_owner, self.hover_widget = self.hit(x, y)
        if pinched and not self.pinched:  # la pince vient de se fermer : « clic »
            self.press(x, y, now)
        elif pinched and self.drag:        # pince maintenue : on déplace / on glisse
            self.drag(x, y)
        elif not pinched:
            self.release()
        self.pinched = pinched

    def press(self, x, y, now):
        owner, widget = self.hover_owner, self.hover_widget
        self.ripples.append((x, y, now))
        if isinstance(widget, Slider):
            self.drag = lambda x, y: widget.set(widget.value_at(x - owner.x))
            self.drag(x, y)
        elif widget is not None:
            widget.flash_until = now + 0.25
            if widget.action:
                widget.action()
        elif owner is not None and owner.on_press:
            owner.on_press()
        elif owner is not None and owner.movable:
            dx, dy = x - owner.x, y - owner.y
            self.panels.remove(owner)
            self.panels.append(owner)  # passe au premier plan
            self.dragging = owner
            self.drag = lambda x, y: owner.move_to(x - dx, y - dy)

    def release(self):
        self.drag = self.dragging = None
        self.pinched = False

    # --- Dessin ---

    def make_backdrop(self):
        """Quadrillage bleu nuit et bords assombris, calculés une seule fois."""
        grid = np.zeros((H, W, 3), np.uint8)
        grid[:] = (18, 10, 4)
        grid[::40, :] = grid[:, ::40] = (44, 30, 10)
        yy, xx = np.mgrid[0:H, 0:W]
        d = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        vignette = np.clip(255 * (1 - 0.45 * d), 0, 255).astype(np.uint8)
        self.grid, self.vignette = grid, cv2.merge([vignette] * 3)

    def background(self, frame):
        scale, ox, oy, fw, fh = self.crop
        # On rogne d'abord la petite image, puis on l'agrandit une seule fois (plus rapide)
        x0, y0 = round(ox / scale), round(oy / scale)
        img = cv2.resize(frame[y0:fh - y0, x0:fw - x0], (W, H))
        img = cv2.addWeighted(img, VIDEO_BRIGHTNESS, self.grid, 1.0, 0)
        return cv2.multiply(img, self.vignette, scale=1 / 255)

    def draw_hand(self, p, hand):
        pts = [self.to_canvas(pt.x, pt.y) for pt in hand[0]]
        for bone in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            p.line(pts[bone.start], pts[bone.end], CYAN_DIM, 2, glow=False)
        for i, pt in enumerate(pts):
            p.circle(pt, 5 if i in (4, 8) else 3, CYAN if i in (4, 8) else WHITE, -1, glow=i in (4, 8))

    def draw_cursor(self, p, now):
        self.ripples = [r for r in self.ripples if now - r[2] < 0.4]
        for x, y, t0 in self.ripples:  # onde qui s'élargit à chaque pincement
            age = (now - t0) / 0.4
            p.circle((x, y), 12 + 50 * age, ORANGE, max(1, int(4 * (1 - age))), top=True)
        if self.cursor is None:
            return
        x, y = self.cursor
        color = ORANGE if self.pinched else WHITE if self.hover_owner else CYAN
        # Le cercle rétrécit à mesure que pouce et index se rapprochent : on voit venir le clic
        openness = min(max((self.ratio - PINCH_ON) / (0.9 - PINCH_ON), 0), 1)
        r = 8 + 18 * openness
        p.circle((x, y), r, color, 2, top=True)
        p.circle((x, y), 3, color, -1, top=True)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:  # petits repères en croix
            p.line((x + dx * (r + 4), y + dy * (r + 4)), (x + dx * (r + 10), y + dy * (r + 10)), color, 2, top=True)
        if self.pinched:
            p.circle((x, y), r + 5, color, 1, top=True)

    def render(self, frame, hand, now):
        p = Painter()
        self.reactor.draw(p, self, now)
        status = clean(self.get_status() if self.get_status else "")
        p.text(status.upper() or "PINCE LE RÉACTEUR POUR PARLER", W // 2, 412, 16, CYAN, bold=True, anchor="ct")
        if status:
            p.text("PINCE LE RÉACTEUR POUR PARLER", W // 2, 438, 12, TEXT_DIM, anchor="ct")
        for panel in self.panels:
            panel.draw(p, self, now)
        if hand is not None:
            self.draw_hand(p, hand)
        if now - self.toast_time < TOAST_SECONDS:
            p.text(fit(self.toast_text, 17, W - 200), W // 2, 652, 17, WHITE, bold=True, anchor="cc")
        elif hand is None and not self.mouse:
            p.text("MONTRE TA MAIN À LA CAMÉRA", W // 2, 652, 17, ORANGE, bold=True, anchor="cc")
        p.text("PINCER = CLIQUER   ·   PINCER UN PANNEAU = LE DÉPLACER   ·   ÉCHAP = QUITTER",
               W // 2, 700, 13, TEXT_DIM, anchor="cc")
        self.draw_cursor(p, now)
        return p.compose(self.background(frame))

    def step(self, frame, hand, now):
        """Une image : suivre la main, réagir, dessiner. Retourne l'image à afficher."""
        self.set_crop(frame)
        self.track(hand, now)
        return self.render(frame, hand, now)

    # --- Boucle principale ---

    def start(self, wait_for=None):
        """Lance le HUD dans son propre thread. wait_for : fonction qui renvoie True quand la
        webcam est libre (la surveillance des gestes doit d'abord l'avoir relâchée)."""
        self.running = True

        def go():
            deadline = time.monotonic() + 3
            while wait_for and not wait_for() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.run()
        threading.Thread(target=go, daemon=True).start()

    def capture_loop(self, latest, ready):
        """Thread de capture : lit la webcam et repère la main (~50 ms par image) pendant que
        le thread principal dessine l'image précédente (~25 ms). Les deux travaillent en même
        temps : ~20 images par seconde au lieu de ~13 s'ils attendaient l'un après l'autre."""
        tracker = None
        try:
            tracker = HandTracker()
            while self.running:
                frame, hand = tracker.read()
                latest["image"] = (frame, hand, time.monotonic())
                ready.set()
        except Exception as e:
            latest["error"] = e
            ready.set()
        finally:
            if tracker:
                tracker.close()  # éteint la webcam

    def run(self):
        self.running = True
        latest, ready = {}, threading.Event()
        capture = threading.Thread(target=self.capture_loop, args=(latest, ready), daemon=True)
        try:
            self.make_backdrop()
            capture.start()
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            if self.fullscreen:
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.resizeWindow(WINDOW, W, H)
            cv2.setMouseCallback(WINDOW, self.on_mouse)
            threading.Thread(target=self.data_loop, daemon=True).start()
            last = time.monotonic()
            while self.running:
                if not ready.wait(5):
                    raise RuntimeError("la webcam ne répond pas")
                ready.clear()
                if "error" in latest:
                    raise latest["error"]
                frame, hand, now = latest["image"]  # la plus récente (on saute les images en retard)
                self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)  # images par seconde, lissé
                last = now
                cv2.imshow(WINDOW, self.step(frame, hand, now))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except Exception as e:
            if not self.on_error:
                raise
            self.on_error(f"Interface holographique indisponible : {e}")
        finally:
            self.running = False
            if capture.is_alive():
                capture.join(timeout=3)  # attend que la webcam soit éteinte
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass
            if self.on_close:
                self.on_close()


if __name__ == "__main__":
    HoloHUD(fullscreen="--fenetre" not in sys.argv).run()
