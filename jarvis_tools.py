"""The tools Jarvis can decide to use on his own.

How it works:
1. Every function decorated with @tool is sent to the model along with a description.
   Ollama builds that description from the name, the parameter types and the docstring.
2. When you say "ouvre Spotify", the model doesn't answer with text; it answers
   with "call ouvrir_application(nom="Spotify")".
3. jarvis_core.ask_model runs the function and sends the result back to the model,
   which then writes its final answer.

A word about language: the tool names, parameters and docstrings are in French on purpose.
They aren't just documentation, they're part of the prompt. Jarvis talks with you in French,
and these descriptions were tuned by trial and error against a benchmark (23/25). Rewriting
them changes how the model behaves, so re-test if you touch them.

To add a tool:
- write a function with typed parameters (str, int...) that returns some text;
- give it a clear docstring with an Args section describing each parameter. That's what
  the model reads to decide WHEN to use it, so take care with the first sentence, and say
  when NOT to use it if another tool is close;
- decorate it with @tool("text shown in the status bar while it runs").
"""
import ast
import csv
import ctypes
import difflib
import functools
import json
import math
import operator
import os
import re
import shutil
import subprocess
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote, quote_plus
from urllib.request import Request, urlopen
import psutil
from ddgs import DDGS
from jarvis_config import greeting
from jarvis_core import add_fact, now_text
from jarvis_reminders import reminders

# The only folders Jarvis is allowed to read or change: (accepted names, path, display name)
FOLDERS = [
    ({"documents", "document", "mes documents"}, Path.home() / "Documents", "Documents"),
    ({"telechargements", "telechargement", "downloads"}, Path.home() / "Downloads", "Téléchargements"),
    ({"bureau", "desktop"}, Path.home() / "Desktop", "Bureau"),
]
MAX_FILE_READ = 4000  # max characters read from a file (the rest is cut off)
TOOLS = []  # filled in by @tool, then handed to ask_model by the front ends


def tool(label):
    """Decorator: registers the function as a tool and attaches a status label to it."""
    def register(func):
        func.label = label
        TOOLS.append(func)
        return func
    return register


# --- Knowledge ---

@tool("🌐 Recherche web...")
def recherche_web(requete: str) -> str:
    """Cherche une information récente ou factuelle sur Internet.

    Args:
        requete: Les mots-clés à rechercher
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(requete, max_results=3)
            return "".join(f"- {r['title']}: {r['body']}\n" for r in results) or "Aucun résultat."
    except Exception as e:
        return f"Recherche impossible : {e}"


@tool("🧠 Mémorisation...")
def memoriser(fait: str) -> str:
    """Enregistre durablement un fait sur l'utilisateur (préférence, info perso, projet...).
    À utiliser quand l'utilisateur demande de retenir ou de se souvenir de quelque chose.

    Args:
        fait: Le fait à retenir, formulé à la troisième personne
    """
    add_fact(fait)
    return f"Fait mémorisé : {fait}"


FORECAST_DAYS = {"demain": 1, "apres-demain": 2, "après-demain": 2}


@tool("🌦️ Météo...")
def meteo(ville: str, jour: str = "aujourd'hui") -> str:
    """Donne la météo d'une ville : le temps actuel, ou la prévision de demain / après-demain.

    Args:
        ville: Le nom de la ville (vide = la ville de l'utilisateur, trouvée automatiquement)
        jour: « aujourd'hui » (temps actuel), « demain » ou « après-demain »
    """
    # wttr.in is free and needs no API key (no city = located from your Internet connection)
    day = FORECAST_DAYS.get(jour.lower().strip().replace(" ", "-"))
    try:
        if day is None:
            # Current weather. The %x bits are wttr.in format codes:
            # %l place, %C sky, %t temperature, %f feels like, %w wind, %h humidity
            fmt = quote("%l : %C, %t (ressenti %f), vent %w, humidité %h", safe="%")
            with urlopen(Request(f"https://wttr.in/{quote(ville)}?format={fmt}&lang=fr",
                                 headers={"User-Agent": "curl"}), timeout=8) as resp:
                return resp.read().decode("utf-8").strip()
        # Forecast: detailed data (JSON), 8 readings per day (every 3 hours)
        with urlopen(Request(f"https://wttr.in/{quote(ville)}?format=j1&lang=fr",
                             headers={"User-Agent": "curl"}), timeout=8) as resp:
            data = json.loads(resp.read())
        forecast = data["weather"][day]
        place = ville or data["nearest_area"][0]["areaName"][0]["value"]
        sky = lambda i: forecast["hourly"][i]["lang_fr"][0]["value"].strip().lower()
        rain = max(int(h["chanceofrain"]) for h in forecast["hourly"][2:7])  # from 6 am to 6 pm
        return (f"{jour.capitalize()} à {place} : de {forecast['mintempC']} à {forecast['maxtempC']} °C, "
                f"matin {sky(3)}, après-midi {sky(5)}, risque de pluie {rain} %.")
    except Exception as e:
        return f"Météo indisponible : {e}"


HOME_CITY = ""  # city for the briefing; empty = worked out from your Internet connection


def build_briefing(hello="Bonjour"):
    """The daily briefing in a few sentences: date, time, local weather, today's reminders.
    Built without the model: it's fast, and nothing can be made up."""
    now = datetime.now()
    parts = [f"{greeting(hello)} Nous sommes {now_text().replace(',', ', il est')}."]
    weather = meteo(HOME_CITY)
    if not weather.startswith("Météo indisponible"):
        parts.append(f"Météo à {weather}.")
    today = [i for i in reminders.pending() if datetime.fromisoformat(i["due"]).date() == now.date()]
    if today:
        items = ", ".join(f"à {datetime.fromisoformat(i['due']):%H:%M}, {i['message']}" for i in today)
        parts.append(f"{'Un rappel' if len(today) == 1 else f'{len(today)} rappels'} aujourd'hui : {items}.")
    else:
        parts.append("Aucun rappel aujourd'hui.")
    return " ".join(parts)


@tool("📋 Briefing...")
def briefing() -> str:
    """Fait le point du jour pour l'utilisateur : date, heure, météo locale et rappels du jour.
    À utiliser pour « fais-moi le point », « le briefing », « quoi de prévu aujourd'hui »."""
    return build_briefing()


@tool("💻 État du PC...")
def etat_pc() -> str:
    """Donne l'état du PC : batterie, utilisation du processeur, mémoire vive, espace disque.
    À utiliser pour « comment va le PC ? », « il me reste combien de batterie ? », « le disque est plein ? »."""
    cpu = psutil.cpu_percent(interval=0.5)  # measured over half a second
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("C:\\")
    parts = [f"processeur utilisé à {cpu:.0f} %",
             f"mémoire vive utilisée à {ram.percent:.0f} % ({ram.available / 1e9:.1f} Go libres)",
             f"disque C : {disk.free / 1e9:.0f} Go libres sur {disk.total / 1e9:.0f}"]
    battery = psutil.sensors_battery()  # None on a desktop PC
    if battery:
        if battery.power_plugged:
            status = "en charge"
        elif battery.secsleft > 0:  # negative when Windows can't estimate it
            status = f"environ {duree_texte(battery.secsleft)} d'autonomie"
        else:
            status = "sur batterie"
        parts.insert(0, f"batterie à {battery.percent:.0f} % ({status})")
    return "État du PC : " + " ; ".join(parts) + "."


# Safe math: we parse the expression instead of running it (eval would run any code at all)
OPERATIONS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
              ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}
FUNCTIONS = {"sqrt": math.sqrt, "racine": math.sqrt, "abs": abs, "round": round, "arrondi": round,
             "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log10, "ln": math.log}
CONSTANTS = {"pi": math.pi, "e": math.e}


def _evaluate(node):
    """Compute one node of the expression; refuse anything that isn't plain math."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in CONSTANTS:
        return CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATIONS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("puissance trop grande")
        return OPERATIONS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATIONS:
        return OPERATIONS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
        return FUNCTIONS[node.func.id](*[_evaluate(a) for a in node.args])
    raise ValueError("expression non autorisée")


@tool("🧮 Calcul...")
def calculer(expression: str) -> str:
    """Calcule exactement une expression mathématique. À utiliser pour TOUT calcul
    (le calcul de tête est peu fiable). Pour « 15 % de 80 », écrire 0.15 * 80.

    Args:
        expression: L'expression, par exemple « (12.5 + 3) * 4 », « 2 ** 10 » ou « sqrt(144) »
    """
    text = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)  # French decimal comma: 3,5 -> 3.5
    try:
        result = _evaluate(ast.parse(text, mode="eval").body)
    except ZeroDivisionError:
        return "Division par zéro impossible."
    except Exception as e:
        return f"Calcul impossible ({e}) : {expression}"
    if isinstance(result, float):
        result = int(result) if result.is_integer() else round(result, 6)
    return f"{expression} = {result}"


@tool("🕒 Heure...")
def heure_et_date() -> str:
    """Donne la date et l'heure actuelles."""
    return now_text()


# --- Timers and reminders (see jarvis_reminders.py) ---

def parse_heure(texte):
    """"18:30", "18h30", "18h", "8 h 05" -> (hours, minutes), or None."""
    # hours, then optionally "h" or ":" followed by the minutes
    match = re.fullmatch(r"\s*(\d{1,2})\s*(?:(?:h|:)\s*(\d{2})?)?\s*", texte.lower())
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2)) if match.group(2) else 0
    return (hours, minutes) if hours < 24 and minutes < 60 else None


def duree_texte(seconds):
    """3725 -> "1 h 2 min"."""
    minutes = max(1, round(seconds / 60))
    return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"


@tool("⏰ Création du rappel...")
def creer_rappel(message: str, minutes: float = 0, heure: str = "") -> str:
    """Crée un minuteur ou un rappel : Jarvis préviendra l'utilisateur à voix haute au bon moment.
    Pour « dans 20 minutes » ou « un minuteur de 5 minutes », utiliser minutes.
    Pour « à 18h30 », utiliser heure.

    Args:
        message: Ce qu'il faudra rappeler, par exemple « sortir le linge » ou « minuteur terminé »
        minutes: Dans combien de minutes prévenir (0 si une heure précise est donnée)
        heure: Heure précise au format HH:MM, par exemple « 18:30 » (vide si minutes est donné)
    """
    now = datetime.now()
    heure = str(heure).strip() if heure not in (None, "", 0) else ""  # the model sometimes sends a number
    if heure:
        parsed = parse_heure(heure)
        if not parsed:
            return f"Heure « {heure} » incompréhensible : utiliser le format HH:MM."
        # The model sometimes splits "18h30" into heure=18 and minutes=30, so we put it back together
        if parsed[1] == 0 and heure.isdigit() and minutes and 0 < float(minutes) < 60:
            parsed = (parsed[0], int(float(minutes)))
        due = now.replace(hour=parsed[0], minute=parsed[1], second=0, microsecond=0)
        if due <= now:
            due += timedelta(days=1)  # that time has already passed today, so it's for tomorrow
    elif minutes and float(minutes) > 0:
        due = now + timedelta(minutes=float(minutes))
    else:
        return "Il faut préciser dans combien de minutes, ou à quelle heure."
    reminders.add(message, due)
    day = "" if due.date() == now.date() else " demain"
    return f"Rappel « {message} » prévu{day} à {due:%H:%M} (dans {duree_texte((due - now).total_seconds())})."


@tool("⏰ Lecture des rappels...")
def lister_rappels() -> str:
    """Liste les minuteurs et rappels prévus."""
    items = reminders.pending()
    if not items:
        return "Aucun rappel prévu."
    now = datetime.now()
    return "\n".join(f"- {datetime.fromisoformat(i['due']):%H:%M} (dans "
                     f"{duree_texte((datetime.fromisoformat(i['due']) - now).total_seconds())}) : {i['message']}"
                     for i in items)


@tool("⏰ Annulation du rappel...")
def annuler_rappel(recherche: str) -> str:
    """Annule un minuteur ou un rappel prévu.

    Args:
        recherche: Un mot du rappel à annuler (ex : « linge »), ou « tous » pour tout annuler
    """
    removed = reminders.cancel(recherche)
    if not removed:
        return f"Aucun rappel ne correspond à « {recherche} »."
    return "Annulé : " + ", ".join(f"« {i['message']} »" for i in removed)


# --- Files (limited to three folders: Documents, Downloads, Desktop) ---

def folder(dossier):
    """Folder name as the user said it -> (path, display name), or (None, error message)."""
    key = dossier.lower().strip().replace("é", "e").replace("è", "e")
    for names, path, label in FOLDERS:
        if key in names:
            return path, label
    return None, f"Dossier « {dossier} » non autorisé : seulement Documents, Téléchargements ou Bureau."


def safe_file(nom, dossier):
    """Path of a file INSIDE the allowed folder (Path(nom).name stops anyone from escaping
    it with "../"), or (None, error message)."""
    path, label = folder(dossier)
    return (path / Path(nom).name, label) if path else (None, label)


@tool("📁 Lecture du dossier...")
def lister_fichiers(dossier: str = "documents") -> str:
    """Liste les fichiers d'un dossier de l'utilisateur, les plus récents en premier.

    Args:
        dossier: « documents », « téléchargements » ou « bureau »
    """
    path, label = folder(dossier)
    if not path:
        return label
    files = sorted((f for f in path.iterdir() if not f.name.startswith(".") and f.name != "desktop.ini"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return f"Le dossier {label} est vide."
    return f"{label} (du plus récent au plus ancien) :\n" + "\n".join(
        f"- {f.name}{'/' if f.is_dir() else ''}" for f in files[:25])


@tool("📄 Lecture du fichier...")
def lire_fichier(nom: str, dossier: str = "documents") -> str:
    """Lit le contenu d'un fichier texte.

    Args:
        nom: Le nom du fichier, avec son extension
        dossier: « documents », « téléchargements » ou « bureau »
    """
    path, label = safe_file(nom, dossier)
    if not path:
        return label
    if not path.is_file():
        return f"Fichier {path.name} introuvable dans {label}."
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return f"Impossible de lire {path.name} (fichier non texte ?)."
    if len(text) > MAX_FILE_READ:
        return text[:MAX_FILE_READ] + f"\n[... fichier tronqué : {len(text)} caractères au total]"
    return text


@tool("📝 Création du fichier...")
def creer_fichier(nom: str, contenu: str = "", dossier: str = "documents") -> str:
    """Crée un NOUVEAU fichier texte (n'écrase jamais un fichier existant).

    Args:
        nom: Le nom du fichier, avec son extension
        contenu: Le texte à écrire dans le fichier (optionnel)
        dossier: « documents » (par défaut), « téléchargements » ou « bureau »
    """
    path, label = safe_file(nom, dossier)
    if not path:
        return label
    if path.exists():
        return f"{path.name} existe déjà dans {label}, je ne l'écrase pas (utiliser ajouter_au_fichier)."
    path.write_text(contenu, encoding="utf-8")
    return f"Fichier {path.name} créé dans {label}."


@tool("✏️ Ajout au fichier...")
def ajouter_au_fichier(nom: str, texte: str, dossier: str = "documents") -> str:
    """Ajoute du texte à la FIN d'un fichier texte existant (liste de courses, notes...).
    Le contenu existant n'est jamais effacé.

    Args:
        nom: Le nom du fichier, avec son extension
        texte: Le texte à ajouter
        dossier: « documents », « téléchargements » ou « bureau »
    """
    path, label = safe_file(nom, dossier)
    if not path:
        return label
    if not path.is_file():
        return f"Fichier {path.name} introuvable dans {label} (utiliser creer_fichier)."
    # If the file doesn't end with a newline, add one before the new text
    ends_with_newline = path.stat().st_size == 0 or path.read_bytes()[-1:] == b"\n"
    with open(path, "a", encoding="utf-8") as f:  # "a" = append at the end, never erase
        f.write(("" if ends_with_newline else "\n") + texte + "\n")
    return f"Texte ajouté à la fin de {path.name}."


# --- Controlling the PC ---

# Common apps: name as the user says it -> Windows command ("start <command>").
# Entries ending with ":" are app links (they work even for Microsoft Store apps).
# Anything not listed here gets looked up in the Start menu.
APPS = {
    "bloc-notes": "notepad", "notepad": "notepad",
    "calculatrice": "calc",
    "explorateur": "explorer", "fichiers": "explorer",
    "chrome": "chrome", "edge": "msedge", "firefox": "firefox",
    "vs code": "code", "vscode": "code", "visual studio code": "code",
    "word": "winword", "excel": "excel", "powerpoint": "powerpnt", "outlook": "outlook",
    "spotify": "spotify:", "discord": "discord:",
    "paramètres": "ms-settings:", "parametres": "ms-settings:",
    "gestionnaire des tâches": "taskmgr", "terminal": "wt", "invite de commandes": "cmd",
    "paint": "mspaint",
}


# Where Windows keeps the shortcuts of installed software
SHORTCUT_DIRS = [
    Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "Microsoft/Windows/Start Menu/Programs",
    Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
    Path.home() / "Desktop",
    Path.home() / "OneDrive/Bureau",
    Path.home() / "OneDrive/Desktop",
    Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "Desktop",
]
IGNORED_WORDS = ("uninstall", "désinstaller", "desinstaller")  # shortcuts we never launch

# Nicknames -> shortcut name (for games that start through a launcher with another name)
SHORTCUT_ALIASES = {
    "league of legends": "client riot", "lol": "client riot", "valorant": "client riot",
    "obs": "obs studio",
}


@functools.lru_cache(maxsize=1)  # only scanned once per Jarvis session
def installed_apps():
    """Every shortcut found: {lowercase name: path of the .lnk}."""
    apps = {}
    for folder in SHORTCUT_DIRS:
        if not folder.is_dir():
            continue
        for lnk in folder.rglob("*.lnk"):
            name = lnk.stem.lower()
            if name and not any(w in name for w in IGNORED_WORDS):
                apps.setdefault(name, lnk)
    return apps


def find_installed_app(name):
    """Find the shortcut closest to the requested name, or None."""
    apps = installed_apps()
    name = SHORTCUT_ALIASES.get(name, name)
    # 1. Exact name
    if name in apps:
        return apps[name]
    # 2. Name contained: "photoshop" finds "Adobe Photoshop 2024" (shortest wins)
    contains = sorted((n for n in apps if name in n), key=len)
    if contains:
        return apps[contains[0]]
    # 3. Typo: "discrod" finds "discord"
    close = difflib.get_close_matches(name, apps, n=1, cutoff=0.75)
    return apps[close[0]] if close else None


@tool("🚀 Ouverture de l'application...")
def ouvrir_application(nom: str) -> str:
    """Ouvre n'importe quelle application ou jeu installé sur le PC (Chrome, Spotify, Discord, League of Legends, Word, calculatrice...).

    Args:
        nom: Le nom de l'application
    """
    key = nom.lower().strip()
    # 1. A common app we know about (the APPS list)
    target = APPS.get(key)
    if not target:
        # 2. A Start menu or desktop shortcut
        shortcut = find_installed_app(key)
        if shortcut:
            os.startfile(shortcut)
            return f"{shortcut.stem} lancé."
        # 3. A known name inside the request ("google chrome" -> chrome)
        target = next((cmd for alias, cmd in APPS.items() if alias in key), None)
    # 4. A program on the PATH (plain names only, so no command injection)
    if not target and re.fullmatch(r"[\w.-]+", key) and shutil.which(key):
        target = key
    if not target:
        suggestions = difflib.get_close_matches(key, installed_apps(), n=3, cutoff=0.4)
        hint = f" Tu voulais peut-être : {', '.join(suggestions)} ?" if suggestions else ""
        return f"Aucune application « {nom} » trouvée sur le PC.{hint}"
    subprocess.Popen(["cmd", "/c", "start", "", target], creationflags=subprocess.CREATE_NO_WINDOW)
    return f"{nom} lancé."


# Programs we must NEVER close: Windows itself, Ollama (the brain), Python (that's Jarvis!)
PROTECTED = {"explorer", "python", "pythonw", "ollama", "ollama app", "svchost", "system", "csrss",
             "wininit", "winlogon", "lsass", "services", "dwm", "smss", "conhost", "sihost",
             "fontdrvhost", "taskhostw", "runtimebroker", "searchhost", "startmenuexperiencehost",
             "shellexperiencehost", "textinputhost", "ctfmon", "registry", "audiodg"}

# Name as the user says it -> name(s) of the running program (without ".exe")
PROCESS_NAMES = {
    "word": ["winword"], "powerpoint": ["powerpnt"], "edge": ["msedge"],
    "vs code": ["code"], "vscode": ["code"], "visual studio code": ["code"],
    "bloc-notes": ["notepad"], "calculatrice": ["calculatorapp", "calculator"], "paint": ["mspaint"],
    "gestionnaire des tâches": ["taskmgr"], "obs": ["obs64"], "obs studio": ["obs64"],
    "league of legends": ["league of legends", "leagueclientux", "leagueclient", "riotclientux"],
    "lol": ["league of legends", "leagueclientux", "leagueclient", "riotclientux"],
    "riot": ["riotclientux", "riotclientservices"],
}


# Programs that might hold unsaved work: never force-closed
NEVER_FORCE = {"winword", "excel", "powerpnt", "notepad", "code", "mspaint", "onenote", "outlook"}


def running_programs():
    """Names (lowercase, no .exe) of the programs currently running."""
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True,
                         errors="replace", creationflags=subprocess.CREATE_NO_WINDOW).stdout
    return {row[0].lower().removesuffix(".exe") for row in csv.reader(out.splitlines()) if row}


@tool("❌ Fermeture de l'application...")
def fermer_application(nom: str, forcer: bool = False) -> str:
    """Ferme une application ouverte sur le PC (Spotify, Chrome, Word, un jeu...).

    Args:
        nom: Le nom de l'application à fermer
        forcer: True seulement si l'utilisateur demande de forcer la fermeture (le travail non enregistré est perdu)
    """
    key = nom.lower().strip()
    compact = key.replace(" ", "")
    running = running_programs() - PROTECTED
    wanted = PROCESS_NAMES.get(key, [key, compact])
    # STRICT matching: a loose match closes the wrong program ("spotify" once hit
    # "spotifyxboxgamebarwebview", a piece of the Xbox Game Bar).
    targets = [p for p in running if p in wanted]
    if not targets:  # almost exact: "obs" -> "obs64" (at most 3 extra characters)
        targets = [p for p in running if p.startswith(compact) and len(p) - len(compact) <= 3]
    if not targets:  # typo: "discrod" -> "discord"
        targets = difflib.get_close_matches(compact, running, n=1, cutoff=0.85)
    if not targets:
        return f"Aucune application « {nom} » n'est ouverte."

    def kill(programs, force):
        # Without /F, Windows politely asks the app to close (like clicking its X), so it
        # gets a chance to offer saving. With /F it's killed on the spot, no questions asked.
        for program in programs:
            command = ["taskkill", "/IM", f"{program}.exe", "/T"] + (["/F"] if force else [])
            subprocess.run(command, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        time.sleep(2)  # give it time to close
        return [p for p in programs if p in running_programs()]

    # Always try a normal close first (the model sometimes asks to force it for no reason)
    still_open = kill(targets, force=False)
    if still_open and forcer:
        # Only force it when no work can be lost (never Word, Notepad...)
        still_open = kill([p for p in still_open if p not in NEVER_FORCE], force=True) + \
                     [p for p in still_open if p in NEVER_FORCE]
    if still_open:
        return (f"Fermeture demandée, mais {', '.join(still_open)} est encore ouvert : il attend peut-être "
                "une confirmation (enregistrer ?) ou reste dans la barre des tâches.")
    return f"{', '.join(targets)} fermé."


# --- Music ---

VK_MEDIA_NEXT, VK_MEDIA_PREV, VK_MEDIA_PLAY_PAUSE = 0xB0, 0xB1, 0xB3  # media keys


@tool("🎵 Contrôle de la musique...")
def controle_musique(action: str) -> str:
    """Contrôle la musique ou la vidéo DÉJÀ en cours (Spotify, YouTube...) : mettre en pause,
    reprendre, passer à la chanson suivante (« suivant », « passe », « prochaine ») ou revenir
    à la précédente. Pour lancer une nouvelle chanson précise, utiliser jouer_musique.

    Args:
        action: « pause », « lecture », « suivant » ou « precedent »
    """
    action = action.lower().strip()
    if action.startswith(("suiv", "next", "passe", "proch")):  # check "prochain" before the "pr" test
        _press(VK_MEDIA_NEXT)
        return "Morceau suivant."
    if action.startswith(("pr", "prev", "reviens")):  # précédent / precedent / previous
        _press(VK_MEDIA_PREV)
        return "Morceau précédent."
    _press(VK_MEDIA_PLAY_PAUSE)  # the same key pauses and resumes (it toggles)
    return "Lecture mise en pause ou reprise."


@tool("🎶 Lancement de la musique...")
def jouer_musique(recherche: str, plateforme: str = "youtube") -> str:
    """Lance une NOUVELLE chanson, un artiste ou une playlist précis demandé par l'utilisateur
    (« mets du Drake », « joue de la musique lofi »). Pas pour pause / suivant : voir controle_musique.

    Args:
        recherche: Ce qu'il faut jouer, par exemple « Drake God's Plan » ou « musique lofi »
        plateforme: Toujours « youtube » (lecture automatique), SAUF si l'utilisateur prononce le mot « Spotify »
    """
    if not recherche.strip():
        return ("Aucune chanson précisée. Pour mettre en pause ou passer à la suivante, "
                "utiliser l'outil controle_musique.")
    if "spotify" in plateforme.lower():
        # Without a Spotify developer account we can open the search, but not start playback
        os.startfile(f"spotify:search:{quote(recherche)}")
        return f"Recherche « {recherche} » ouverte dans Spotify : il reste à cliquer sur le titre."
    url = find_youtube_video(recherche)
    if url:
        webbrowser.open(url)  # a YouTube video starts playing by itself when opened
        return f"Lecture de « {recherche} » sur YouTube : {url}"
    # Last resort: the YouTube results page (you'll have to click a video)
    webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(recherche)}")
    return f"Résultats YouTube pour « {recherche} » ouverts : il reste à choisir la vidéo."


def find_youtube_video(recherche):
    """URL of the first YouTube video found, or None. The search service is flaky,
    so we try a video search first, then a regular web search filtered on YouTube."""
    attempts = [lambda d: d.videos(recherche, max_results=8),
                lambda d: d.text(f"{recherche} youtube", max_results=10)]
    for search in attempts:
        try:
            with DDGS() as ddgs:
                for result in search(ddgs):
                    link = result.get("content") or result.get("href") or ""
                    if "youtube.com/watch" in link:
                        return link
        except Exception:
            continue  # "no results" or service down: move on to the next attempt
    return None


@tool("🌍 Ouverture du navigateur...")
def ouvrir_site(adresse_ou_recherche: str) -> str:
    """Ouvre un site web dans le navigateur, ou lance une recherche Google.

    Args:
        adresse_ou_recherche: Une URL (ex: youtube.com) ou des mots à rechercher
    """
    target = adresse_ou_recherche.strip()
    # Does it look like an address ("youtube.com", "https://x.fr/page")? If not, search Google
    if re.fullmatch(r"(https?://)?[\w-]+(\.[\w-]+)+(/\S*)?", target):
        url = target if target.startswith("http") else f"https://{target}"
    else:
        url = f"https://www.google.com/search?q={quote_plus(target)}"
    webbrowser.open(url)
    return f"Ouvert : {url}"


# Fallback volume control: we simulate the keyboard's media keys (no dependencies).
# Windows key codes: mute, volume down, volume up
VK_VOLUME_MUTE, VK_VOLUME_DOWN, VK_VOLUME_UP = 0xAD, 0xAE, 0xAF


def _press(vk, times=1):
    """Press and release a key, `times` times."""
    for _ in range(times):
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP


def speakers():
    """Speaker volume control (pycaw), or None if it's not available.
    pycaw goes through COM (Windows), which has to be initialized in EVERY thread that uses it:
    tools run in the answer thread, not the window's."""
    try:
        import comtypes
        from pycaw.pycaw import AudioUtilities
        comtypes.CoInitialize()
        return AudioUtilities.GetSpeakers().EndpointVolume
    except Exception:
        return None


@tool("🔊 Réglage du volume...")
def regler_volume(niveau: int) -> str:
    """Règle le volume du PC à une valeur précise.

    Args:
        niveau: Le volume voulu, de 0 (muet) à 100
    """
    niveau = max(0, min(100, int(niveau)))
    control = speakers()
    if control:
        control.SetMasterVolumeLevelScalar(niveau / 100, None)  # direct and exact
        if niveau > 0:
            control.SetMute(0, None)  # setting the volume implies you want to hear something
    else:  # fallback: keyboard keys (each press = 2%, so go down to 0 and back up)
        _press(VK_VOLUME_DOWN, 50)
        _press(VK_VOLUME_UP, round(niveau / 2))
    return f"Volume réglé à {niveau} %."


@tool("🔊 Lecture du volume...")
def lire_volume() -> str:
    """Donne le volume actuel du PC (et s'il est coupé)."""
    control = speakers()
    if not control:
        return "Impossible de lire le volume sur ce PC."
    level = round(control.GetMasterVolumeLevelScalar() * 100)
    return f"Volume à {level} %{', son coupé' if control.GetMute() else ''}."


@tool("🔇 Coupure du son...")
def couper_son() -> str:
    """Coupe ou rétablit le son du PC (bascule)."""
    control = speakers()
    if not control:
        _press(VK_VOLUME_MUTE)
        return "Son basculé (coupé / rétabli)."
    muted = not control.GetMute()
    control.SetMute(int(muted), None)
    return "Son coupé." if muted else "Son rétabli."


@tool("🔒 Verrouillage...")
def verrouiller_pc() -> str:
    """Verrouille la session Windows."""
    ctypes.windll.user32.LockWorkStation()
    return "Session verrouillée."
