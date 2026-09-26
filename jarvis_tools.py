"""Outils que Jarvis peut appeler de lui-même.

Comment ça marche :
1. Chaque fonction décorée par @tool est envoyée au modèle avec sa description.
   Ollama construit cette description à partir du nom, des types et de la docstring.
2. Quand l'utilisateur demande « ouvre Spotify », le modèle répond non pas par du texte
   mais par « appelle ouvrir_application(nom="Spotify") ».
3. jarvis_core.ask_model exécute la fonction et renvoie son résultat au modèle,
   qui formule alors sa réponse finale.

Pour ajouter un outil :
- écrire une fonction avec des paramètres typés (str, int...) qui retourne un texte ;
- lui donner une docstring claire, avec une section Args décrivant chaque paramètre
  (c'est ce que lit le modèle pour savoir QUAND l'utiliser : soigner la 1re phrase) ;
- la décorer avec @tool("texte affiché dans la barre de statut pendant l'exécution").
"""
import csv
import ctypes
import difflib
import functools
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
from ddgs import DDGS
from jarvis_core import add_fact, now_text
from jarvis_reminders import reminders

DOCUMENTS_DIR = Path.home() / "Documents"  # seul dossier que Jarvis peut lire / modifier
TOOLS = []  # rempli automatiquement par @tool, puis passé à ask_model par les interfaces


def tool(label):
    """Décorateur : enregistre la fonction comme outil et lui associe un libellé."""
    def register(func):
        func.label = label
        TOOLS.append(func)
        return func
    return register


# --- Connaissances ---

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


@tool("🌦️ Météo...")
def meteo(ville: str) -> str:
    """Donne la météo actuelle d'une ville.

    Args:
        ville: Le nom de la ville
    """
    # Service gratuit wttr.in, sans clé d'API. Les %x sont ses codes de format :
    # %l lieu, %C ciel, %t température, %f ressenti, %w vent, %h humidité
    fmt = quote("%l : %C, %t (ressenti %f), vent %w, humidité %h", safe="%")
    url = f"https://wttr.in/{quote(ville)}?format={fmt}&lang=fr"
    try:
        with urlopen(Request(url, headers={"User-Agent": "curl"}), timeout=8) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception as e:
        return f"Météo indisponible : {e}"


@tool("🕒 Heure...")
def heure_et_date() -> str:
    """Donne la date et l'heure actuelles."""
    return now_text()


# --- Minuteurs et rappels (voir jarvis_reminders.py) ---

def parse_heure(texte):
    """« 18:30 », « 18h30 », « 18h », « 8 h 05 » -> (heures, minutes), ou None."""
    # heures, puis éventuellement « h » ou « : » suivi des minutes
    match = re.fullmatch(r"\s*(\d{1,2})\s*(?:(?:h|:)\s*(\d{2})?)?\s*", texte.lower())
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2)) if match.group(2) else 0
    return (hours, minutes) if hours < 24 and minutes < 60 else None


def duree_texte(seconds):
    """3725 -> « 1 h 2 min »."""
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
    if heure:
        parsed = parse_heure(heure)
        if not parsed:
            return f"Heure « {heure} » incompréhensible : utiliser le format HH:MM."
        due = now.replace(hour=parsed[0], minute=parsed[1], second=0, microsecond=0)
        if due <= now:
            due += timedelta(days=1)  # heure déjà passée aujourd'hui : c'est pour demain
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


# --- Fichiers (limités au dossier Documents) ---

@tool("📁 Lecture du dossier Documents...")
def lister_fichiers() -> str:
    """Liste les fichiers du dossier Documents de l'utilisateur."""
    files = sorted(DOCUMENTS_DIR.iterdir())
    return "\n".join(f.name for f in files[:30]) or "Le dossier Documents est vide."


@tool("📄 Lecture du fichier...")
def lire_fichier(nom: str) -> str:
    """Lit le contenu d'un fichier texte du dossier Documents.

    Args:
        nom: Le nom du fichier, avec son extension
    """
    path = DOCUMENTS_DIR / Path(nom).name  # empêche de sortir de Documents
    if not path.is_file():
        return f"Fichier {path.name} introuvable dans Documents."
    try:
        return path.read_text(encoding="utf-8")[:2000]
    except (OSError, UnicodeDecodeError):
        return f"Impossible de lire {path.name} (fichier non texte ?)."


@tool("📝 Création du fichier...")
def creer_fichier(nom: str, contenu: str = "") -> str:
    """Crée un nouveau fichier texte dans le dossier Documents.

    Args:
        nom: Le nom du fichier, avec son extension
        contenu: Le texte à écrire dans le fichier (optionnel)
    """
    path = DOCUMENTS_DIR / Path(nom).name
    if path.exists():
        return f"{path.name} existe déjà, je ne l'écrase pas."
    path.write_text(contenu, encoding="utf-8")
    return f"Fichier {path.name} créé dans Documents."


# --- Contrôle du PC ---

# Applications courantes : nom dit par l'utilisateur -> commande Windows (« start <commande> »).
# Les entrées finissant par « : » sont des liens d'application (ouverts même si l'appli
# vient du Microsoft Store). Tout ce qui n'est pas ici est cherché dans le menu Démarrer.
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


# Dossiers où Windows range les raccourcis des logiciels installés
SHORTCUT_DIRS = [
    Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "Microsoft/Windows/Start Menu/Programs",
    Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
    Path.home() / "Desktop",
    Path.home() / "OneDrive/Bureau",
    Path.home() / "OneDrive/Desktop",
    Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "Desktop",
]
IGNORED_WORDS = ("uninstall", "désinstaller", "desinstaller")  # raccourcis à ne jamais lancer

# Surnoms -> nom du raccourci (quand le jeu se lance via un launcher au nom différent)
SHORTCUT_ALIASES = {
    "league of legends": "client riot", "lol": "client riot", "valorant": "client riot",
    "obs": "obs studio",
}


@functools.lru_cache(maxsize=1)  # le scan n'est fait qu'une fois par lancement de Jarvis
def installed_apps():
    """Tous les raccourcis trouvés : {nom en minuscules: chemin du .lnk}."""
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
    """Trouve le raccourci le plus proche du nom demandé, ou None."""
    apps = installed_apps()
    name = SHORTCUT_ALIASES.get(name, name)
    # 1. Nom exact
    if name in apps:
        return apps[name]
    # 2. Nom contenu : « photoshop » trouve « Adobe Photoshop 2024 » (le plus court gagne)
    contains = sorted((n for n in apps if name in n), key=len)
    if contains:
        return apps[contains[0]]
    # 3. Faute de frappe : « discrod » trouve « discord »
    close = difflib.get_close_matches(name, apps, n=1, cutoff=0.75)
    return apps[close[0]] if close else None


@tool("🚀 Ouverture de l'application...")
def ouvrir_application(nom: str) -> str:
    """Ouvre n'importe quelle application ou jeu installé sur le PC (Chrome, Spotify, Discord, League of Legends, Word, calculatrice...).

    Args:
        nom: Le nom de l'application
    """
    key = nom.lower().strip()
    # 1. Application courante connue (liste APPS)
    target = APPS.get(key)
    if not target:
        # 2. Raccourci du menu Démarrer ou du Bureau
        shortcut = find_installed_app(key)
        if shortcut:
            os.startfile(shortcut)
            return f"{shortcut.stem} lancé."
        # 3. Nom connu contenu dans la demande (« ouvre google chrome » -> chrome)
        target = next((cmd for alias, cmd in APPS.items() if alias in key), None)
    # 4. Programme présent dans le PATH (nom simple uniquement : pas d'injection de commande)
    if not target and re.fullmatch(r"[\w.-]+", key) and shutil.which(key):
        target = key
    if not target:
        suggestions = difflib.get_close_matches(key, installed_apps(), n=3, cutoff=0.4)
        hint = f" Tu voulais peut-être : {', '.join(suggestions)} ?" if suggestions else ""
        return f"Aucune application « {nom} » trouvée sur le PC.{hint}"
    subprocess.Popen(["cmd", "/c", "start", "", target], creationflags=subprocess.CREATE_NO_WINDOW)
    return f"{nom} lancé."


# Programmes à ne JAMAIS fermer : Windows lui-même, Ollama (le cerveau), Python (Jarvis !)
PROTECTED = {"explorer", "python", "pythonw", "ollama", "ollama app", "svchost", "system", "csrss",
             "wininit", "winlogon", "lsass", "services", "dwm", "smss", "conhost", "sihost",
             "fontdrvhost", "taskhostw", "runtimebroker", "searchhost", "startmenuexperiencehost",
             "shellexperiencehost", "textinputhost", "ctfmon", "registry", "audiodg"}

# Nom dit par l'utilisateur -> nom(s) du programme en cours d'exécution (sans « .exe »)
PROCESS_NAMES = {
    "word": ["winword"], "powerpoint": ["powerpnt"], "edge": ["msedge"],
    "vs code": ["code"], "vscode": ["code"], "visual studio code": ["code"],
    "bloc-notes": ["notepad"], "calculatrice": ["calculatorapp", "calculator"], "paint": ["mspaint"],
    "gestionnaire des tâches": ["taskmgr"], "obs": ["obs64"], "obs studio": ["obs64"],
    "league of legends": ["league of legends", "leagueclientux", "leagueclient", "riotclientux"],
    "lol": ["league of legends", "leagueclientux", "leagueclient", "riotclientux"],
    "riot": ["riotclientux", "riotclientservices"],
}


# Programmes où l'on peut avoir du travail non enregistré : jamais de fermeture forcée
NEVER_FORCE = {"winword", "excel", "powerpnt", "notepad", "code", "mspaint", "onenote", "outlook"}


def running_programs():
    """Noms (en minuscules, sans .exe) des programmes en cours d'exécution."""
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
    # Correspondance STRICTE : un nom trop vague fermerait le mauvais programme (« spotify »
    # visait « spotifyxboxgamebarwebview », un module de la Xbox Game Bar).
    targets = [p for p in running if p in wanted]
    if not targets:  # presque exact : « obs » -> « obs64 » (3 caractères de plus au maximum)
        targets = [p for p in running if p.startswith(compact) and len(p) - len(compact) <= 3]
    if not targets:  # faute de frappe : « discrod » -> « discord »
        targets = difflib.get_close_matches(compact, running, n=1, cutoff=0.85)
    if not targets:
        return f"Aucune application « {nom} » n'est ouverte."

    def kill(programs, force):
        # Sans /F : Windows demande poliment à l'appli de se fermer (comme cliquer sur la croix),
        # elle peut donc proposer d'enregistrer. Avec /F : fermeture immédiate, sans question.
        for program in programs:
            command = ["taskkill", "/IM", f"{program}.exe", "/T"] + (["/F"] if force else [])
            subprocess.run(command, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        time.sleep(2)  # laisse le temps de se fermer
        return [p for p in programs if p in running_programs()]

    # Toujours une fermeture normale d'abord (le modèle demande parfois « forcer » sans raison)
    still_open = kill(targets, force=False)
    if still_open and forcer:
        # Forcer seulement si aucun travail ne peut être perdu (jamais Word, Bloc-notes...)
        still_open = kill([p for p in still_open if p not in NEVER_FORCE], force=True) + \
                     [p for p in still_open if p in NEVER_FORCE]
    if still_open:
        return (f"Fermeture demandée, mais {', '.join(still_open)} est encore ouvert : il attend peut-être "
                "une confirmation (enregistrer ?) ou reste dans la barre des tâches.")
    return f"{', '.join(targets)} fermé."


# --- Musique ---

VK_MEDIA_NEXT, VK_MEDIA_PREV, VK_MEDIA_PLAY_PAUSE = 0xB0, 0xB1, 0xB3  # touches multimédia


@tool("🎵 Contrôle de la musique...")
def controle_musique(action: str) -> str:
    """Contrôle la musique ou la vidéo DÉJÀ en cours (Spotify, YouTube...) : mettre en pause,
    reprendre, passer à la chanson suivante (« suivant », « passe », « prochaine ») ou revenir
    à la précédente. Pour lancer une nouvelle chanson précise, utiliser jouer_musique.

    Args:
        action: « pause », « lecture », « suivant » ou « precedent »
    """
    action = action.lower().strip()
    if action.startswith(("suiv", "next", "passe", "proch")):  # « prochain » avant le test « pr »
        _press(VK_MEDIA_NEXT)
        return "Morceau suivant."
    if action.startswith(("pr", "prev", "reviens")):  # précédent / precedent / previous
        _press(VK_MEDIA_PREV)
        return "Morceau précédent."
    _press(VK_MEDIA_PLAY_PAUSE)  # une seule touche pour pause ET reprise (bascule)
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
        # Sans compte développeur Spotify, on peut ouvrir la recherche mais pas lancer la lecture
        os.startfile(f"spotify:search:{quote(recherche)}")
        return f"Recherche « {recherche} » ouverte dans Spotify : il reste à cliquer sur le titre."
    url = find_youtube_video(recherche)
    if url:
        webbrowser.open(url)  # une vidéo YouTube se lance toute seule à l'ouverture
        return f"Lecture de « {recherche} » sur YouTube : {url}"
    # Dernier recours : la page de résultats YouTube (il faudra cliquer sur une vidéo)
    webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(recherche)}")
    return f"Résultats YouTube pour « {recherche} » ouverts : il reste à choisir la vidéo."


def find_youtube_video(recherche):
    """Adresse de la 1re vidéo YouTube trouvée, ou None. Le service de recherche est capricieux :
    on essaie la recherche de vidéos, puis une recherche web classique filtrée sur YouTube."""
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
            continue  # « aucun résultat » ou service indisponible : on passe à l'essai suivant
    return None


@tool("🌍 Ouverture du navigateur...")
def ouvrir_site(adresse_ou_recherche: str) -> str:
    """Ouvre un site web dans le navigateur, ou lance une recherche Google.

    Args:
        adresse_ou_recherche: Une URL (ex: youtube.com) ou des mots à rechercher
    """
    target = adresse_ou_recherche.strip()
    # Ressemble à une adresse (« youtube.com », « https://x.fr/page ») ? Sinon : recherche Google
    if re.fullmatch(r"(https?://)?[\w-]+(\.[\w-]+)+(/\S*)?", target):
        url = target if target.startswith("http") else f"https://{target}"
    else:
        url = f"https://www.google.com/search?q={quote_plus(target)}"
    webbrowser.open(url)
    return f"Ouvert : {url}"


# Le volume est piloté en simulant les touches multimédia du clavier (aucune dépendance).
# Codes des touches Windows : muet, volume -, volume +
VK_VOLUME_MUTE, VK_VOLUME_DOWN, VK_VOLUME_UP = 0xAD, 0xAE, 0xAF


def _press(vk, times=1):
    """Appuie puis relâche une touche, `times` fois."""
    for _ in range(times):
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP


@tool("🔊 Réglage du volume...")
def regler_volume(niveau: int) -> str:
    """Règle le volume du PC.

    Args:
        niveau: Le volume voulu, de 0 (muet) à 100
    """
    niveau = max(0, min(100, int(niveau)))
    _press(VK_VOLUME_DOWN, 50)          # chaque appui = 2 %, on part de 0
    _press(VK_VOLUME_UP, round(niveau / 2))
    return f"Volume réglé à {niveau} %."


@tool("🔇 Coupure du son...")
def couper_son() -> str:
    """Coupe ou rétablit le son du PC (bascule)."""
    _press(VK_VOLUME_MUTE)
    return "Son basculé (coupé / rétabli)."


@tool("🔒 Verrouillage...")
def verrouiller_pc() -> str:
    """Verrouille la session Windows."""
    ctypes.windll.user32.LockWorkStation()
    return "Session verrouillée."
