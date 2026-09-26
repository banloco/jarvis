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
import ctypes
import difflib
import functools
import os
import re
import shutil
import subprocess
import webbrowser
from pathlib import Path
from urllib.parse import quote, quote_plus
from urllib.request import Request, urlopen
from ddgs import DDGS
from jarvis_core import add_fact, now_text

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
