"""Réglages personnels de chaque utilisateur, lus dans config.json (jamais versionné).

config.json est facultatif. Exemple (voir config.example.json) :
    {"prenom": "Tony"}

Sans prénom, Jarvis parle de « l'utilisateur » au modèle et affiche « Vous » dans le chat.
"""
import json
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

try:
    CONFIG = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
except (OSError, ValueError):  # absent ou mal écrit : réglages par défaut
    CONFIG = {}

PRENOM = str(CONFIG.get("prenom", "")).strip()
USER_LABEL = PRENOM or "Vous"          # nom affiché devant ses messages
USER_REF = PRENOM or "l'utilisateur"   # façon d'en parler dans les instructions du modèle


def greeting(hello):
    """« Bonjour Tony. » ou « Bonjour. » sans prénom."""
    return f"{hello} {PRENOM}.".replace(" .", ".")
