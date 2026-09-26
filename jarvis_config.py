"""Your personal settings, read from config.json (never committed).

config.json is optional. Example (see config.example.json):
    {"name": "Tony"}

Without a name, Jarvis refers to you as "l'utilisateur" when talking to the model,
and shows "Vous" in the chat.
"""
import json
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

try:
    CONFIG = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
except (OSError, ValueError):  # missing or badly written: defaults it is
    CONFIG = {}

# "prenom" is the old name of the setting; still accepted so older config files keep working
NAME = str(CONFIG.get("name") or CONFIG.get("prenom") or "").strip()
USER_LABEL = NAME or "Vous"          # shown in front of your messages
USER_REF = NAME or "l'utilisateur"   # how the model's instructions refer to you


def greeting(hello):
    """"Bonjour Tony." or just "Bonjour." when no name is set."""
    return f"{hello} {NAME}.".replace(" .", ".")
