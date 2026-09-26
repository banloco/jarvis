"""Sets up Jarvis's shortcuts.

    python installer.py             -> desktop shortcut + start with Windows
                                       + offline speech recognition (Vosk, ~41 MB)
    python installer.py --retirer   -> removes both shortcuts

The desktop shortcut opens the Jarvis window. The startup one launches Jarvis hidden
(--cache): only his icon shows up next to the clock, he listens for "Hey Jarvis" and gives
you the daily briefing. Both use pythonw.exe, which doesn't open a black console window
(messages go to jarvis.log instead).
"""
import sys
from pathlib import Path
import win32com.client  # comes with pywin32; lets us create Windows shortcuts (.lnk)
import jarvis_hud as hud
from jarvis_core import download_offline_speech_model

BASE = Path(__file__).resolve().parent
PYTHONW = Path(sys.executable).with_name("pythonw.exe")
ICON = BASE / "jarvis.ico"

shell = win32com.client.Dispatch("WScript.Shell")
SHORTCUTS = {  # location -> arguments
    Path(shell.SpecialFolders("Desktop")) / "Jarvis.lnk": "",
    Path(shell.SpecialFolders("Startup")) / "Jarvis.lnk": "--cache",
}


def create_shortcut(path, args):
    shortcut = shell.CreateShortcut(str(path))
    shortcut.TargetPath = str(PYTHONW)
    shortcut.Arguments = f'"{BASE / "jarvis_ui.py"}" {args}'.strip()
    shortcut.WorkingDirectory = str(BASE)
    shortcut.IconLocation = str(ICON)
    shortcut.Description = "Jarvis, assistant personnel"
    shortcut.save()


def install():
    # .ico with several sizes (Windows picks the right one depending on where it's shown)
    hud.make_icon(256).save(ICON, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)])
    for path, args in SHORTCUTS.items():
        create_shortcut(path, args)
        print(f"Créé : {path}")
    print("Jarvis démarrera désormais avec Windows (caché, icône près de l'horloge).")
    print("Téléchargement de la reconnaissance vocale hors ligne (~41 Mo, une seule fois)...")
    download_offline_speech_model()
    print("Reconnaissance vocale hors ligne prête.")


def uninstall():
    for path in SHORTCUTS:
        if path.exists():
            path.unlink()
            print(f"Supprimé : {path}")


if __name__ == "__main__":
    uninstall() if "--retirer" in sys.argv else install()
