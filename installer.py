"""Installe les raccourcis de Jarvis.

    python installer.py             -> raccourci sur le Bureau + lancement au démarrage de Windows
                                       + reconnaissance vocale hors ligne (Vosk, ~41 Mo)
    python installer.py --retirer   -> supprime ces deux raccourcis

Le raccourci du Bureau ouvre la fenêtre de Jarvis. Celui du démarrage de Windows lance
Jarvis caché (--cache) : seule son icône apparaît près de l'horloge, il écoute « Hey Jarvis »
et fait le briefing du jour. Les deux utilisent pythonw.exe, qui n'ouvre pas de console noire
(les messages vont alors dans jarvis.log).
"""
import sys
from pathlib import Path
import win32com.client  # fourni par pywin32 : permet de créer des raccourcis Windows (.lnk)
import jarvis_hud as hud
from jarvis_core import download_offline_speech_model

BASE = Path(__file__).resolve().parent
PYTHONW = Path(sys.executable).with_name("pythonw.exe")
ICON = BASE / "jarvis.ico"

shell = win32com.client.Dispatch("WScript.Shell")
SHORTCUTS = {  # emplacement -> arguments
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
    # Icône .ico en plusieurs tailles (Windows choisit la bonne selon l'endroit)
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
