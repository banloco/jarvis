"""Minuteurs et rappels.

Les rappels sont enregistrés dans reminders.json : ils survivent à la fermeture de Jarvis.
Un fil d'exécution vérifie chaque seconde si un rappel est arrivé à échéance et appelle
alors on_due(message, retard_en_minutes). L'interface branche on_due sur la voix et l'écran.
Un rappel arrivé à échéance pendant que Jarvis était fermé est annoncé au démarrage,
avec son retard.

Utilisation : les outils creer_rappel / lister_rappels / annuler_rappel (jarvis_tools.py)
passent par l'instance partagée `reminders`.
"""
import threading
import time
from datetime import datetime
from pathlib import Path
from jarvis_core import load_json, save_json

REMINDERS_FILE = Path(__file__).resolve().parent / "reminders.json"
LATE_AFTER = 60  # secondes : au-delà, le rappel est annoncé comme « manqué »


class Reminders:
    def __init__(self, path=REMINDERS_FILE):
        self.path = path
        self.lock = threading.Lock()  # les outils (thread de réponse) et la vérification se croisent
        self.items = load_json(path, [])  # [{"message", "due" (date ISO), "created"}]
        self.on_due = lambda message, late_minutes: print(f"⏰ Rappel : {message}")
        self.started = False

    def start(self):
        """Lance la vérification en arrière-plan (une seule fois)."""
        if not self.started:
            self.started = True
            threading.Thread(target=self._loop, daemon=True).start()

    def add(self, message, due):
        with self.lock:
            item = {"message": message, "due": due.isoformat(timespec="seconds"),
                    "created": datetime.now().isoformat(timespec="seconds")}
            self.items.append(item)
            self._save()
        return item

    def pending(self):
        """Rappels à venir, du plus proche au plus lointain."""
        with self.lock:
            return sorted(self.items, key=lambda i: i["due"])

    def cancel(self, search):
        """Annule les rappels dont le message contient `search` (« tous » = tout annuler).
        Retourne la liste des rappels annulés."""
        search = search.lower().strip()
        with self.lock:
            removed = [i for i in self.items if search in ("tous", "tout") or search in i["message"].lower()]
            self.items = [i for i in self.items if i not in removed]
            self._save()
        return removed

    def _save(self):
        save_json(self.path, self.items)

    def _loop(self):
        while True:
            now = datetime.now()
            with self.lock:
                due = [i for i in self.items if datetime.fromisoformat(i["due"]) <= now]
                if due:
                    self.items = [i for i in self.items if i not in due]
                    self._save()
            for item in due:  # appelé hors du verrou : on_due peut prendre du temps
                late = (now - datetime.fromisoformat(item["due"])).total_seconds()
                self.on_due(item["message"], round(late / 60) if late > LATE_AFTER else 0)
            time.sleep(1)


reminders = Reminders()  # instance partagée par les outils et l'interface
