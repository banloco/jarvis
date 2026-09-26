"""Timers and reminders.

Reminders are saved in reminders.json, so they survive Jarvis being closed.
A background thread checks every second whether one is due, and if so calls
on_due(message, minutes_late). The window hooks on_due up to the voice and the screen.
If a reminder came due while Jarvis was closed, it's announced at startup, with how late it is.

The tools creer_rappel / lister_rappels / annuler_rappel (jarvis_tools.py) all go through
the shared `reminders` instance.
"""
import threading
import time
from datetime import datetime
from pathlib import Path
from jarvis_core import load_json, save_json

REMINDERS_FILE = Path(__file__).resolve().parent / "reminders.json"
LATE_AFTER = 60  # seconds; past that, the reminder is announced as "missed"


class Reminders:
    def __init__(self, path=REMINDERS_FILE):
        self.path = path
        self.lock = threading.Lock()  # the tools (answer thread) and the checker run side by side
        self.items = load_json(path, [])  # [{"message", "due" (ISO date), "created"}]
        self.on_due = lambda message, late_minutes: print(f"⏰ Rappel : {message}")
        self.started = False

    def start(self):
        """Start checking in the background (only once)."""
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
        """Upcoming reminders, soonest first."""
        with self.lock:
            return sorted(self.items, key=lambda i: i["due"])

    def cancel(self, search):
        """Cancel the reminders whose message contains `search` ("tous" cancels them all).
        Returns the list of cancelled reminders."""
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
            for item in due:  # called outside the lock, since on_due can take a while
                late = (now - datetime.fromisoformat(item["due"])).total_seconds()
                self.on_due(item["message"], round(late / 60) if late > LATE_AFTER else 0)
            time.sleep(1)


reminders = Reminders()  # shared by the tools and the window
