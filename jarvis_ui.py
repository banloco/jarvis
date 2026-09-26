"""Jarvis's window, Iron Man style. Run: python jarvis_ui.py

All the smarts are in jarvis_core.py; this file only deals with what you see.
On the left: the arc reactor (it reacts to what Jarvis is doing), the clock, the status
lights and the buttons. On the right: the conversation.

Three ways to talk to Jarvis: type, click 🎤, or say "Hey Jarvis"
(wake word, see jarvis_wake.py; the ÉCOUTE button turns it on and off).
GESTES button: control with hand gestures and moves (see jarvis_gestures.py).
INTERFACE HOLO button: full-screen holographic display you drive with your hand (see jarvis_holo.py).

One thing to keep in mind: the model takes several seconds to answer. So the window doesn't
freeze, each question is handled in its own thread. But Tkinter doesn't allow touching the
window from another thread, so those threads send their updates through a queue (self.ui),
which the window empties every 30 ms.
"""
import queue
import sys
import threading
import time
import winsound
from datetime import date, datetime
import customtkinter as ctk
import pystray
import jarvis_hud as hud
from jarvis_core import (BASE_DIR, MODEL, load_history, save_history, listen_voice, ask_model,
                         warm_up, load_json, save_json)
from jarvis_voice import Speaker, SentenceStreamer
from jarvis_tools import TOOLS, build_briefing
from jarvis_wake import WakeWordListener
from jarvis_config import USER_LABEL, greeting
from jarvis_gestures import GestureWatcher, GESTURE_ACTIONS, model_ready
from jarvis_holo import HoloHUD
from jarvis_reminders import reminders

ctk.set_appearance_mode("dark")
STATE_FILE = BASE_DIR / "jarvis_state.json"  # a little state kept between runs (date of the last briefing)


def hud_font(size, bold=False):
    return ctk.CTkFont(family=hud.FONT, size=size, weight="bold" if bold else "normal")


class JarvisApp(ctk.CTk):
    def __init__(self, hidden=False):
        """hidden=True: start without a window, just the icon next to the clock
        (that's how it starts with Windows)."""
        super().__init__(fg_color=hud.BG)
        from PIL import ImageTk
        self.icon_image = ImageTk.PhotoImage(hud.make_icon(64))  # keep a reference, or it gets garbage-collected
        self.iconphoto(True, self.icon_image)                   # window and taskbar icon
        self.title("J.A.R.V.I.S")
        self.geometry("1100x680")
        self.minsize(900, 560)
        self.history = load_history()    # recent conversation (shared with jarvis.py)
        self.speaker = Speaker()
        self.busy = False                # True while a question is being handled
        self.listening = False           # True while listening to the mic (brighter reactor)
        self.error_until = 0.0           # the reactor stays red until then
        self.status_text = ""            # status text, also read by the holo display
        self.holo = None                 # the holographic display, when it's open
        self.gestures_before_holo = False  # whether to turn gesture watching back on when the HUD closes
        self.ui_queue = queue.Queue()    # screen updates sent over by other threads
        # Wake word: paused while Jarvis is thinking or talking
        self.wake = WakeWordListener(on_wake=self.on_wake, on_command=self.on_voice_command,
                                     on_error=lambda msg: self.ui(self.on_wake_error, msg),
                                     is_paused=lambda: self.busy or self.speaker.is_speaking(),
                                     on_idle=self.on_followup_idle)
        # Gestures start off (the webcam only turns on with the GESTES button)
        self.gestures = GestureWatcher(on_gesture=lambda name: self.ui(self.on_gesture, name),
                                       on_error=lambda msg: self.ui(self.on_gesture_error, msg))
        self.setup_ui()
        # Warm up the model while the window opens (and start Ollama if it isn't running)
        warm_up(self.history, TOOLS,
                on_error=lambda msg: self.ui(self.show_error, msg),
                on_status=lambda msg: self.ui(self.set_status, msg))
        self.poll_ui_queue()
        self.tick_clock()
        self.wake.start()
        # Reminders: shown on screen and said out loud (checker thread -> self.ui queue)
        reminders.on_due = lambda message, late: self.ui(self.announce_reminder, message, late)
        reminders.start()
        # Background mode: the X hides the window, Jarvis keeps running next to the clock
        self.protocol("WM_DELETE_WINDOW", self.hide_to_tray)
        self.tray_hint_shown = False
        self.setup_tray()
        if hidden:
            self.withdraw()
        self.after(1700, self.greet)  # after the reactor's startup animation

    # --- Background mode (icon next to the clock) ---

    def setup_tray(self):
        """Icon in the notification area. Its menu runs in another thread, so it goes
        through self.ui like everything else that touches the window."""
        menu = pystray.Menu(
            pystray.MenuItem("Afficher Jarvis", lambda: self.ui(self.show_window), default=True),
            pystray.MenuItem("Interface holographique", lambda: self.ui(self.open_holo)),
            pystray.MenuItem("Quitter", lambda: self.ui(self.quit_app)))
        self.tray = pystray.Icon("jarvis", hud.make_icon(64), "Jarvis", menu)
        self.tray.run_detached()

    def hide_to_tray(self):
        self.withdraw()
        if not self.tray_hint_shown:  # explain once where Jarvis went
            self.tray_hint_shown = True
            self.tray.notify("Jarvis reste actif en arrière-plan. Clic droit sur son icône > Quitter "
                             "pour l'arrêter.", "Jarvis")

    def show_window(self):
        self.deiconify()
        self.lift()
        self.focus_force()

    def quit_app(self):
        self.wake.enabled = False
        self.gestures.enabled = False  # turns the webcam off if it was on
        if self.holo:
            self.holo.close()
        self.speaker.stop()
        self.tray.stop()
        # Don't cancel all the scheduled self.after tasks first: CustomTkinter needs them
        # to tear itself down, otherwise closing hangs
        self.destroy()

    # --- Building the window ---

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header across the full width
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(16, 0))
        ctk.CTkLabel(header, text="J.A.R.V.I.S", font=hud_font(30, True),
                     text_color=hud.CYAN).pack(side="left")
        ctk.CTkLabel(header, text="  JUST A RATHER VERY INTELLIGENT SYSTEM", font=hud_font(11),
                     text_color=hud.TEXT_DIM).pack(side="left", pady=(10, 0))
        self.clock = ctk.CTkLabel(header, text="", font=hud_font(13), text_color=hud.CYAN)
        self.clock.pack(side="right")
        ctk.CTkFrame(self, height=1, fg_color=hud.CYAN_DIM).grid(  # thin line under the header
            row=0, column=0, columnspan=2, sticky="sew", padx=24)

        self.setup_side_panel()
        self.setup_chat()

    def setup_side_panel(self):
        """Left column: reactor, status, status lights, buttons."""
        side = ctk.CTkFrame(self, fg_color="transparent", width=300)
        side.grid(row=1, column=0, sticky="ns", padx=(24, 12), pady=16)

        self.reactor = hud.ArcReactor(side, 260, get_state=self.reactor_state)
        self.reactor.pack(pady=(4, 8))

        self.status = ctk.CTkLabel(side, text="INITIALISATION...", font=hud_font(13, True),
                                   text_color=hud.CYAN, wraplength=280)
        self.status.pack(pady=(0, 14))

        # Status lights: ● on, ○ off
        box = ctk.CTkFrame(side, fg_color=hud.PANEL, border_color=hud.CYAN_DIM, border_width=1,
                           corner_radius=6)
        box.pack(fill="x", pady=(0, 14))
        self.indicators = {}
        for key in ["VOIX", "ÉCOUTE", "GESTES", "MODÈLE"]:
            row = ctk.CTkFrame(box, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=3)
            ctk.CTkLabel(row, text=key, font=hud_font(12), text_color=hud.TEXT_DIM).pack(side="left")
            self.indicators[key] = ctk.CTkLabel(row, text="", font=hud_font(12, True))
            self.indicators[key].pack(side="right")

        # Toggle buttons
        buttons = ctk.CTkFrame(side, fg_color="transparent")
        buttons.pack(fill="x")
        buttons.grid_columnconfigure((0, 1, 2), weight=1)
        self.mute_btn = self.hud_button(buttons, "VOIX", self.toggle_mute)
        self.wake_btn = self.hud_button(buttons, "ÉCOUTE", self.toggle_wake)
        self.gesture_btn = self.hud_button(buttons, "GESTES", self.toggle_gestures)
        for i, b in enumerate([self.mute_btn, self.wake_btn, self.gesture_btn]):
            b.grid(row=0, column=i, padx=3, sticky="ew")
        self.hud_button(side, "INTERFACE HOLO", self.open_holo).pack(fill="x", pady=(8, 0))
        self.hud_button(side, "NOUVELLE CONVERSATION", self.clear_history).pack(fill="x", pady=(8, 0))
        self.update_indicators()

    def setup_chat(self):
        """Right column: the conversation and the input box."""
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew", padx=(12, 24), pady=16)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.chat_box = ctk.CTkTextbox(right, font=hud_font(14), wrap="word", state="disabled",
                                       fg_color=hud.PANEL, border_color=hud.CYAN_DIM, border_width=1,
                                       text_color=hud.WHITE, corner_radius=6)
        self.chat_box.grid(row=0, column=0, sticky="nsew")
        self.chat_box.tag_config("user", foreground=hud.USER)
        self.chat_box.tag_config("jarvis", foreground=hud.CYAN)
        self.chat_box.tag_config("error", foreground=hud.ERROR)

        bar = ctk.CTkFrame(right, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        bar.grid_columnconfigure(0, weight=1)
        self.input_field = ctk.CTkEntry(bar, placeholder_text="Parlez à Jarvis...", height=42,
                                        font=hud_font(14), fg_color=hud.PANEL, text_color=hud.WHITE,
                                        border_color=hud.CYAN_DIM, placeholder_text_color=hud.TEXT_DIM)
        self.input_field.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.input_field.bind("<Return>", lambda e: self.send_text())
        self.input_field.focus()
        self.send_btn = self.hud_button(bar, "ENVOYER", self.send_text, width=110, height=42)
        self.send_btn.grid(row=0, column=1)
        self.voice_btn = self.hud_button(bar, "🎤", self.send_voice, width=50, height=42)
        self.voice_btn.grid(row=0, column=2, padx=(8, 0))

    def hud_button(self, parent, text, command, width=0, height=32):
        return ctk.CTkButton(parent, text=text, command=command, width=width, height=height,
                             font=hud_font(12, True), fg_color="transparent", hover_color=hud.CYAN_DIM,
                             border_color=hud.CYAN, border_width=1, text_color=hud.CYAN, corner_radius=4)

    # --- Visual state ---

    def reactor_state(self):
        """Called about 30 times a second by the reactor."""
        if time.monotonic() < self.error_until:
            return "error"
        if self.listening:
            return "listening"
        if self.speaker.is_speaking():
            return "speaking"
        if self.busy:
            return "thinking"
        return "idle"

    def update_indicators(self):
        states = {"VOIX": not self.speaker.muted, "ÉCOUTE": self.wake.enabled, "GESTES": self.gestures.enabled}
        for key, on in states.items():
            self.indicators[key].configure(text="● ACTIF" if on else "○ COUPÉ",
                                           text_color=hud.CYAN if on else hud.TEXT_DIM)
        self.indicators["MODÈLE"].configure(text=MODEL.upper(), text_color=hud.CYAN)
        for btn, on in [(self.mute_btn, states["VOIX"]), (self.wake_btn, states["ÉCOUTE"]),
                        (self.gesture_btn, states["GESTES"])]:
            btn.configure(fg_color=hud.CYAN_DIM if on else "transparent")

    def tick_clock(self):
        self.clock.configure(text=datetime.now().strftime("%d.%m.%Y  —  %H:%M:%S"))
        self.after(1000, self.tick_clock)

    def show_previous_conversation(self, count=10):
        """Show the last few exchanges again (Jarvis remembers them, so you might as well see them)."""
        shown = [m for m in self.history if m["role"] in ("user", "assistant") and m["content"].strip()]
        if not shown:
            return
        self.write("── conversation précédente ──\n\n", "jarvis")
        for m in shown[-count:]:
            self.add_message(USER_LABEL if m["role"] == "user" else "Jarvis", m["content"])
        self.write("── nouvelle session ──\n\n", "jarvis")

    def greet(self):
        """Greeting: the full briefing on the first start of the day, just a hello after that."""
        self.show_previous_conversation()
        hour = datetime.now().hour
        hello = "Bonjour" if 5 <= hour < 18 else "Bonsoir"
        state = load_json(STATE_FILE, {})
        if state.get("last_briefing") == date.today().isoformat():
            self.say_and_show(f"{greeting(hello)} Tous les systèmes sont opérationnels.")
            return
        state["last_briefing"] = date.today().isoformat()
        save_json(STATE_FILE, state)
        self.set_status("Préparation du briefing...")
        # The weather comes from the Internet, so the briefing is built outside the window thread
        threading.Thread(target=lambda: self.ui(self.say_and_show, build_briefing(hello)), daemon=True).start()

    def say_and_show(self, message):
        self.add_message("Jarvis", message)
        self.speaker.say(message)
        self.set_status(self.idle_status())

    # --- Talking between threads and the window ---

    def ui(self, fn, *args):
        """From any thread: ask the window to run fn(*args) as soon as it can."""
        self.ui_queue.put((fn, args))

    def poll_ui_queue(self):
        """Run the pending updates, then check again in 30 ms."""
        try:
            while True:
                fn, args = self.ui_queue.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        self.after(30, self.poll_ui_queue)

    # --- Display ---

    def write(self, text, tag=None):
        """Add text at the end of the chat (tag = optional color)."""
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", text, tag)
        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")  # scroll to the bottom

    def add_message(self, sender, message, tag=None):
        """Show a whole message: "NAME > message"."""
        tag = tag or ("jarvis" if sender == "Jarvis" else "user")
        self.write(f"{sender.upper()} > ", tag)
        self.write(f"{message}\n\n")

    def set_status(self, text):
        self.status_text = text
        self.status.configure(text=text.upper())

    def idle_status(self):
        return "Dites « Hey Jarvis »" if self.wake.enabled else "En attente"

    def set_busy(self, busy, status=None):
        """Disable (or re-enable) the buttons while a question is being handled."""
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.send_btn.configure(state=state)
        self.voice_btn.configure(state=state)
        self.set_status(status or self.idle_status())

    def announce_reminder(self, message, late_minutes):
        """A reminder is due: beep, message, voice, and the window comes to the front."""
        text = f"Rappel : {message}"
        if late_minutes:  # came due while Jarvis was closed
            text = f"Rappel manqué (il y a {late_minutes} min) : {message}"
        threading.Thread(target=lambda: [winsound.Beep(988, 150) for _ in range(3)], daemon=True).start()
        self.add_message("Jarvis", "⏰ " + text)
        self.speaker.say(text)
        self.deiconify()  # bring the window back if it was minimized
        self.lift()
        self.attributes("-topmost", True)
        self.after(1000, lambda: self.attributes("-topmost", False))

    def show_error(self, message):
        self.error_until = time.monotonic() + 3  # red reactor for 3 seconds
        self.add_message("Jarvis", message, "error")

    # --- Buttons ---

    def toggle_mute(self):
        self.speaker.muted = not self.speaker.muted
        if self.speaker.muted:
            self.speaker.stop()  # also cuts off the current sentence
        self.update_indicators()

    def toggle_wake(self):
        self.wake.enabled = not self.wake.enabled
        self.update_indicators()
        if not self.busy:
            self.set_status(self.idle_status())

    def on_wake_error(self, message):
        """The wake word can't work: switch it off and say so."""
        self.wake.enabled = False
        self.wake_btn.configure(state="disabled")
        self.update_indicators()
        self.set_status(message)

    def toggle_gestures(self):
        self.gestures.enabled = not self.gestures.enabled
        if self.gestures.enabled:
            self.gestures.start()  # only starts the thread once
        self.update_indicators()
        if not self.busy:
            if not self.gestures.enabled:
                self.set_status(self.idle_status())
            elif model_ready():
                self.set_status("✋ Gestes et mouvements activés")
            else:  # without a trained model, swipe and pinch still work
                self.set_status("✋ Mouvements activés (aucun geste entraîné)")

    def on_gesture(self, name):
        """A gesture was recognized: run its action and show it."""
        action, description = GESTURE_ACTIONS.get(name, (None, "aucune action"))
        if action == "ecouter":
            self.send_voice()  # same as clicking 🎤 (ignored if Jarvis is busy)
            return             # send_voice shows "Écoute en cours" by itself
        if callable(action):
            threading.Thread(target=action, daemon=True).start()  # don't freeze the window
        if not self.busy:      # don't overwrite "Jarvis réfléchit..."
            self.set_status(f"✋ {name} → {description}")

    def on_gesture_error(self, message):
        self.gestures.enabled = False
        self.update_indicators()
        self.set_status(message)

    # --- Holographic display ---

    def open_holo(self):
        """Open the full-screen holographic display you drive with your hand (see jarvis_holo.py)."""
        if self.holo and self.holo.running:
            return
        # Only one app at a time can use the webcam, so gesture watching is paused
        self.gestures_before_holo = self.gestures.enabled
        self.gestures.enabled = False
        self.update_indicators()
        # These are called from the HUD's thread: they either just read values,
        # or go through self.ui to act on the window
        self.holo = HoloHUD(on_talk=lambda: self.ui(self.send_voice),
                            get_state=self.reactor_state,
                            get_status=lambda: self.status_text,
                            get_reply=self.last_reply,
                            on_close=lambda: self.ui(self.on_holo_closed),
                            on_error=lambda msg: self.ui(self.show_error, msg))
        self.holo.start(wait_for=lambda: not self.gestures.running)  # waits until the webcam is free

    def on_holo_closed(self):
        if self.gestures_before_holo:  # gesture watching picks up again
            self.gestures.enabled = True
            self.gestures.start()
        self.update_indicators()

    def last_reply(self):
        """Jarvis's last answer (for the holo display)."""
        for m in reversed(list(self.history)):  # a copy, since the history changes while an answer comes in
            if m["role"] == "assistant" and m["content"].strip():
                return m["content"]
        return ""

    def clear_history(self):
        """Clear the recent conversation. Long-term memory and facts are left alone."""
        if self.busy:
            return
        self.history.clear()
        save_history(self.history)
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0", "end")
        self.chat_box.configure(state="disabled")
        self.add_message("Jarvis", "Conversation effacée. Mes souvenirs à long terme sont conservés.")

    # --- Handling a question (outside the window thread) ---

    def send_text(self):
        text = self.input_field.get().strip()
        if not text or self.busy:
            return
        self.input_field.delete(0, "end")
        self.speaker.stop()  # cut Jarvis off if he was still talking
        self.add_message(USER_LABEL, text)
        self.set_busy(True, "Jarvis réfléchit...")
        threading.Thread(target=self.process, args=(text,), daemon=True).start()

    def send_voice(self):
        if self.busy:
            return
        self.speaker.stop()
        self.listening = True
        self.set_busy(True, "🎤 Écoute en cours...")
        threading.Thread(target=self.voice_process, daemon=True).start()

    # Called from the wake word thread
    def on_wake(self, followup=False):
        self.busy = True  # set right away, so it can't trigger twice before the screen updates
        self.listening = True
        self.ui(self.set_busy, True, "🎤 Je vous écoute encore..." if followup else "🎤 Je vous écoute...")

    def on_followup_idle(self):
        """Conversation mode: nobody followed up, so Jarvis goes back to sleep."""
        self.listening = False
        self.ui(self.set_busy, False)

    def on_voice_command(self, text, error):
        self.listening = False
        if not text:
            self.ui(self.set_busy, False, error)
            return
        self.ui(self.add_message, USER_LABEL, text)
        self.ui(self.set_status, "Jarvis réfléchit...")
        threading.Thread(target=self.process, args=(text, True), daemon=True).start()

    def voice_process(self):
        text, error = listen_voice()
        self.listening = False
        if not text:
            self.ui(self.set_busy, False, error)
            return
        self.ui(self.add_message, USER_LABEL, text)
        self.ui(self.set_status, "Jarvis réfléchit...")
        self.process(text, True)

    def process(self, user_input, from_voice=False):
        """Runs in a thread: ask Jarvis, then show and speak the answer as it comes in.
        from_voice: the question was spoken -> conversation mode after the answer."""
        try:
            self.ui(self.write, "JARVIS > ", "jarvis")
            voice = SentenceStreamer(self.speaker)  # speaks sentence by sentence, while the text is still coming

            def on_token(piece):
                self.ui(self.write, piece)
                voice.feed(piece)

            try:
                ask_model(self.history, user_input, TOOLS, on_token=on_token,
                          on_status=lambda text: self.ui(self.set_status, text))
            finally:
                self.ui(self.write, "\n\n")
            voice.flush()  # last sentence
            if from_voice:
                # Conversation mode: once Jarvis is done talking, he listens for a follow-up
                # for a few seconds, no "Hey Jarvis" needed (the wake word waits for the voice to finish)
                self.wake.listen_again()
        except ConnectionError:
            self.ui(self.show_error, "Ollama ne répond pas. Lancez-le avec `ollama serve`.")
        except Exception as e:
            self.ui(self.show_error, f"Erreur : {e}")
        finally:
            self.ui(self.set_busy, False)


if __name__ == "__main__":
    # When started with pythonw (shortcut, Windows startup) there's no console, so
    # messages and errors go to jarvis.log instead, to help figure out what went wrong.
    if sys.stdout is None:
        log = open(BASE_DIR / "jarvis.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
        print(f"\n--- Démarrage de Jarvis, {datetime.now():%Y-%m-%d %H:%M} ---")
    app = JarvisApp(hidden="--cache" in sys.argv)
    app.mainloop()
