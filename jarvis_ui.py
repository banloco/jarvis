"""Interface graphique de Jarvis, style Iron Man. Lancer : python jarvis_ui.py

Toute l'intelligence est dans jarvis_core.py ; ce fichier ne gère que l'affichage.
À gauche : le réacteur arc (qui réagit à l'état de Jarvis), l'horloge, les voyants
et les boutons. À droite : la conversation.

Trois façons de parler à Jarvis : taper, cliquer sur 🎤, ou dire « Hey Jarvis »
(réveil vocal, voir jarvis_wake.py ; bouton ÉCOUTE pour l'activer / le couper).
Bouton GESTES : contrôle par gestes et mouvements de la main (voir jarvis_gestures.py).

Point important : le modèle met plusieurs secondes à répondre. Pour que la fenêtre
ne gèle pas, chaque question est traitée dans un thread séparé. Mais Tkinter interdit
de modifier la fenêtre depuis un autre thread : ces threads passent donc leurs
mises à jour par une file (self.ui), que la fenêtre vide toutes les 30 ms.
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
from jarvis_gestures import GestureWatcher, GESTURE_ACTIONS, model_ready
from jarvis_reminders import reminders

ctk.set_appearance_mode("dark")
STATE_FILE = BASE_DIR / "jarvis_state.json"  # petit état entre deux lancements (date du dernier briefing)


def hud_font(size, bold=False):
    return ctk.CTkFont(family=hud.FONT, size=size, weight="bold" if bold else "normal")


class JarvisApp(ctk.CTk):
    def __init__(self, hidden=False):
        """hidden=True : démarre sans fenêtre, seulement l'icône près de l'horloge
        (utilisé au démarrage de Windows)."""
        super().__init__(fg_color=hud.BG)
        from PIL import ImageTk
        self.icon_image = ImageTk.PhotoImage(hud.make_icon(64))  # gardée en mémoire (sinon effacée)
        self.iconphoto(True, self.icon_image)                   # icône de la fenêtre et de la barre des tâches
        self.title("J.A.R.V.I.S")
        self.geometry("1100x680")
        self.minsize(900, 560)
        self.history = load_history()    # conversation récente (partagée avec jarvis.py)
        self.speaker = Speaker()
        self.busy = False                # True pendant qu'une question est en cours
        self.listening = False           # True pendant l'écoute du micro (réacteur plus vif)
        self.error_until = 0.0           # le réacteur reste rouge jusqu'à cet instant
        self.ui_queue = queue.Queue()    # mises à jour de l'écran envoyées par les threads
        # Réveil vocal : en pause pendant que Jarvis réfléchit ou parle
        self.wake = WakeWordListener(on_wake=self.on_wake, on_command=self.on_voice_command,
                                     on_error=lambda msg: self.ui(self.on_wake_error, msg),
                                     is_paused=lambda: self.busy or self.speaker.is_speaking(),
                                     on_idle=self.on_followup_idle)
        # Gestes : désactivés au démarrage (la webcam ne s'allume que via le bouton GESTES)
        self.gestures = GestureWatcher(on_gesture=lambda name: self.ui(self.on_gesture, name),
                                       on_error=lambda msg: self.ui(self.on_gesture_error, msg))
        self.setup_ui()
        # Prépare le modèle pendant que la fenêtre s'ouvre (et lance Ollama s'il ne tourne pas)
        warm_up(self.history, TOOLS,
                on_error=lambda msg: self.ui(self.show_error, msg),
                on_status=lambda msg: self.ui(self.set_status, msg))
        self.poll_ui_queue()
        self.tick_clock()
        self.wake.start()
        # Rappels : annoncés à l'écran et à voix haute (thread de vérification -> file self.ui)
        reminders.on_due = lambda message, late: self.ui(self.announce_reminder, message, late)
        reminders.start()
        # Arrière-plan : la croix cache la fenêtre, Jarvis reste actif près de l'horloge
        self.protocol("WM_DELETE_WINDOW", self.hide_to_tray)
        self.tray_hint_shown = False
        self.setup_tray()
        if hidden:
            self.withdraw()
        self.after(1700, self.greet)  # après la séquence d'allumage du réacteur

    # --- Arrière-plan (icône près de l'horloge) ---

    def setup_tray(self):
        """Icône dans la zone de notification. Ses menus s'exécutent dans un autre thread :
        ils passent donc par self.ui, comme tout ce qui touche à la fenêtre."""
        menu = pystray.Menu(
            pystray.MenuItem("Afficher Jarvis", lambda: self.ui(self.show_window), default=True),
            pystray.MenuItem("Quitter", lambda: self.ui(self.quit_app)))
        self.tray = pystray.Icon("jarvis", hud.make_icon(64), "Jarvis", menu)
        self.tray.run_detached()

    def hide_to_tray(self):
        self.withdraw()
        if not self.tray_hint_shown:  # explique une fois où est passé Jarvis
            self.tray_hint_shown = True
            self.tray.notify("Jarvis reste actif en arrière-plan. Clic droit sur son icône > Quitter "
                             "pour l'arrêter.", "Jarvis")

    def show_window(self):
        self.deiconify()
        self.lift()
        self.focus_force()

    def quit_app(self):
        self.wake.enabled = False
        self.gestures.enabled = False  # éteint la webcam si elle était allumée
        self.speaker.stop()
        self.tray.stop()
        # Ne PAS annuler toutes les tâches programmées (self.after) avant : CustomTkinter
        # en a besoin pour se détruire, sinon la fermeture reste bloquée
        self.destroy()

    # --- Construction de la fenêtre ---

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # En-tête sur toute la largeur
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(16, 0))
        ctk.CTkLabel(header, text="J.A.R.V.I.S", font=hud_font(30, True),
                     text_color=hud.CYAN).pack(side="left")
        ctk.CTkLabel(header, text="  JUST A RATHER VERY INTELLIGENT SYSTEM", font=hud_font(11),
                     text_color=hud.TEXT_DIM).pack(side="left", pady=(10, 0))
        self.clock = ctk.CTkLabel(header, text="", font=hud_font(13), text_color=hud.CYAN)
        self.clock.pack(side="right")
        ctk.CTkFrame(self, height=1, fg_color=hud.CYAN_DIM).grid(  # fine ligne sous l'en-tête
            row=0, column=0, columnspan=2, sticky="sew", padx=24)

        self.setup_side_panel()
        self.setup_chat()

    def setup_side_panel(self):
        """Colonne de gauche : réacteur, statut, voyants, boutons."""
        side = ctk.CTkFrame(self, fg_color="transparent", width=300)
        side.grid(row=1, column=0, sticky="ns", padx=(24, 12), pady=16)

        self.reactor = hud.ArcReactor(side, 260, get_state=self.reactor_state)
        self.reactor.pack(pady=(4, 8))

        self.status = ctk.CTkLabel(side, text="INITIALISATION...", font=hud_font(13, True),
                                   text_color=hud.CYAN, wraplength=280)
        self.status.pack(pady=(0, 14))

        # Voyants : ● actif, ○ coupé
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

        # Boutons interrupteurs
        buttons = ctk.CTkFrame(side, fg_color="transparent")
        buttons.pack(fill="x")
        buttons.grid_columnconfigure((0, 1, 2), weight=1)
        self.mute_btn = self.hud_button(buttons, "VOIX", self.toggle_mute)
        self.wake_btn = self.hud_button(buttons, "ÉCOUTE", self.toggle_wake)
        self.gesture_btn = self.hud_button(buttons, "GESTES", self.toggle_gestures)
        for i, b in enumerate([self.mute_btn, self.wake_btn, self.gesture_btn]):
            b.grid(row=0, column=i, padx=3, sticky="ew")
        self.hud_button(side, "NOUVELLE CONVERSATION", self.clear_history).pack(fill="x", pady=(8, 0))
        self.update_indicators()

    def setup_chat(self):
        """Colonne de droite : conversation et saisie."""
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

    # --- État visuel ---

    def reactor_state(self):
        """Appelée ~30 fois par seconde par le réacteur."""
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
        """Réaffiche les derniers échanges (Jarvis s'en souvient : autant les voir)."""
        shown = [m for m in self.history if m["role"] in ("user", "assistant") and m["content"].strip()]
        if not shown:
            return
        self.write("── conversation précédente ──\n\n", "jarvis")
        for m in shown[-count:]:
            self.add_message("l'utilisateur" if m["role"] == "user" else "Jarvis", m["content"])
        self.write("── nouvelle session ──\n\n", "jarvis")

    def greet(self):
        """Accueil : briefing complet au premier lancement de la journée, simple salut ensuite."""
        self.show_previous_conversation()
        hour = datetime.now().hour
        hello = "Bonjour" if 5 <= hour < 18 else "Bonsoir"
        state = load_json(STATE_FILE, {})
        if state.get("last_briefing") == date.today().isoformat():
            self.say_and_show(f"{hello} l'utilisateur. Tous les systèmes sont opérationnels.")
            return
        state["last_briefing"] = date.today().isoformat()
        save_json(STATE_FILE, state)
        self.set_status("Préparation du briefing...")
        # La météo passe par Internet : on prépare le briefing hors du thread de la fenêtre
        threading.Thread(target=lambda: self.ui(self.say_and_show, build_briefing(hello)), daemon=True).start()

    def say_and_show(self, message):
        self.add_message("Jarvis", message)
        self.speaker.say(message)
        self.set_status(self.idle_status())

    # --- Communication entre threads et fenêtre ---

    def ui(self, fn, *args):
        """Depuis un thread : demande à la fenêtre d'exécuter fn(*args) dès que possible."""
        self.ui_queue.put((fn, args))

    def poll_ui_queue(self):
        """Exécute les mises à jour en attente, puis se reprogramme dans 30 ms."""
        try:
            while True:
                fn, args = self.ui_queue.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        self.after(30, self.poll_ui_queue)

    # --- Affichage ---

    def write(self, text, tag=None):
        """Ajoute du texte à la fin du chat (tag = couleur optionnelle)."""
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", text, tag)
        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")  # défile jusqu'en bas

    def add_message(self, sender, message, tag=None):
        """Affiche un message complet : « NOM > message »."""
        tag = tag or sender.lower()
        self.write(f"{sender.upper()} > ", tag)
        self.write(f"{message}\n\n")

    def set_status(self, text):
        self.status.configure(text=text.upper())

    def idle_status(self):
        return "Dites « Hey Jarvis »" if self.wake.enabled else "En attente"

    def set_busy(self, busy, status=None):
        """Bloque (ou débloque) les boutons pendant qu'une question est traitée."""
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.send_btn.configure(state=state)
        self.voice_btn.configure(state=state)
        self.set_status(status or self.idle_status())

    def announce_reminder(self, message, late_minutes):
        """Un rappel arrive : bip, message, voix, et fenêtre ramenée au premier plan."""
        text = f"Rappel : {message}"
        if late_minutes:  # arrivé pendant que Jarvis était fermé
            text = f"Rappel manqué (il y a {late_minutes} min) : {message}"
        threading.Thread(target=lambda: [winsound.Beep(988, 150) for _ in range(3)], daemon=True).start()
        self.add_message("Jarvis", "⏰ " + text)
        self.speaker.say(text)
        self.deiconify()  # ré-ouvre la fenêtre si elle était réduite
        self.lift()
        self.attributes("-topmost", True)
        self.after(1000, lambda: self.attributes("-topmost", False))

    def show_error(self, message):
        self.error_until = time.monotonic() + 3  # réacteur rouge 3 secondes
        self.add_message("Jarvis", message, "error")

    # --- Boutons ---

    def toggle_mute(self):
        self.speaker.muted = not self.speaker.muted
        if self.speaker.muted:
            self.speaker.stop()  # coupe aussi la phrase en cours
        self.update_indicators()

    def toggle_wake(self):
        self.wake.enabled = not self.wake.enabled
        self.update_indicators()
        if not self.busy:
            self.set_status(self.idle_status())

    def on_wake_error(self, message):
        """Le réveil vocal ne peut pas fonctionner : on le désactive et on prévient."""
        self.wake.enabled = False
        self.wake_btn.configure(state="disabled")
        self.update_indicators()
        self.set_status(message)

    def toggle_gestures(self):
        self.gestures.enabled = not self.gestures.enabled
        if self.gestures.enabled:
            self.gestures.start()  # ne démarre le thread qu'une fois
        self.update_indicators()
        if not self.busy:
            if not self.gestures.enabled:
                self.set_status(self.idle_status())
            elif model_ready():
                self.set_status("✋ Gestes et mouvements activés")
            else:  # sans modèle entraîné, glisser et pincer marchent quand même
                self.set_status("✋ Mouvements activés (aucun geste entraîné)")

    def on_gesture(self, name):
        """Un geste a été reconnu : on exécute son action et on l'affiche."""
        action, description = GESTURE_ACTIONS.get(name, (None, "aucune action"))
        if action == "ecouter":
            self.send_voice()  # comme un clic sur 🎤 (ignoré si Jarvis est occupé)
            return             # send_voice affiche lui-même « Écoute en cours »
        if callable(action):
            threading.Thread(target=action, daemon=True).start()  # ne pas geler la fenêtre
        if not self.busy:      # ne pas écraser « Jarvis réfléchit... »
            self.set_status(f"✋ {name} → {description}")

    def on_gesture_error(self, message):
        self.gestures.enabled = False
        self.update_indicators()
        self.set_status(message)

    def clear_history(self):
        """Vide la conversation récente. La mémoire à long terme et les faits restent intacts."""
        if self.busy:
            return
        self.history.clear()
        save_history(self.history)
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0", "end")
        self.chat_box.configure(state="disabled")
        self.add_message("Jarvis", "Conversation effacée. Mes souvenirs à long terme sont conservés.")

    # --- Traitement d'une question (hors du thread de la fenêtre) ---

    def send_text(self):
        text = self.input_field.get().strip()
        if not text or self.busy:
            return
        self.input_field.delete(0, "end")
        self.speaker.stop()  # on coupe Jarvis s'il parlait encore
        self.add_message("l'utilisateur", text)
        self.set_busy(True, "Jarvis réfléchit...")
        threading.Thread(target=self.process, args=(text,), daemon=True).start()

    def send_voice(self):
        if self.busy:
            return
        self.speaker.stop()
        self.listening = True
        self.set_busy(True, "🎤 Écoute en cours...")
        threading.Thread(target=self.voice_process, daemon=True).start()

    # Appelés depuis le thread du réveil vocal
    def on_wake(self, followup=False):
        self.busy = True  # immédiat : empêche un 2e déclenchement avant la mise à jour de l'écran
        self.listening = True
        self.ui(self.set_busy, True, "🎤 Je vous écoute encore..." if followup else "🎤 Je vous écoute...")

    def on_followup_idle(self):
        """Mode conversation : personne n'a enchaîné, Jarvis se rendort."""
        self.listening = False
        self.ui(self.set_busy, False)

    def on_voice_command(self, text, error):
        self.listening = False
        if not text:
            self.ui(self.set_busy, False, error)
            return
        self.ui(self.add_message, "l'utilisateur", text)
        self.ui(self.set_status, "Jarvis réfléchit...")
        threading.Thread(target=self.process, args=(text, True), daemon=True).start()

    def voice_process(self):
        text, error = listen_voice()
        self.listening = False
        if not text:
            self.ui(self.set_busy, False, error)
            return
        self.ui(self.add_message, "l'utilisateur", text)
        self.ui(self.set_status, "Jarvis réfléchit...")
        self.process(text, True)

    def process(self, user_input, from_voice=False):
        """Tourne dans un thread : interroge Jarvis, affiche et prononce la réponse au fil de l'eau.
        from_voice : la question a été posée à la voix -> mode conversation après la réponse."""
        try:
            self.ui(self.write, "JARVIS > ", "jarvis")
            voice = SentenceStreamer(self.speaker)  # parle phrase par phrase, pendant l'écriture

            def on_token(piece):
                self.ui(self.write, piece)
                voice.feed(piece)

            try:
                ask_model(self.history, user_input, TOOLS, on_token=on_token,
                          on_status=lambda text: self.ui(self.set_status, text))
            finally:
                self.ui(self.write, "\n\n")
            voice.flush()  # dernière phrase
            if from_voice:
                # Mode conversation : dès que Jarvis aura fini de parler, il écoute la suite
                # quelques secondes sans « Hey Jarvis » (le réveil vocal attend la fin de la voix)
                self.wake.listen_again()
        except ConnectionError:
            self.ui(self.show_error, "Ollama ne répond pas. Lancez-le avec `ollama serve`.")
        except Exception as e:
            self.ui(self.show_error, f"Erreur : {e}")
        finally:
            self.ui(self.set_busy, False)


if __name__ == "__main__":
    # Lancé avec pythonw (raccourci, démarrage de Windows), il n'y a pas de console :
    # les messages et erreurs sont alors écrits dans jarvis.log pour pouvoir diagnostiquer.
    if sys.stdout is None:
        log = open(BASE_DIR / "jarvis.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
        print(f"\n--- Démarrage de Jarvis, {datetime.now():%Y-%m-%d %H:%M} ---")
    app = JarvisApp(hidden="--cache" in sys.argv)
    app.mainloop()
