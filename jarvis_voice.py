"""Voix de Jarvis.

Voix principale : voix neuronale Microsoft via edge-tts (naturelle, gratuite, Internet requis).
Voix de secours : Hortense (pyttsx3, hors ligne), utilisée automatiquement si edge-tts échoue.

Pour que Jarvis commence à parler sans attendre la fin de sa réponse, le texte est découpé
en phrases (SentenceStreamer) et passe par deux étages qui travaillent en parallèle :

    phrases --> [synthèse : texte -> fichier mp3] --> [lecture du mp3] --> haut-parleurs

Pendant que la phrase 1 est lue, la phrase 2 est déjà en cours de synthèse.
Voix disponibles : python -m edge_tts --list-voices (françaises : fr-FR-...)
"""
import asyncio
import ctypes
import os
import queue
import re
import tempfile
import threading
import time
import edge_tts

TTS_VOICE = "fr-FR-HenriNeural"   # autre voix masculine : "fr-FR-RemyMultilingualNeural"
TTS_RATE = "+5%"                  # vitesse : "-10%" plus lent, "+20%" plus rapide
FALLBACK_VOICE_ID = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_FR-FR_HORTENSE_11.0"
MIN_SENTENCE = 15                 # une « phrase » plus courte est regroupée avec la suivante

_mci = ctypes.windll.winmm.mciSendStringW  # lecteur audio intégré à Windows (aucune installation)


def clean_for_speech(text):
    """Retire ce qui ne se prononce pas : symboles de mise en forme, adresses web."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*#`_>|\[\]]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class Speaker:
    """File d'attente de phrases à prononcer, jouées dans l'ordre."""

    def __init__(self):
        self.texts = queue.Queue()   # (génération, texte) en attente de synthèse
        self.audio = queue.Queue()   # (génération, fichier mp3 ou None, texte) prêts à jouer
        self.pending = 0             # phrases pas encore entièrement prononcées
        self.done = threading.Condition()
        self.generation = 0          # augmente à chaque stop() : les anciennes phrases sont jetées
        self.muted = False
        threading.Thread(target=self._synthesize_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    # --- Utilisation ---

    def say(self, text, wait=False):
        """Ajoute une phrase à dire. wait=True bloque jusqu'à ce que TOUT soit prononcé."""
        text = clean_for_speech(text)
        if self.muted or not text:
            return
        with self.done:
            self.pending += 1
        self.texts.put((self.generation, text))
        if wait:
            self.wait()

    def wait(self):
        with self.done:
            self.done.wait_for(lambda: self.pending == 0)

    def is_speaking(self):
        """True tant qu'une phrase est en attente, en synthèse ou en cours de lecture."""
        return self.pending > 0

    def stop(self):
        """Coupe la parole immédiatement et oublie les phrases en attente."""
        self.generation += 1

    # --- Étage 1 : texte -> mp3 ---

    def _synthesize_loop(self):
        loop = asyncio.new_event_loop()  # edge-tts est asynchrone : il lui faut une boucle à lui
        while True:
            generation, text = self.texts.get()
            if generation != self.generation:  # arrêté entre-temps
                self._finished()
                continue
            fd, path = tempfile.mkstemp(suffix=".mp3", prefix="jarvis_")
            os.close(fd)
            try:
                loop.run_until_complete(edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE).save(path))
                self.audio.put((generation, path, text))
            except Exception:  # pas d'Internet, service indisponible... -> voix de secours
                os.remove(path)
                self.audio.put((generation, None, text))

    # --- Étage 2 : lecture ---

    def _play_loop(self):
        fallback = None  # moteur hors ligne, créé seulement si nécessaire
        while True:
            generation, path, text = self.audio.get()
            try:
                if generation == self.generation and not self.muted:
                    if path:
                        self._play_mp3(path, generation)
                    else:
                        fallback = fallback or self._fallback_engine()
                        fallback.say(text)
                        fallback.runAndWait()
            except Exception:
                pass  # un problème de lecture ne doit jamais bloquer les phrases suivantes
            finally:
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                self._finished()

    def _play_mp3(self, path, generation):
        """Lit le fichier ; s'interrompt si stop() est appelé pendant la lecture."""
        _mci(f'open "{path}" type mpegvideo alias jarvis_voice', None, 0, 0)
        try:
            _mci("play jarvis_voice", None, 0, 0)
            status = ctypes.create_unicode_buffer(32)
            while generation == self.generation and not self.muted:
                _mci("status jarvis_voice mode", status, 32, 0)
                if status.value != "playing":
                    break
                time.sleep(0.05)
        finally:
            _mci("close jarvis_voice", None, 0, 0)

    @staticmethod
    def _fallback_engine():
        try:
            import pythoncom  # la voix Windows (SAPI5) exige COM initialisé dans ce thread
            pythoncom.CoInitialize()
        except ImportError:
            pass
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        try:
            engine.setProperty("voice", FALLBACK_VOICE_ID)
        except Exception:
            pass
        return engine

    def _finished(self):
        with self.done:
            self.pending -= 1
            self.done.notify_all()


class SentenceStreamer:
    """Reçoit la réponse morceau par morceau (streaming) et envoie chaque phrase
    à la voix dès qu'elle est complète. Appeler flush() à la fin de la réponse."""

    SENTENCE_END = re.compile(r"[.!?…]+\s+|\n+")

    def __init__(self, speaker):
        self.speaker = speaker
        self.buffer = ""

    def feed(self, piece):
        self.buffer += piece
        start = 0
        for match in self.SENTENCE_END.finditer(self.buffer):
            sentence = self.buffer[start:match.end()]
            if len(sentence.strip()) >= MIN_SENTENCE:  # trop court : on attend la suite
                self.speaker.say(sentence)
                start = match.end()
        self.buffer = self.buffer[start:]

    def flush(self):
        if self.buffer.strip():
            self.speaker.say(self.buffer)
        self.buffer = ""
