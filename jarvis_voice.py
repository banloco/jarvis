"""Jarvis's voice.

Main voice: Microsoft's neural voices through edge-tts (natural, free, needs Internet).
Backup voice: Hortense (pyttsx3, offline), used automatically whenever edge-tts fails.

So that Jarvis can start talking before his answer is finished, the text is cut into
sentences (SentenceStreamer) that go through two stages working side by side:

    sentences --> [synthesis: text -> mp3 file] --> [mp3 playback] --> speakers

While sentence 1 is playing, sentence 2 is already being synthesized.
To list the available voices: python -m edge_tts --list-voices (French ones start with fr-FR-)
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

TTS_VOICE = "fr-FR-HenriNeural"   # another male voice: "fr-FR-RemyMultilingualNeural"
TTS_RATE = "+5%"                  # speed: "-10%" is slower, "+20%" faster
FALLBACK_VOICE_ID = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_FR-FR_HORTENSE_11.0"
MIN_SENTENCE = 15                 # anything shorter gets merged with the next sentence

_mci = ctypes.windll.winmm.mciSendStringW  # the audio player built into Windows (nothing to install)


def clean_for_speech(text):
    """Strip what can't be read aloud: formatting symbols, web addresses."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*#`_>|\[\]←↑→↓↖↗↘↙]", "", text)
    text = text.replace("°C", " degrés").replace("km/h", " kilomètres heure")
    return re.sub(r"\s+", " ", text).strip()


class Speaker:
    """A queue of sentences to say, played in order."""

    def __init__(self):
        self.texts = queue.Queue()   # (generation, text) waiting to be synthesized
        self.audio = queue.Queue()   # (generation, mp3 file or None, text) ready to play
        self.pending = 0             # sentences not fully spoken yet
        self.done = threading.Condition()
        self.generation = 0          # bumped on every stop(): older sentences get dropped
        self.muted = False
        threading.Thread(target=self._synthesize_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    # --- Using it ---

    def say(self, text, wait=False):
        """Queue a sentence. wait=True blocks until everything has been said."""
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
        """True while a sentence is queued, being synthesized or playing."""
        return self.pending > 0

    def stop(self):
        """Stop talking right now and forget the queued sentences."""
        self.generation += 1

    # --- Stage 1: text -> mp3 ---

    def _synthesize_loop(self):
        loop = asyncio.new_event_loop()  # edge-tts is async, so it needs its own event loop
        while True:
            generation, text = self.texts.get()
            if generation != self.generation:  # stopped in the meantime
                self._finished()
                continue
            fd, path = tempfile.mkstemp(suffix=".mp3", prefix="jarvis_")
            os.close(fd)
            try:
                loop.run_until_complete(edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE).save(path))
                self.audio.put((generation, path, text))
            except Exception:  # no Internet, service down... -> backup voice
                os.remove(path)
                self.audio.put((generation, None, text))

    # --- Stage 2: playback ---

    def _play_loop(self):
        fallback = None  # offline engine, only created if we ever need it
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
                pass  # a playback problem must never hold up the next sentences
            finally:
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                self._finished()

    def _play_mp3(self, path, generation):
        """Play the file, and cut it short if stop() is called meanwhile."""
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
            import pythoncom  # the Windows voice (SAPI5) needs COM initialized in this thread
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
    """Takes the answer piece by piece (streaming) and hands each sentence to the voice
    as soon as it's complete. Call flush() once the answer is over."""

    SENTENCE_END = re.compile(r"[.!?…]+\s+|\n+")

    def __init__(self, speaker):
        self.speaker = speaker
        self.buffer = ""

    def feed(self, piece):
        self.buffer += piece
        start = 0
        for match in self.SENTENCE_END.finditer(self.buffer):
            sentence = self.buffer[start:match.end()]
            if len(sentence.strip()) >= MIN_SENTENCE:  # too short: wait for more
                self.speaker.say(sentence)
                start = match.end()
        self.buffer = self.buffer[start:]

    def flush(self):
        if self.buffer.strip():
            self.speaker.say(self.buffer)
        self.buffer = ""
