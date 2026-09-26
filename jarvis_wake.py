"""Wake word: say "Hey Jarvis", then what you want.

How it works:
1. The mic stays open all the time. Every 80 ms block goes through openWakeWord
   (a small neural network running locally; nothing is sent over the Internet).
2. When it's confident enough that it heard "Hey Jarvis", it beeps and records
   what comes next until you stop talking (jarvis_core.record_until_silence).
3. Only that request is sent to Google to be turned into text.

Listening pauses while Jarvis is thinking or talking (is_paused); otherwise he could
wake himself up by hearing his own voice.

Conversation mode: after an answer, the window calls listen_again(). Once Jarvis has
finished talking, a lower beep means he's still listening for FOLLOWUP_SECONDS seconds,
no "Hey Jarvis" needed. If nobody says anything, he goes back to sleep (on_idle).

Tune it to your voice: python jarvis_wake.py calibrer
(measures your scores and saves a threshold that suits you in wake_config.json)
"""
import queue
import sys
import threading
import winsound
from collections import deque
from pathlib import Path
import numpy as np
from jarvis_core import load_json, mic_stream, read_frames, record_until_silence, save_json, transcribe, volume

WAKE_MODEL = "hey_jarvis"  # pre-trained model that ships with openWakeWord
WAKE_THRESHOLD = 0.5       # default minimum confidence (0 to 1); calibration overrides it
CONFIG_FILE = Path(__file__).resolve().parent / "wake_config.json"  # threshold from calibration
FOLLOWUP_SECONDS = 5       # conversation mode: how long to keep listening after an answer (0 = off)


def load_threshold():
    """The threshold saved by calibration, or WAKE_THRESHOLD."""
    return float(load_json(CONFIG_FILE, {}).get("threshold", WAKE_THRESHOLD))


def load_wake_model():
    """Load the "Hey Jarvis" model (downloaded the first time)."""
    from openwakeword.model import Model  # imported here so Jarvis still runs without openwakeword
    from openwakeword.utils import download_models
    download_models([WAKE_MODEL])
    return Model(wakeword_models=[WAKE_MODEL], inference_framework="onnx")


class WakeWordListener:
    def __init__(self, on_wake, on_command, on_error, is_paused=lambda: False, on_idle=lambda: None):
        """
        on_wake(followup)       : listening starts (followup=True in conversation mode)
        on_command(text, err)   : the transcribed request (text), or an error message (err)
        on_error(message)       : the wake word can't work
        is_paused() -> bool     : True when we shouldn't listen (Jarvis busy or talking)
        on_idle()               : conversation mode ended without anyone saying anything
        """
        self.on_wake = on_wake
        self.on_command = on_command
        self.on_error = on_error
        self.is_paused = is_paused
        self.on_idle = on_idle
        self.enabled = True     # on/off switch (the ÉCOUTE button in the window)
        self.followup = False   # True = listen for a follow-up as soon as Jarvis stops talking

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def listen_again(self):
        """Conversation mode: listen for a follow-up after the answer, no "Hey Jarvis" needed."""
        if self.enabled and FOLLOWUP_SECONDS > 0:
            self.followup = True

    def _run(self):
        try:
            model = load_wake_model()
        except ImportError:
            return self.on_error("Réveil vocal indisponible : pip install openwakeword")
        except Exception as e:
            return self.on_error(f"Réveil vocal indisponible : {e}")
        threshold = load_threshold()

        frames = queue.Queue()
        levels = deque(maxlen=40)  # loudness over the last ~3 seconds, to estimate background noise
        paused = False
        try:
            with mic_stream(frames):
                while True:
                    frame = frames.get()
                    if not self.enabled or self.is_paused():
                        # Forget what was heard before the pause, but only once.
                        # reset() is expensive: calling it on every block (12 times a second)
                        # kept a whole CPU core busy while Jarvis was thinking or talking.
                        if not paused:
                            model.reset()
                            paused = True
                        continue
                    paused = False
                    levels.append(volume(frame))
                    noise = float(np.percentile(levels, 20))  # estimated background noise

                    # Conversation mode: Jarvis just finished talking, listen for what comes next
                    if self.followup:
                        self.followup = False
                        self.on_wake(True)
                        winsound.Beep(660, 90)  # lower, shorter beep: "still listening"
                        audio = record_until_silence(read_frames(frames), noise_floor=noise,
                                                     no_speech=FOLLOWUP_SECONDS)
                        if audio is None:
                            self.on_idle()  # nobody spoke: Jarvis goes back to sleep quietly
                        else:
                            self.on_command(*transcribe(audio))
                        continue

                    if model.predict(frame)[WAKE_MODEL] < threshold:
                        continue

                    # "Hey Jarvis" heard
                    model.reset()
                    self.on_wake(False)
                    winsound.Beep(880, 120)  # beep: "I'm listening"
                    audio = record_until_silence(read_frames(frames), noise_floor=noise)
                    self.on_command(*transcribe(audio))
        except Exception as e:
            self.on_error(f"Micro indisponible : {e}")


# --- Calibrating on the user's voice ---

FRAMES_PER_SECOND = 12.5  # 80 ms blocks


def record_scores(model, frames, seconds):
    """Listen for `seconds` seconds: score and loudness of every block, with a live bar."""
    model.reset()
    scores, levels = [], []
    for _ in range(int(seconds * FRAMES_PER_SECOND)):
        frame = frames.get()
        score = float(model.predict(frame)[WAKE_MODEL])
        scores.append(score)
        levels.append(volume(frame))
        bar = "█" * int(score * 30)
        print(f"\r  score {score:4.2f} |{bar:<30}|  volume {levels[-1]:6.0f}   ", end="", flush=True)
    print()
    return scores, levels


def peaks(scores, min_score=0.02, gap=8):
    """Best score of each "Hey Jarvis": runs of blocks above min_score, separated by
    at least `gap` quieter blocks (~0.6 s)."""
    groups, current, quiet = [], [], 0
    for s in scores:
        if s > min_score:
            current.append(s)
            quiet = 0
        elif current:
            quiet += 1
            if quiet >= gap:
                groups.append(max(current))
                current = []
    if current:
        groups.append(max(current))
    return groups


SAFETY_MARGIN = 0.1  # the threshold always stays at least 0.1 above your normal speech


def recommend(false_max, all_peaks, attempts=5):
    """Suggested threshold: below your "Hey Jarvis", well above your normal speech.
    Returns (threshold or None, explanation).

    - We only keep your `attempts` best peaks, and only the ones clearly above your normal
      speech (small noise spikes aren't real attempts; an early version suggested 0.05!).
    - We aim for 70% of the median, so one bad attempt doesn't drag everything down,
      and we never go below "normal speech + safety margin"."""
    floor = false_max + SAFETY_MARGIN
    real = sorted((p for p in all_peaks if p > max(0.1, floor)), reverse=True)[:attempts]
    if not real:
        return None, ("Aucun « Hey Jarvis » reconnu. Essaie de le dire à l'anglaise, « Hé Djar-viss », "
                      "plus près du micro, puis relance le calibrage.")
    threshold = round(min(max(0.7 * float(np.median(real)), floor), 0.9), 2)
    heard = sum(p >= threshold for p in real)
    note = f"Avec ce seuil, {heard} de tes {attempts} « Hey Jarvis » sont reconnus."
    if heard < 3:
        note += (" C'est peu : articule « Hey Jarvis » à l'anglaise (« Hé Djar-viss »), "
                 "rapproche-toi du micro, puis relance le calibrage.")
    return threshold, note


def calibrate(ask=input):
    print("Calibrage de « Hey Jarvis » (Jarvis doit être fermé : il utilise aussi le micro).")
    model = load_wake_model()
    frames = queue.Queue()
    with mic_stream(frames):
        frames.get()  # the mic is ready
        ask("\n1/2 — Appuie sur Entrée puis PARLE NORMALEMENT 10 secondes, sans dire « Jarvis »...")
        normal, levels = record_scores(model, frames, 10)
        ask("\n2/2 — Appuie sur Entrée puis dis « Hey Jarvis » 5 fois, avec 2 secondes entre chaque...")
        said, _ = record_scores(model, frames, 16)

    false_max, true_peaks = max(normal), sorted(peaks(said), reverse=True)[:5]
    print(f"\nVoix ordinaire : score maximum {false_max:.2f}")
    print(f"« Hey Jarvis »  : meilleurs scores {', '.join(f'{p:.2f}' for p in true_peaks) or '—'}")
    if max(levels) < 300:
        print("⚠ Ton micro capte très faiblement : rapproche-toi ou monte son volume dans Windows.")
    threshold, note = recommend(false_max, true_peaks)
    print(note)
    if threshold is None:
        return
    print(f"Seuil proposé : {threshold} (actuel : {load_threshold()})")
    if ask("Enregistrer ce seuil ? (o/n) ").strip().lower().startswith("o"):
        save_json(CONFIG_FILE, {"threshold": threshold})
        print(f"Enregistré dans {CONFIG_FILE.name}. Relance Jarvis pour l'utiliser.")


if __name__ == "__main__":
    if sys.argv[1:] == ["calibrer"]:
        calibrate()
    else:
        print(__doc__)
