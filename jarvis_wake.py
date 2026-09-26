"""Réveil vocal : dites « Hey Jarvis », puis votre demande.

Fonctionnement :
1. Le micro reste ouvert en permanence ; chaque bloc de 80 ms est analysé par
   openWakeWord (un petit réseau de neurones local, rien n'est envoyé sur Internet).
2. Quand il reconnaît « Hey Jarvis » avec assez de confiance, on émet un bip
   et on enregistre la suite jusqu'au silence (jarvis_core.record_until_silence).
3. Seule cette demande est envoyée à Google pour être transcrite en texte.

L'écoute est mise en pause pendant que Jarvis réfléchit ou parle (is_paused),
sinon il pourrait se déclencher en s'entendant lui-même.

Mode conversation : après une réponse, l'interface appelle listen_again(). Dès que Jarvis
a fini de parler, un bip grave signale qu'il écoute encore FOLLOWUP_SECONDS secondes,
sans « Hey Jarvis ». Si personne ne parle, il se rendort (on_idle).
"""
import queue
import threading
import winsound
from collections import deque
import numpy as np
from jarvis_core import mic_stream, read_frames, record_until_silence, transcribe, volume

WAKE_MODEL = "hey_jarvis"  # modèle pré-entraîné fourni par openWakeWord
WAKE_THRESHOLD = 0.5       # confiance minimale (0 à 1) : baisse-le s'il ne t'entend pas,
                           # monte-le s'il se déclenche tout seul
FOLLOWUP_SECONDS = 5       # mode conversation : durée d'écoute après une réponse (0 = désactivé)


class WakeWordListener:
    def __init__(self, on_wake, on_command, on_error, is_paused=lambda: False, on_idle=lambda: None):
        """
        on_wake(suite)          : l'écoute commence (suite=True en mode conversation)
        on_command(texte, err)  : demande transcrite (texte), ou message d'erreur (err)
        on_error(message)       : le réveil vocal ne peut pas fonctionner
        is_paused() -> bool     : True quand il ne faut pas écouter (Jarvis occupé / parle)
        on_idle()               : mode conversation terminé sans que personne ne parle
        """
        self.on_wake = on_wake
        self.on_command = on_command
        self.on_error = on_error
        self.is_paused = is_paused
        self.on_idle = on_idle
        self.enabled = True     # interrupteur on/off (bouton ÉCOUTE de l'interface)
        self.followup = False   # True = écouter une suite dès que Jarvis a fini de parler

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def listen_again(self):
        """Mode conversation : écouter une suite, sans « Hey Jarvis », après la réponse."""
        if self.enabled and FOLLOWUP_SECONDS > 0:
            self.followup = True

    def _run(self):
        try:
            # Import ici : si openwakeword n'est pas installé, le reste de Jarvis marche quand même
            from openwakeword.model import Model
            from openwakeword.utils import download_models
            download_models([WAKE_MODEL])  # ne télécharge que la 1re fois
            model = Model(wakeword_models=[WAKE_MODEL], inference_framework="onnx")
        except ImportError:
            return self.on_error("Réveil vocal indisponible : pip install openwakeword")
        except Exception as e:
            return self.on_error(f"Réveil vocal indisponible : {e}")

        frames = queue.Queue()
        levels = deque(maxlen=40)  # volume des ~3 dernières secondes, pour estimer le bruit ambiant
        paused = False
        try:
            with mic_stream(frames):
                while True:
                    frame = frames.get()
                    if not self.enabled or self.is_paused():
                        # Oublie ce qui a été entendu avant la pause, UNE seule fois :
                        # reset() est coûteux, et l'appeler à chaque bloc (12 fois par seconde)
                        # occupait un cœur entier pendant que Jarvis réfléchissait ou parlait.
                        if not paused:
                            model.reset()
                            paused = True
                        continue
                    paused = False
                    levels.append(volume(frame))
                    noise = float(np.percentile(levels, 20))  # bruit ambiant estimé

                    # Mode conversation : Jarvis vient de finir de parler, on écoute la suite
                    if self.followup:
                        self.followup = False
                        self.on_wake(True)
                        winsound.Beep(660, 90)  # bip plus grave et court : « je t'écoute encore »
                        audio = record_until_silence(read_frames(frames), noise_floor=noise,
                                                     no_speech=FOLLOWUP_SECONDS)
                        if audio is None:
                            self.on_idle()  # personne n'a parlé : Jarvis se rendort, sans message
                        else:
                            self.on_command(*transcribe(audio))
                        continue

                    if model.predict(frame)[WAKE_MODEL] < WAKE_THRESHOLD:
                        continue

                    # « Hey Jarvis » détecté
                    model.reset()
                    self.on_wake(False)
                    winsound.Beep(880, 120)  # bip : « je t'écoute »
                    audio = record_until_silence(read_frames(frames), noise_floor=noise)
                    self.on_command(*transcribe(audio))
        except Exception as e:
            self.on_error(f"Micro indisponible : {e}")
