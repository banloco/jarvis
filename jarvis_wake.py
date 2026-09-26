"""Réveil vocal : dites « Hey Jarvis », puis votre demande.

Fonctionnement :
1. Le micro reste ouvert en permanence ; chaque bloc de 80 ms est analysé par
   openWakeWord (un petit réseau de neurones local, rien n'est envoyé sur Internet).
2. Quand il reconnaît « Hey Jarvis » avec assez de confiance, on émet un bip
   et on enregistre la suite jusqu'au silence (jarvis_core.record_until_silence).
3. Seule cette demande est envoyée à Google pour être transcrite en texte.

L'écoute est mise en pause pendant que Jarvis réfléchit ou parle (is_paused),
sinon il pourrait se déclencher en s'entendant lui-même.
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


class WakeWordListener:
    def __init__(self, on_wake, on_command, on_error, is_paused=lambda: False):
        """
        on_wake()               : « Hey Jarvis » détecté, l'enregistrement commence
        on_command(texte, err)  : demande transcrite (texte), ou message d'erreur (err)
        on_error(message)       : le réveil vocal ne peut pas fonctionner
        is_paused() -> bool     : True quand il ne faut pas écouter (Jarvis occupé / parle)
        """
        self.on_wake = on_wake
        self.on_command = on_command
        self.on_error = on_error
        self.is_paused = is_paused
        self.enabled = True  # interrupteur on/off (bouton 👂 de l'interface)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

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
                    if model.predict(frame)[WAKE_MODEL] < WAKE_THRESHOLD:
                        continue

                    # « Hey Jarvis » détecté
                    model.reset()
                    self.on_wake()
                    winsound.Beep(880, 120)  # bip : « je t'écoute »
                    audio = record_until_silence(read_frames(frames),
                                                 noise_floor=float(np.percentile(levels, 20)))
                    self.on_command(*transcribe(audio))
        except Exception as e:
            self.on_error(f"Micro indisponible : {e}")
