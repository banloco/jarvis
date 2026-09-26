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

Calibrage sur ta voix : python jarvis_wake.py calibrer
(mesure tes scores et enregistre le seuil adapté dans wake_config.json)
"""
import queue
import sys
import threading
import winsound
from collections import deque
from pathlib import Path
import numpy as np
from jarvis_core import load_json, mic_stream, read_frames, record_until_silence, save_json, transcribe, volume

WAKE_MODEL = "hey_jarvis"  # modèle pré-entraîné fourni par openWakeWord
WAKE_THRESHOLD = 0.5       # confiance minimale (0 à 1) par défaut, remplacée par le calibrage
CONFIG_FILE = Path(__file__).resolve().parent / "wake_config.json"  # seuil issu du calibrage
FOLLOWUP_SECONDS = 5       # mode conversation : durée d'écoute après une réponse (0 = désactivé)


def load_threshold():
    """Seuil enregistré par le calibrage, sinon WAKE_THRESHOLD."""
    return float(load_json(CONFIG_FILE, {}).get("threshold", WAKE_THRESHOLD))


def load_wake_model():
    """Charge le modèle « Hey Jarvis » (téléchargé la 1re fois)."""
    from openwakeword.model import Model  # import ici : Jarvis marche aussi sans openwakeword
    from openwakeword.utils import download_models
    download_models([WAKE_MODEL])
    return Model(wakeword_models=[WAKE_MODEL], inference_framework="onnx")


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
            model = load_wake_model()
        except ImportError:
            return self.on_error("Réveil vocal indisponible : pip install openwakeword")
        except Exception as e:
            return self.on_error(f"Réveil vocal indisponible : {e}")
        threshold = load_threshold()

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

                    if model.predict(frame)[WAKE_MODEL] < threshold:
                        continue

                    # « Hey Jarvis » détecté
                    model.reset()
                    self.on_wake(False)
                    winsound.Beep(880, 120)  # bip : « je t'écoute »
                    audio = record_until_silence(read_frames(frames), noise_floor=noise)
                    self.on_command(*transcribe(audio))
        except Exception as e:
            self.on_error(f"Micro indisponible : {e}")


# --- Calibrage sur la voix de l'utilisateur ---

FRAMES_PER_SECOND = 12.5  # blocs de 80 ms


def record_scores(model, frames, seconds):
    """Écoute `seconds` secondes : score de chaque bloc + volume, avec une barre en direct."""
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
    """Meilleur score de chaque « Hey Jarvis » : groupes de blocs au-dessus de min_score,
    séparés par au moins `gap` blocs (~0,6 s) plus bas."""
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


SAFETY_MARGIN = 0.1  # le seuil reste toujours au moins 0,1 au-dessus de ta voix ordinaire


def recommend(false_max, all_peaks, attempts=5):
    """Seuil proposé : sous tes « Hey Jarvis », nettement au-dessus de ta voix ordinaire.
    Retourne (seuil ou None, explication).

    - On ne garde que tes `attempts` meilleurs pics, et seulement ceux qui dépassent
      nettement ta voix ordinaire (les petits pics de bruit ne sont pas des tentatives).
    - On vise 70 % de la MÉDIANE (un essai raté ne fait pas tout baisser),
      sans jamais descendre sous « voix ordinaire + marge de sécurité »."""
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
        frames.get()  # le micro est prêt
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
