"""Hand gesture control through the webcam.

Two kinds of gestures:

1. POSES (fist, open hand, thumbs up...): learned from YOUR examples.
   - MediaPipe (a model from Google, already trained) finds 21 points on the hand:
     the wrist, the knuckles and the fingertips.
   - A small neural network, trained on your examples, recognizes the pose from where
     those points are and how the fingers are bent.

2. MOVES (no training needed, worked out from how the hand moves):
   - swipe left / right / up / down;
   - pinch thumb and index, then move your hand up or down: changes the volume smoothly.

Command line (from the Jarvis folder):
    python jarvis_gestures.py collecter poing      # record examples of the "poing" (fist) gesture
    python jarvis_gestures.py lister               # how many examples per gesture
    python jarvis_gestures.py supprimer poing      # delete a gesture's examples
    python jarvis_gestures.py entrainer            # train the model on all the examples
    python jarvis_gestures.py tester               # live recognition, to check how it does
    python jarvis_gestures.py importer             # add thousands of public examples (HaGRID)

HaGRID is a public dataset of gestures filmed by thousands of people (CC BY-SA 4.0 license,
https://github.com/hukenovs/hagrid). "importer" pulls out fist, open hand, thumbs, peace,
rock and "nothing". Then "entrainer --hagrid" adds the gestures you haven't recorded yourself.
Your own examples are more reliable, which is why --hagrid is off by default.

Tips for good examples:
- "rien" (nothing) is essential and needs VARIETY: relaxed hand, side view, half closed,
  moving, typing on the keyboard... Anything that should NOT trigger an action.
- For each gesture, vary the distance and the angle, and switch between left and right hand.
- After training, the list of mix-ups tells you which gesture to record again.

In the window, the GESTES button turns watching on (see GESTURE_ACTIONS below).
"""
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

BASE_DIR = Path(__file__).resolve().parent
HAND_MODEL = BASE_DIR / "models" / "hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
DATA_FILE = BASE_DIR / "gestures_data.npz"          # your examples (hand points + gesture name)
GESTURE_MODEL = BASE_DIR / "gestures_model.joblib"  # the trained model
MODEL_VERSION = 3   # bump this when the feature computation changes (forces retraining)
DATA_FORMAT = 2     # format of the saved examples: 21 points (x, y), true image proportions

# Public HaGRID data (thousands of people): see the "importer" command
HAGRID_ZIP = BASE_DIR / "data" / "hagrid" / "annotations.zip"
HAGRID_FILE = BASE_DIR / "data" / "hagrid" / "hagrid_gestures.npz"
HAGRID_URL = ("https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/"
              "annotations_with_landmarks/annotations.zip")
HAGRID_LABELS = {  # HaGRID name -> Jarvis gesture name
    "fist": "poing", "palm": "main_ouverte", "like": "pouce_haut", "dislike": "pouce_bas",
    "peace": "victoire", "rock": "rock", "no_gesture": "rien",
}
HAGRID_PER_GESTURE = 3000  # examples imported per gesture
HAGRID_ASPECT = 4 / 3      # assumed width / height of the HaGRID photos (not provided;
                           # we tried 9:16, 1:1, 4:3 and 16:9: 4:3 was slightly better, all within 2%)

# Poses
NEUTRAL = "rien"           # the gesture that never triggers anything
SAMPLES_PER_GESTURE = 300  # examples recorded by default by "collecter"
MIN_SAMPLES = 50           # minimum per gesture before training
COUNTDOWN = 3              # seconds to get your hand in place before recording
AUGMENT_COPIES = 3         # slightly distorted copies of each example added for training
CONFIDENCE = 0.85          # minimum certainty (0 to 1) to accept a gesture
VOTE_FRAMES = 5            # frames we average the decision over (~0.3 s)
HOLD_SECONDS = 0.5         # how long the gesture has to be held
COOLDOWN = 2.0             # minimum time between two pose triggers

# Moves (distances as a fraction of the image width / height)
SWIPE_DISTANCE = 0.20      # minimum travel for a swipe (20% of the image)
SWIPE_WINDOW = 0.4         # ...done in under 0.4 s
SWIPE_COOLDOWN = 1.0       # time between two swipes
HAND_SETTLE = 0.5          # a hand that just appeared doesn't count yet (avoids fake swipes)
STILL_DISTANCE = 0.05      # below this, the hand counts as still (a pose is possible)
PINCH_RATIO = 0.35         # thumb and index closer than 35% of the palm = pinch
PINCH_HOLD = 0.3           # how long to pinch before volume mode kicks in
PINCH_STEP = 0.03          # vertical travel for one volume step (2%)

WATCH_FPS = 15             # frames analyzed per second while watching (easier on the CPU)

# MediaPipe's 21 points: 0 = wrist, then 4 points per finger (base -> tip)
WRIST = 0
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
TIPS = [4, 8, 12, 16, 20]
PALM = [0, 5, 9, 13, 17]   # wrist + finger bases: the middle of the palm


# --- Step 1: find the hand ---

class HandTracker:
    """Opens the webcam and gives back, for each frame, the 21 points of the hand (or None)."""

    def __init__(self):
        if not HAND_MODEL.exists():  # downloaded the first time (7.5 MB)
            HAND_MODEL.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,  # follows the hand from one frame to the next
            num_hands=1)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW opens much faster on Windows
        self.start = time.monotonic()

    def read(self):
        """Returns (image, hand): the image is mirrored (like a selfie),
        hand = (points, image width/height) or None if no hand is visible."""
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError("Webcam indisponible (utilisée par une autre application ?)")
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.monotonic() - self.start) * 1000)
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                  timestamp_ms)
        if not result.hand_landmarks:
            return frame, None
        h, w = frame.shape[:2]
        return frame, (result.hand_landmarks[0], w / h)

    def close(self):
        self.camera.release()
        self.landmarker.close()


# --- Step 2: describe the hand with numbers ---

def hand_vector(points, aspect):
    """21 points -> 42 numbers (x, y of each point). This is what gets saved.

    - MediaPipe coordinates are fractions of the image width and height, so we multiply x
      by width/height to get the hand's real proportions back (otherwise, at 640x480,
      a hand looks narrower than it is).
    - Depth (z) is dropped: it isn't reliable, and HaGRID doesn't have it anyway.
    """
    coords = np.array([[p.x * aspect, p.y] for p in points], dtype=np.float32)
    return (coords - coords[WRIST]).flatten()  # origin = the wrist


def canonical(P):
    """Mirror the "backwards" hands so they all face the same way (like a right hand seen
    from the front). Which way it faces comes from the shape of the hand: is the index to
    the left or the right of the pinky, seen from the wrist? That turned out more reliable
    than MediaPipe's left/right label, which is sometimes wrong."""
    index, pinky = P[:, 5] - P[:, WRIST], P[:, 17] - P[:, WRIST]
    flip = index[:, 0] * pinky[:, 1] - index[:, 1] * pinky[:, 0] < 0
    P = P.copy()
    P[flip, :, 0] *= -1
    return P


def enrich(vectors):
    """Turn examples (42 numbers per row) into features for the model.

    1. Every hand faces the same way (see canonical).
    2. Everything is scaled by the SIZE OF THE PALM (wrist -> base of the middle finger),
       which doesn't change when you bend your fingers. (Dividing by the largest distance,
       like we did at first, erased the difference between a fist and an open hand.)
    3. Extra measurements that describe the shape of the hand directly:
       - the angle at each joint (straight finger ≈ flat, bent finger ≈ sharp): 15 values
       - the distance from each fingertip to the wrist: 5 values
       - the distance from the thumb to the other fingertips (pinch, OK sign...): 4 values
       - the spread between neighboring fingers: 3 values
    """
    P = canonical(np.asarray(vectors, dtype=np.float32).reshape(-1, 21, 2))
    palm = np.linalg.norm(P[:, 9], axis=1)
    P = P / np.maximum(palm, 1e-6)[:, None, None]

    angles = []
    for finger in FINGERS:
        chain = [WRIST] + finger
        for a, b, c in zip(chain, chain[1:], chain[2:]):  # angle at point b between a-b and b-c
            u, v = P[:, a] - P[:, b], P[:, c] - P[:, b]
            cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-6)
            angles.append(cos)
    tip_wrist = [np.linalg.norm(P[:, t], axis=1) for t in TIPS]
    thumb_tips = [np.linalg.norm(P[:, 4] - P[:, t], axis=1) for t in TIPS[1:]]
    spread = [np.linalg.norm(P[:, a] - P[:, b], axis=1) for a, b in zip(TIPS[1:], TIPS[2:])]
    return np.hstack([P.reshape(len(P), -1), np.stack(angles + tip_wrist + thumb_tips + spread, axis=1)])


def augment(vectors, copies, rng):
    """Make slightly distorted copies of the examples: hand rotated by up to ±20°, a bit
    stretched, with a little jitter. That way the model learns to recognize a gesture
    even when you do it a bit differently from your examples."""
    P = np.repeat(np.asarray(vectors, dtype=np.float32).reshape(-1, 21, 2), copies, axis=0)
    theta = rng.uniform(-np.pi / 9, np.pi / 9, len(P))            # rotation within the image plane
    cos, sin = np.cos(theta), np.sin(theta)
    x, y = P[:, :, 0].copy(), P[:, :, 1].copy()
    P[:, :, 0], P[:, :, 1] = cos[:, None] * x - sin[:, None] * y, sin[:, None] * x + cos[:, None] * y
    P *= 1 + rng.normal(0, 0.05, (len(P), 1, 2))                   # slight stretch along each axis
    scale = np.abs(P).max(axis=(1, 2), keepdims=True)
    P += rng.normal(0, 0.01, P.shape) * scale                      # jitter
    return P.reshape(len(P), -1)


# --- Display ---

def draw_hand(frame, points):
    """Draw the points and bones of the hand on the image."""
    h, w = frame.shape[:2]
    xy = [(int(p.x * w), int(p.y * h)) for p in points]
    for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, xy[c.start], xy[c.end], (255, 191, 0), 2)
    for x, y in xy:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)


def draw_text(frame, text, line=0, color=(255, 191, 0)):
    cv2.putText(frame, text, (10, 30 + 30 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# --- Training data ---

def load_data():
    """Returns (X, y): your examples, one per row of X (42 numbers), gesture names in y."""
    if not DATA_FILE.exists():
        return np.empty((0, 42), dtype=np.float32), np.empty(0, dtype=str)
    data = np.load(DATA_FILE)
    X, y = data["X"], data["y"]
    if ("format" not in data.files) or int(data["format"]) < DATA_FORMAT:
        # Old format (x, y, z as fractions of a 640x480 image): keep x, y and fix the
        # proportions. A copy of the old file is kept, just in case.
        backup = DATA_FILE.with_name("gestures_data.v1.npz")
        if not backup.exists():
            backup.write_bytes(DATA_FILE.read_bytes())
        X = X.reshape(-1, 21, 3)[:, :, :2] * np.array([640 / 480, 1], dtype=np.float32)
        X = X.reshape(len(X), -1)
        save_data(X, y)
    return X, y


def save_data(X, y):
    np.savez_compressed(DATA_FILE, X=X, y=y, format=DATA_FORMAT)


def import_hagrid(per_gesture=HAGRID_PER_GESTURE):
    """Pull the hand points of the gestures we care about (HAGRID_LABELS) out of HaGRID
    (public annotations, ~686 MB), pick `per_gesture` at random for each gesture, and save
    them to HAGRID_FILE. We don't need the photos themselves."""
    import json
    import zipfile
    if not HAGRID_ZIP.exists():
        HAGRID_ZIP.parent.mkdir(parents=True, exist_ok=True)
        print(f"Téléchargement de HaGRID (686 Mo) vers {HAGRID_ZIP}...")
        urllib.request.urlretrieve(HAGRID_URL, HAGRID_ZIP)

    by_label = {name: [] for name in HAGRID_LABELS.values()}
    with zipfile.ZipFile(HAGRID_ZIP) as archive:
        for path in archive.namelist():
            # "test" is held back by the HaGRID authors; .ipynb_checkpoints are stray
            # copies left in the archive
            if not path.endswith(".json") or "/test/" in path or ".ipynb_checkpoints" in path:
                continue
            print(f"  lecture de {path}...", flush=True)
            for entry in json.loads(archive.read(path)).values():
                # A photo can have several hands, and "no_gesture" can show up in any file
                # (often the person's other hand)
                for hand_label, points in zip(entry["labels"], entry.get("hand_landmarks", [])):
                    if hand_label in HAGRID_LABELS and len(points) == 21:
                        by_label[HAGRID_LABELS[hand_label]].append(points)

    rng = np.random.default_rng(0)
    X, y = [], []
    for name, hands in by_label.items():
        picked = rng.choice(len(hands), min(per_gesture, len(hands)), replace=False)
        P = np.array(hands, dtype=np.float32)[picked]
        X.append((P - P[:, :1]).reshape(len(P), -1))  # origin = wrist; x isn't corrected yet
        y += [name] * len(P)
        print(f"  {name:13} {len(hands):7} mains disponibles, {len(P)} gardées")
    np.savez_compressed(HAGRID_FILE, X=np.vstack(X), y=np.array(y))
    print(f"Enregistré : {HAGRID_FILE}")


def load_hagrid(aspect=HAGRID_ASPECT):
    """The imported HaGRID examples (empty if "importer" hasn't been run).
    x is multiplied by `aspect` (photo width / height) to get the real proportions."""
    if not HAGRID_FILE.exists():
        return np.empty((0, 42), dtype=np.float32), np.empty(0, dtype=str)
    data = np.load(HAGRID_FILE)
    X = data["X"].reshape(-1, 21, 2) * np.array([aspect, 1], dtype=np.float32)
    return X.reshape(len(X), -1), data["y"]


def collect(name, count=SAMPLES_PER_GESTURE):
    """Open the webcam and record `count` examples of the gesture `name`.
    SPACE starts a countdown and then recording; SPACE again pauses."""
    tracker = HandTracker()
    samples, recording, countdown_end = [], False, None
    print(f"Geste « {name} » : ESPACE = démarrer/pause, Q = quitter.")
    print("Pendant l'enregistrement, bouge un peu la main : distance, angle, main gauche/droite.")
    try:
        while len(samples) < count:
            frame, hand = tracker.read()
            if countdown_end and time.monotonic() >= countdown_end:
                recording, countdown_end = True, None
            if hand:
                draw_hand(frame, hand[0])
                if recording:
                    samples.append(hand_vector(*hand))

            if countdown_end:
                status = f"prépare le geste... {int(countdown_end - time.monotonic()) + 1}"
            else:
                status = "ENREGISTREMENT" if recording else "en pause (ESPACE pour commencer)"
            draw_text(frame, f"{name} : {len(samples)}/{count} - {status}",
                      color=(0, 0, 255) if recording else (255, 191, 0))
            if not hand:
                draw_text(frame, "Aucune main visible", 1, (0, 165, 255))
            cv2.imshow("Jarvis - collecte", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if recording or countdown_end:
                    recording, countdown_end = False, None
                else:
                    countdown_end = time.monotonic() + COUNTDOWN
            elif key in (ord("q"), 27):  # Q or Esc
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()

    if samples:
        X, y = load_data()
        save_data(np.vstack([X, samples]), np.concatenate([y, [name] * len(samples)]))
        print(f"{len(samples)} exemples de « {name} » ajoutés.")
    show_counts()


def show_counts():
    _, y = load_data()
    if not len(y):
        print("Aucun exemple enregistré.")
        return
    names, counts = np.unique(y, return_counts=True)
    for n, c in zip(names, counts):
        action = GESTURE_ACTIONS.get(n, (None, "aucune action"))[1]
        print(f"  {n:15} {c:5} exemples   -> {action}")
    if NEUTRAL not in names:
        print(f"  ⚠ Pense à collecter le geste « {NEUTRAL} » (main détendue / quelconque).")


def delete(name):
    X, y = load_data()
    keep = y != name
    save_data(X[keep], y[keep])
    print(f"{(~keep).sum()} exemples de « {name} » supprimés.")


# --- Step 3: train and use the model ---

def new_model():
    """Value scaling + a small neural network (2 hidden layers)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
                                       early_stopping=True, random_state=0))


def fit(X, y, rng, extra=None):
    """Train a model on your examples + their distorted copies, plus any extra
    examples `extra` = (X, y) (HaGRID, not distorted)."""
    Xa = np.vstack([X, augment(X, AUGMENT_COPIES, rng)])
    ya = np.concatenate([y, np.repeat(y, AUGMENT_COPIES)])
    if extra is not None and len(extra[1]):
        Xa, ya = np.vstack([Xa, extra[0]]), np.concatenate([ya, extra[1]])
    return new_model().fit(enrich(Xa), ya)


def train(use_hagrid=False):
    """Estimate the accuracy ON YOUR EXAMPLES, show the mix-ups, then train and save
    the final model.

    use_hagrid: add the HaGRID examples of the gestures you have NOT recorded
    (thumbs, peace, rock...), so they're recognized without collecting them.
    Measured on real data (Sept 2026): your examples alone = 74%; with HaGRID for the
    other gestures = 68% on your own gestures. That's why it's off by default.
    (Mixing HaGRID into your own gestures didn't help either: 72% at best, and HaGRID's
    "nothing" doesn't look like yours.)"""
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold

    X, y = load_data()
    extra = None
    if use_hagrid:
        H, hy = load_hagrid()
        if not len(hy):
            return print("Aucune donnée HaGRID : lance d'abord « importer ».")
        missing = ~np.isin(hy, np.unique(y))  # only the gestures you haven't recorded
        extra = (H[missing], hy[missing])
        print(f"+ HaGRID pour : {', '.join(sorted(set(extra[1])))} ({missing.sum()} exemples)")
    names, counts = np.unique(y, return_counts=True)
    if len(names) < 2:
        return print("Il faut au moins 2 gestes (dont « rien ») pour entraîner.")
    if counts.min() < MIN_SAMPLES:
        return print(f"Chaque geste doit avoir au moins {MIN_SAMPLES} exemples : lance « lister ».")
    rng = np.random.default_rng(0)

    # Cross-validation: 5 times over, train on 80% of your examples (+ HaGRID) and test on
    # the other 20%. Since examples are recorded in a row, each test covers a stretch of
    # time the model hasn't seen, which keeps the estimate honest.
    predicted = np.empty_like(y, dtype=object)
    for train_idx, test_idx in StratifiedKFold(n_splits=5).split(X, y):
        model = fit(X[train_idx], y[train_idx], rng, extra)
        predicted[test_idx] = model.predict(enrich(X[test_idx]))
    predicted = predicted.astype(str)
    print(classification_report(y, predicted, digits=3, zero_division=0))

    # The most common mix-ups tell you which gesture to record again or make more distinct
    matrix = confusion_matrix(y, predicted, labels=names)
    confusions = [(matrix[i, j] / matrix[i].sum(), names[i], names[j])
                  for i in range(len(names)) for j in range(len(names)) if i != j and matrix[i, j]]
    for rate, real, guess in sorted(confusions, reverse=True)[:3]:
        if rate >= 0.05:
            print(f"  ⚠ {rate:.0%} des « {real} » pris pour « {guess} »")

    model = fit(X, y, rng, extra)  # final model, on every example
    joblib.dump({"version": MODEL_VERSION, "model": model}, GESTURE_MODEL)
    print(f"Modèle enregistré : {GESTURE_MODEL.name}")


def load_model():
    """The trained model, or None if there isn't one or it's from an older version."""
    if not GESTURE_MODEL.exists():
        return None
    saved = joblib.load(GESTURE_MODEL)
    if not isinstance(saved, dict) or saved.get("version") != MODEL_VERSION:
        return None  # old format: "entrainer" needs to be run again
    return saved["model"]


# --- Step 4: live recognition (poses + moves) ---

class GestureEngine:
    """Gets the hand frame after frame and returns the gestures to trigger.
    It doesn't run any action itself; that's up to the caller (watcher or test).

    Possible events: a pose name, "glisser_gauche/droite/haut/bas" (swipes),
    "pincer" (entering volume mode), then "volume+" / "volume-" at each step."""

    def __init__(self, model):
        self.model = model  # None = moves only
        self.reset()
        self.last_fired = self.last_swipe = 0.0
        self.label, self.confidence = None, 0.0  # latest prediction (for display)

    def reset(self):
        """Hand lost from view: forget whatever was going on."""
        self.votes = deque(maxlen=VOTE_FRAMES)
        self.track = deque()          # recent palm positions: (time, x, y)
        self.hand_since = None        # when the hand appeared
        self.current, self.since = None, 0.0
        self.pinch_since, self.pinching, self.pinch_y = None, False, 0.0

    def update(self, hand, now):
        if hand is None:
            self.reset()
            self.label, self.confidence = None, 0.0
            return []
        points, aspect = hand
        xy = np.array([[p.x, p.y] for p in points])  # coordinates in the image (0 to 1)
        palm_x, palm_y = xy[PALM].mean(axis=0)
        palm_size = np.linalg.norm(xy[9] - xy[WRIST])
        if self.hand_since is None:
            self.hand_since = now

        # 1. Pinch: thumb and index touching, index stretched out (otherwise it's a fist)
        index_extended = np.linalg.norm(xy[8] - xy[WRIST]) > np.linalg.norm(xy[6] - xy[WRIST])
        pinched = index_extended and np.linalg.norm(xy[4] - xy[8]) < PINCH_RATIO * palm_size
        if pinched:
            events = []
            if self.pinch_since is None:
                self.pinch_since = now
            elif not self.pinching and now - self.pinch_since >= PINCH_HOLD:
                self.pinching, self.pinch_y = True, palm_y
                events.append("pincer")
            if self.pinching:  # hand going up = volume up, going down = volume down
                steps = int((self.pinch_y - palm_y) / PINCH_STEP)
                if steps:
                    events += ["volume+" if steps > 0 else "volume-"] * abs(steps)
                    self.pinch_y -= steps * PINCH_STEP
                self.track.clear()
                return events
        else:
            self.pinch_since, self.pinching = None, False

        # 2. Swipe: the palm travels a long way in a short time
        self.track.append((now, palm_x, palm_y))
        while now - self.track[0][0] > SWIPE_WINDOW:
            self.track.popleft()
        dx, dy = palm_x - self.track[0][1], palm_y - self.track[0][2]
        moving = np.hypot(dx, dy) > STILL_DISTANCE
        settled = now - self.hand_since >= HAND_SETTLE
        if settled and now - self.last_swipe >= SWIPE_COOLDOWN:
            direction = None
            if abs(dx) > SWIPE_DISTANCE and abs(dx) > 2 * abs(dy):
                direction = "droite" if dx > 0 else "gauche"  # the image is mirrored: right = your right
            elif abs(dy) > SWIPE_DISTANCE and abs(dy) > 2 * abs(dx):
                direction = "bas" if dy > 0 else "haut"
            if direction:
                self.last_swipe = now
                self.track.clear()
                self.current, self.since = None, now  # no pose right after a swipe
                return [f"glisser_{direction}"]

        # 3. Pose: average the predictions over the last few frames
        if self.model is None:
            return []
        probs = self.model.predict_proba(enrich([hand_vector(points, aspect)]))[0]
        self.votes.append(probs)
        mean = np.mean(self.votes, axis=0)
        best = int(np.argmax(mean))
        self.label, self.confidence = self.model.classes_[best], float(mean[best])

        gesture = None
        if self.confidence >= CONFIDENCE and self.label != NEUTRAL and not moving:
            gesture = self.label
        if gesture != self.current:  # new gesture (or none anymore): restart the timer
            self.current, self.since = gesture, now
        elif gesture and now - self.since >= HOLD_SECONDS and now - self.last_fired >= COOLDOWN:
            self.last_fired = now
            self.since = now + 3600  # don't fire again while the gesture is still held
            return [gesture]
        return []


def live_test():
    """Show the recognized gesture, its certainty and any detected moves, live."""
    model = load_model()
    if model is None:
        print("Aucun modèle à jour : seuls les mouvements seront reconnus (lance « entrainer »).")
    engine = GestureEngine(model)
    tracker = HandTracker()
    last_event, last_event_time = "", 0.0
    print("Q pour quitter.")
    try:
        while True:
            frame, hand = tracker.read()
            now = time.monotonic()
            events = engine.update(hand, now)
            if events:
                last_event, last_event_time = events[-1], now
            if hand:
                draw_hand(frame, hand[0])
            if engine.label:
                ok = engine.confidence >= CONFIDENCE
                draw_text(frame, f"{engine.label}  {engine.confidence:.0%}",
                          color=(0, 200, 0) if ok else (0, 165, 255))
            if engine.pinching:
                draw_text(frame, "PINCEMENT : monte / descends pour le volume", 1, (0, 200, 255))
            if now - last_event_time < 1.5:  # the last event stays on screen for 1.5 s
                description = GESTURE_ACTIONS.get(last_event, (None, ""))[1]
                draw_text(frame, f">> {last_event}  {description}", 2, (0, 255, 0))
            cv2.imshow("Jarvis - test des gestes", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


# --- Actions ---

VK_MEDIA_NEXT, VK_MEDIA_PREV, VK_MEDIA_PLAY_PAUSE = 0xB0, 0xB1, 0xB3  # media keys


def _actions():
    """Maps a gesture or move name to (action, description).
    The action is a function, "ecouter" (handled by the window: Jarvis listens to the mic),
    or None (just displayed). To change something, rename the key or swap the action."""
    from jarvis_tools import _press, VK_VOLUME_UP, VK_VOLUME_DOWN, couper_son, verrouiller_pc
    return {
        # Poses (record them under these names)
        "main_ouverte": ("ecouter", "Jarvis t'écoute"),
        "poing": (couper_son, "Son coupé / rétabli"),
        "pouce_haut": (lambda: _press(VK_VOLUME_UP, 5), "Volume +10 %"),
        "pouce_bas": (lambda: _press(VK_VOLUME_DOWN, 5), "Volume -10 %"),
        "victoire": (lambda: _press(VK_MEDIA_PLAY_PAUSE), "Lecture / pause musique"),
        "rock": (verrouiller_pc, "PC verrouillé"),
        # Moves (recognized without training)
        "glisser_droite": (lambda: _press(VK_MEDIA_NEXT), "Morceau suivant"),
        "glisser_gauche": (lambda: _press(VK_MEDIA_PREV), "Morceau précédent"),
        "glisser_haut": (lambda: _press(VK_VOLUME_UP, 5), "Volume +10 %"),
        "glisser_bas": (lambda: _press(VK_VOLUME_DOWN, 5), "Volume -10 %"),
        "pincer": (None, "Volume : monte / descends la main en pinçant"),
        "volume+": (lambda: _press(VK_VOLUME_UP), "Volume +2 %"),
        "volume-": (lambda: _press(VK_VOLUME_DOWN), "Volume -2 %"),
    }


GESTURE_ACTIONS = _actions()


class GestureWatcher:
    """Watches the webcam in the background and calls on_gesture(name) for every
    gesture or move it recognizes.

    Usage: set enabled = True, then call start(). Setting enabled back to False stops
    watching and turns the webcam off (the thread ends)."""

    def __init__(self, on_gesture, on_error):
        self.on_gesture = on_gesture
        self.on_error = on_error
        self.enabled = False
        self.running = False  # True while the watching thread is alive

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        tracker = None
        try:
            # Reloaded every time it's switched on, so a fresh training is picked up.
            # Without a model, only the moves (swipe, pinch) are recognized.
            engine = GestureEngine(load_model())
            tracker = HandTracker()
            while self.enabled:
                time.sleep(1 / WATCH_FPS)
                _, hand = tracker.read()
                for event in engine.update(hand, time.monotonic()):
                    self.on_gesture(event)
        except Exception as e:
            self.enabled = False
            self.on_error(f"Gestes indisponibles : {e}")
        finally:
            if tracker:
                tracker.close()  # turns the webcam off
            self.running = False
            if self.enabled:  # switched back on while this thread was ending: start again
                self.start()


def model_ready():
    """True if an up-to-date pose model exists."""
    return load_model() is not None


if __name__ == "__main__":
    commands = {"collecter": lambda args: collect(args[0], int(args[1]) if len(args) > 1 else SAMPLES_PER_GESTURE),
                "lister": lambda args: show_counts(),
                "supprimer": lambda args: delete(args[0]),
                "entrainer": lambda args: train(use_hagrid="--hagrid" in args),
                "tester": lambda args: live_test(),
                "importer": lambda args: import_hagrid(int(args[0]) if args else HAGRID_PER_GESTURE)}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(__doc__)
        print("Gestes et mouvements prévus :")
        for g, (_, d) in GESTURE_ACTIONS.items():
            print(f"  {g:15} {d}")
    else:
        commands[sys.argv[1]](sys.argv[2:])
