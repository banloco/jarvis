"""Contrôle par gestes de la main, via la webcam.

Deux familles de gestes :

1. GESTES FIGÉS (poing, main ouverte, pouce levé...) : appris sur TES exemples.
   - MediaPipe (modèle de Google, déjà entraîné) repère 21 points sur la main :
     poignet, articulations et bouts des doigts.
   - Un petit réseau de neurones, entraîné sur tes exemples, reconnaît le geste
     à partir de la position de ces points et de l'angle des doigts.

2. MOUVEMENTS (aucun entraînement nécessaire, calculés à partir du déplacement) :
   - glisser la main vers la gauche / droite / le haut / le bas ;
   - pincer pouce + index puis monter / descendre la main : règle le volume en continu.

Utilisation en ligne de commande (depuis le dossier Jarvis) :
    python jarvis_gestures.py collecter poing      # enregistre des exemples du geste « poing »
    python jarvis_gestures.py lister               # nombre d'exemples par geste
    python jarvis_gestures.py supprimer poing      # efface les exemples d'un geste
    python jarvis_gestures.py entrainer            # entraîne le modèle sur tous les exemples
    python jarvis_gestures.py tester               # reconnaissance en direct, pour vérifier
    python jarvis_gestures.py importer             # ajoute des milliers d'exemples publics (HaGRID)

HaGRID : base publique de gestes filmés par des milliers de personnes (licence CC BY-SA 4.0,
https://github.com/hukenovs/hagrid). « importer » en extrait poing, main ouverte, pouces,
victoire, rock et « rien ». Puis « entrainer --hagrid » ajoute les gestes que tu n'as pas
enregistrés toi-même. Tes propres exemples restent plus fiables : sans --hagrid par défaut.

Conseils pour de bons exemples :
- « rien » est indispensable et doit être VARIÉ : main détendue, de profil, à moitié
  fermée, qui bouge, qui tape au clavier... Tout ce qui ne doit PAS déclencher d'action.
- Pour chaque geste, varie la distance, l'angle, et alterne main gauche / droite.
- Après l'entraînement, la liste des confusions indique quel geste recollecter.

Dans l'interface, le bouton ✋ active la surveillance (voir GESTURE_ACTIONS plus bas).
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
DATA_FILE = BASE_DIR / "gestures_data.npz"          # tes exemples (points de la main + nom du geste)
GESTURE_MODEL = BASE_DIR / "gestures_model.joblib"  # le modèle entraîné
MODEL_VERSION = 3   # à changer si le calcul des caractéristiques change (force un réentraînement)
DATA_FORMAT = 2     # format des exemples stockés : 21 points (x, y), proportions réelles de l'image

# Données publiques HaGRID (des milliers de personnes) : voir la commande « importer »
HAGRID_ZIP = BASE_DIR / "data" / "hagrid" / "annotations.zip"
HAGRID_FILE = BASE_DIR / "data" / "hagrid" / "hagrid_gestures.npz"
HAGRID_URL = ("https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/"
              "annotations_with_landmarks/annotations.zip")
HAGRID_LABELS = {  # nom HaGRID -> nom de geste Jarvis
    "fist": "poing", "palm": "main_ouverte", "like": "pouce_haut", "dislike": "pouce_bas",
    "peace": "victoire", "rock": "rock", "no_gesture": "rien",
}
HAGRID_PER_GESTURE = 3000  # exemples importés par geste
HAGRID_ASPECT = 4 / 3      # largeur / hauteur supposée des photos HaGRID (taille non fournie ;
                           # 9:16, 1:1, 4:3 et 16:9 testés : 4:3 légèrement meilleur, écarts < 2 %)

# Gestes figés
NEUTRAL = "rien"           # geste qui ne déclenche jamais rien
SAMPLES_PER_GESTURE = 300  # exemples enregistrés par défaut par « collecter »
MIN_SAMPLES = 50           # minimum par geste pour entraîner
COUNTDOWN = 3              # secondes pour mettre la main en place avant l'enregistrement
AUGMENT_COPIES = 3         # copies déformées de chaque exemple ajoutées à l'entraînement
CONFIDENCE = 0.85          # certitude minimale (0 à 1) pour accepter un geste
VOTE_FRAMES = 5            # images sur lesquelles on moyenne la décision (~0,3 s)
HOLD_SECONDS = 0.5         # durée pendant laquelle le geste doit être tenu
COOLDOWN = 2.0             # délai minimal entre deux déclenchements d'un geste figé

# Mouvements (distances en fraction de la largeur / hauteur de l'image)
SWIPE_DISTANCE = 0.20      # déplacement minimal pour un glissement (20 % de l'image)
SWIPE_WINDOW = 0.4         # ... réalisé en moins de 0,4 s
SWIPE_COOLDOWN = 1.0       # délai entre deux glissements
HAND_SETTLE = 0.5          # une main qui vient d'apparaître ne compte pas (évite les faux glissements)
STILL_DISTANCE = 0.05      # en dessous, la main est considérée immobile (geste figé possible)
PINCH_RATIO = 0.35         # pouce-index plus proches que 35 % de la paume = pincement
PINCH_HOLD = 0.3           # durée de pincement avant d'entrer en mode volume
PINCH_STEP = 0.03          # déplacement vertical pour un cran de volume (2 %)

WATCH_FPS = 15             # images analysées par seconde en surveillance (économise le processeur)

# Numéros des 21 points MediaPipe : 0 = poignet, puis 4 points par doigt (base -> bout)
WRIST = 0
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
TIPS = [4, 8, 12, 16, 20]
PALM = [0, 5, 9, 13, 17]   # poignet + bases des doigts : le centre de la paume


# --- Étage 1 : repérer la main ---

class HandTracker:
    """Ouvre la webcam et renvoie, pour chaque image, les 21 points de la main (ou None)."""

    def __init__(self):
        if not HAND_MODEL.exists():  # téléchargé la 1re fois (7,5 Mo)
            HAND_MODEL.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,  # suit la main d'une image à l'autre
            num_hands=1)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW : ouverture rapide sous Windows
        self.start = time.monotonic()

    def read(self):
        """Retourne (image, main) : image en miroir (comme un selfie),
        main = (points, largeur/hauteur de l'image) ou None si aucune main n'est visible."""
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


# --- Étage 2 : décrire la main par des nombres ---

def hand_vector(points, aspect):
    """Les 21 points -> 42 nombres (x, y de chaque point). C'est ce qui est stocké.

    - Les coordonnées MediaPipe sont en fraction de la largeur et de la hauteur de l'image :
      on multiplie x par largeur/hauteur pour retrouver les vraies proportions de la main
      (sinon, en 640x480, une main paraît plus étroite qu'elle n'est).
    - La profondeur (z) n'est pas gardée : peu fiable, et absente des données HaGRID.
    """
    coords = np.array([[p.x * aspect, p.y] for p in points], dtype=np.float32)
    return (coords - coords[WRIST]).flatten()  # origine = le poignet


def canonical(P):
    """Retourne horizontalement les mains « à l'envers » pour que toutes aient la même
    orientation (comme une main droite vue de face). Le sens est déduit de la forme de la
    main : l'index est-il à gauche ou à droite de l'auriculaire, vu depuis le poignet ?
    Plus fiable que l'étiquette gauche/droite de MediaPipe, qui se trompe parfois."""
    index, pinky = P[:, 5] - P[:, WRIST], P[:, 17] - P[:, WRIST]
    flip = index[:, 0] * pinky[:, 1] - index[:, 1] * pinky[:, 0] < 0
    P = P.copy()
    P[flip, :, 0] *= -1
    return P


def enrich(vectors):
    """Transforme des exemples (42 nombres par ligne) en caractéristiques pour le modèle.

    1. Même orientation pour toutes les mains (voir canonical).
    2. Mise à l'échelle par la TAILLE DE LA PAUME (poignet -> base du majeur), qui ne change
       pas quand on plie les doigts. (Diviser par la plus grande distance, comme au début,
       effaçait la différence entre un poing et une main ouverte.)
    3. Ajout de mesures qui décrivent directement la forme de la main :
       - l'angle de chaque articulation (doigt tendu ≈ droit, doigt plié ≈ coudé) : 15 valeurs
       - la distance de chaque bout de doigt au poignet : 5 valeurs
       - la distance du pouce aux autres bouts de doigts (pincement, OK...) : 4 valeurs
       - l'écartement entre doigts voisins : 3 valeurs
    """
    P = canonical(np.asarray(vectors, dtype=np.float32).reshape(-1, 21, 2))
    palm = np.linalg.norm(P[:, 9], axis=1)
    P = P / np.maximum(palm, 1e-6)[:, None, None]

    angles = []
    for finger in FINGERS:
        chain = [WRIST] + finger
        for a, b, c in zip(chain, chain[1:], chain[2:]):  # angle au point b entre a-b et b-c
            u, v = P[:, a] - P[:, b], P[:, c] - P[:, b]
            cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-6)
            angles.append(cos)
    tip_wrist = [np.linalg.norm(P[:, t], axis=1) for t in TIPS]
    thumb_tips = [np.linalg.norm(P[:, 4] - P[:, t], axis=1) for t in TIPS[1:]]
    spread = [np.linalg.norm(P[:, a] - P[:, b], axis=1) for a, b in zip(TIPS[1:], TIPS[2:])]
    return np.hstack([P.reshape(len(P), -1), np.stack(angles + tip_wrist + thumb_tips + spread, axis=1)])


def augment(vectors, copies, rng):
    """Crée des copies légèrement déformées des exemples : main tournée de ±20°,
    un peu étirée, avec un léger tremblement. Le modèle apprend ainsi à reconnaître
    un geste même s'il est fait un peu différemment de tes exemples."""
    P = np.repeat(np.asarray(vectors, dtype=np.float32).reshape(-1, 21, 2), copies, axis=0)
    theta = rng.uniform(-np.pi / 9, np.pi / 9, len(P))            # rotation dans le plan de l'image
    cos, sin = np.cos(theta), np.sin(theta)
    x, y = P[:, :, 0].copy(), P[:, :, 1].copy()
    P[:, :, 0], P[:, :, 1] = cos[:, None] * x - sin[:, None] * y, sin[:, None] * x + cos[:, None] * y
    P *= 1 + rng.normal(0, 0.05, (len(P), 1, 2))                   # étirement léger par axe
    scale = np.abs(P).max(axis=(1, 2), keepdims=True)
    P += rng.normal(0, 0.01, P.shape) * scale                      # tremblement
    return P.reshape(len(P), -1)


# --- Affichage ---

def draw_hand(frame, points):
    """Dessine les points et les os de la main sur l'image."""
    h, w = frame.shape[:2]
    xy = [(int(p.x * w), int(p.y * h)) for p in points]
    for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, xy[c.start], xy[c.end], (255, 191, 0), 2)
    for x, y in xy:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)


def draw_text(frame, text, line=0, color=(255, 191, 0)):
    cv2.putText(frame, text, (10, 30 + 30 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# --- Données d'entraînement ---

def load_data():
    """Retourne (X, y) : tes exemples, un par ligne de X (42 nombres), son nom de geste dans y."""
    if not DATA_FILE.exists():
        return np.empty((0, 42), dtype=np.float32), np.empty(0, dtype=str)
    data = np.load(DATA_FILE)
    X, y = data["X"], data["y"]
    if ("format" not in data.files) or int(data["format"]) < DATA_FORMAT:
        # Ancien format (x, y, z en fraction d'une image 640x480) : on garde x, y et on
        # corrige les proportions. Une copie de l'ancien fichier est conservée par sécurité.
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
    """Extrait de HaGRID (annotations publiques, ~686 Mo) les points de main des gestes
    qui nous intéressent (HAGRID_LABELS), en tire `per_gesture` au hasard par geste,
    et les enregistre dans HAGRID_FILE. Les photos elles-mêmes ne sont pas nécessaires."""
    import json
    import zipfile
    if not HAGRID_ZIP.exists():
        HAGRID_ZIP.parent.mkdir(parents=True, exist_ok=True)
        print(f"Téléchargement de HaGRID (686 Mo) vers {HAGRID_ZIP}...")
        urllib.request.urlretrieve(HAGRID_URL, HAGRID_ZIP)

    by_label = {name: [] for name in HAGRID_LABELS.values()}
    with zipfile.ZipFile(HAGRID_ZIP) as archive:
        for path in archive.namelist():
            # « test » est gardé de côté par les auteurs de HaGRID ; .ipynb_checkpoints = copies
            # parasites laissées dans l'archive
            if not path.endswith(".json") or "/test/" in path or ".ipynb_checkpoints" in path:
                continue
            print(f"  lecture de {path}...", flush=True)
            for entry in json.loads(archive.read(path)).values():
                # Une image peut contenir plusieurs mains ; « no_gesture » peut apparaître
                # dans n'importe quel fichier (la 2e main de la personne, souvent)
                for hand_label, points in zip(entry["labels"], entry.get("hand_landmarks", [])):
                    if hand_label in HAGRID_LABELS and len(points) == 21:
                        by_label[HAGRID_LABELS[hand_label]].append(points)

    rng = np.random.default_rng(0)
    X, y = [], []
    for name, hands in by_label.items():
        picked = rng.choice(len(hands), min(per_gesture, len(hands)), replace=False)
        P = np.array(hands, dtype=np.float32)[picked]
        X.append((P - P[:, :1]).reshape(len(P), -1))  # origine = poignet ; x non corrigé ici
        y += [name] * len(P)
        print(f"  {name:13} {len(hands):7} mains disponibles, {len(P)} gardées")
    np.savez_compressed(HAGRID_FILE, X=np.vstack(X), y=np.array(y))
    print(f"Enregistré : {HAGRID_FILE}")


def load_hagrid(aspect=HAGRID_ASPECT):
    """Exemples HaGRID importés (vide si « importer » n'a pas été lancé).
    x est multiplié par `aspect` (largeur / hauteur des photos) pour les vraies proportions."""
    if not HAGRID_FILE.exists():
        return np.empty((0, 42), dtype=np.float32), np.empty(0, dtype=str)
    data = np.load(HAGRID_FILE)
    X = data["X"].reshape(-1, 21, 2) * np.array([aspect, 1], dtype=np.float32)
    return X.reshape(len(X), -1), data["y"]


def collect(name, count=SAMPLES_PER_GESTURE):
    """Ouvre la webcam et enregistre `count` exemples du geste `name`.
    ESPACE lance un compte à rebours puis l'enregistrement ; ESPACE à nouveau = pause."""
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
            elif key in (ord("q"), 27):  # Q ou Échap
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


# --- Étage 3 : entraîner et utiliser le modèle ---

def new_model():
    """Normalisation des valeurs + petit réseau de neurones (2 couches cachées)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500,
                                       early_stopping=True, random_state=0))


def fit(X, y, rng, extra=None):
    """Entraîne un modèle sur tes exemples + leurs copies déformées,
    plus d'éventuels exemples supplémentaires `extra` = (X, y) (HaGRID, non déformés)."""
    Xa = np.vstack([X, augment(X, AUGMENT_COPIES, rng)])
    ya = np.concatenate([y, np.repeat(y, AUGMENT_COPIES)])
    if extra is not None and len(extra[1]):
        Xa, ya = np.vstack([Xa, extra[0]]), np.concatenate([ya, extra[1]])
    return new_model().fit(enrich(Xa), ya)


def train(use_hagrid=False):
    """Estime la précision SUR TES EXEMPLES, affiche les confusions,
    puis entraîne et enregistre le modèle final.

    use_hagrid : ajoute les exemples HaGRID des gestes que tu n'as PAS enregistrés
    (pouces, victoire, rock...), pour les reconnaître sans les collecter.
    Mesuré sur tes données (sept. 2026) : tes exemples seuls = 74 % ; avec HaGRID pour les
    autres gestes = 68 % sur tes propres gestes. Désactivé par défaut pour cette raison.
    (Mélanger HaGRID à tes propres gestes n'aidait pas non plus : 72 % au mieux, et le
    « rien » de HaGRID ne ressemble pas au tien.)"""
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold

    X, y = load_data()
    extra = None
    if use_hagrid:
        H, hy = load_hagrid()
        if not len(hy):
            return print("Aucune donnée HaGRID : lance d'abord « importer ».")
        missing = ~np.isin(hy, np.unique(y))  # seulement les gestes que tu n'as pas enregistrés
        extra = (H[missing], hy[missing])
        print(f"+ HaGRID pour : {', '.join(sorted(set(extra[1])))} ({missing.sum()} exemples)")
    names, counts = np.unique(y, return_counts=True)
    if len(names) < 2:
        return print("Il faut au moins 2 gestes (dont « rien ») pour entraîner.")
    if counts.min() < MIN_SAMPLES:
        return print(f"Chaque geste doit avoir au moins {MIN_SAMPLES} exemples : lance « lister ».")
    rng = np.random.default_rng(0)

    # Validation croisée : 5 fois, on entraîne sur 80 % de tes exemples (+ HaGRID) et on
    # teste sur les 20 % restants. Les exemples étant enregistrés à la suite, chaque test
    # porte sur un moment que le modèle n'a pas vu : l'estimation est honnête.
    predicted = np.empty_like(y, dtype=object)
    for train_idx, test_idx in StratifiedKFold(n_splits=5).split(X, y):
        model = fit(X[train_idx], y[train_idx], rng, extra)
        predicted[test_idx] = model.predict(enrich(X[test_idx]))
    predicted = predicted.astype(str)
    print(classification_report(y, predicted, digits=3, zero_division=0))

    # Les confusions les plus fréquentes : indiquent quel geste recollecter / mieux distinguer
    matrix = confusion_matrix(y, predicted, labels=names)
    confusions = [(matrix[i, j] / matrix[i].sum(), names[i], names[j])
                  for i in range(len(names)) for j in range(len(names)) if i != j and matrix[i, j]]
    for rate, real, guess in sorted(confusions, reverse=True)[:3]:
        if rate >= 0.05:
            print(f"  ⚠ {rate:.0%} des « {real} » pris pour « {guess} »")

    model = fit(X, y, rng, extra)  # modèle final, sur tous les exemples
    joblib.dump({"version": MODEL_VERSION, "model": model}, GESTURE_MODEL)
    print(f"Modèle enregistré : {GESTURE_MODEL.name}")


def load_model():
    """Le modèle entraîné, ou None s'il n'existe pas ou date d'une ancienne version."""
    if not GESTURE_MODEL.exists():
        return None
    saved = joblib.load(GESTURE_MODEL)
    if not isinstance(saved, dict) or saved.get("version") != MODEL_VERSION:
        return None  # ancien format : il faut relancer « entrainer »
    return saved["model"]


# --- Étage 4 : reconnaître en direct (gestes figés + mouvements) ---

class GestureEngine:
    """Reçoit la main image après image et renvoie les gestes à déclencher.
    Aucune action n'est exécutée ici : c'est le rôle de l'appelant (surveillance ou test).

    Événements possibles : nom d'un geste figé, "glisser_gauche/droite/haut/bas",
    "pincer" (entrée en mode volume), puis "volume+" / "volume-" à chaque cran."""

    def __init__(self, model):
        self.model = model  # None = seulement les mouvements
        self.reset()
        self.last_fired = self.last_swipe = 0.0
        self.label, self.confidence = None, 0.0  # dernière prédiction (pour l'affichage)

    def reset(self):
        """Main perdue de vue : on oublie tout ce qui était en cours."""
        self.votes = deque(maxlen=VOTE_FRAMES)
        self.track = deque()          # positions récentes de la paume : (temps, x, y)
        self.hand_since = None        # depuis quand la main est visible
        self.current, self.since = None, 0.0
        self.pinch_since, self.pinching, self.pinch_y = None, False, 0.0

    def update(self, hand, now):
        if hand is None:
            self.reset()
            self.label, self.confidence = None, 0.0
            return []
        points, aspect = hand
        xy = np.array([[p.x, p.y] for p in points])  # coordonnées dans l'image (0 à 1)
        palm_x, palm_y = xy[PALM].mean(axis=0)
        palm_size = np.linalg.norm(xy[9] - xy[WRIST])
        if self.hand_since is None:
            self.hand_since = now

        # 1. Pincement : pouce et index se touchent, index tendu (sinon c'est un poing)
        index_extended = np.linalg.norm(xy[8] - xy[WRIST]) > np.linalg.norm(xy[6] - xy[WRIST])
        pinched = index_extended and np.linalg.norm(xy[4] - xy[8]) < PINCH_RATIO * palm_size
        if pinched:
            events = []
            if self.pinch_since is None:
                self.pinch_since = now
            elif not self.pinching and now - self.pinch_since >= PINCH_HOLD:
                self.pinching, self.pinch_y = True, palm_y
                events.append("pincer")
            if self.pinching:  # main qui monte = volume +, qui descend = volume -
                steps = int((self.pinch_y - palm_y) / PINCH_STEP)
                if steps:
                    events += ["volume+" if steps > 0 else "volume-"] * abs(steps)
                    self.pinch_y -= steps * PINCH_STEP
                self.track.clear()
                return events
        else:
            self.pinch_since, self.pinching = None, False

        # 2. Glissement : grand déplacement de la paume en peu de temps
        self.track.append((now, palm_x, palm_y))
        while now - self.track[0][0] > SWIPE_WINDOW:
            self.track.popleft()
        dx, dy = palm_x - self.track[0][1], palm_y - self.track[0][2]
        moving = np.hypot(dx, dy) > STILL_DISTANCE
        settled = now - self.hand_since >= HAND_SETTLE
        if settled and now - self.last_swipe >= SWIPE_COOLDOWN:
            direction = None
            if abs(dx) > SWIPE_DISTANCE and abs(dx) > 2 * abs(dy):
                direction = "droite" if dx > 0 else "gauche"  # l'image est en miroir : droite = ta droite
            elif abs(dy) > SWIPE_DISTANCE and abs(dy) > 2 * abs(dx):
                direction = "bas" if dy > 0 else "haut"
            if direction:
                self.last_swipe = now
                self.track.clear()
                self.current, self.since = None, now  # pas de geste figé juste après
                return [f"glisser_{direction}"]

        # 3. Geste figé : moyenne des prédictions sur les dernières images
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
        if gesture != self.current:  # nouveau geste (ou plus de geste) : on relance le chrono
            self.current, self.since = gesture, now
        elif gesture and now - self.since >= HOLD_SECONDS and now - self.last_fired >= COOLDOWN:
            self.last_fired = now
            self.since = now + 3600  # pas de 2e déclenchement tant que le geste reste tenu
            return [gesture]
        return []


def live_test():
    """Affiche en direct le geste reconnu, la certitude et les mouvements détectés."""
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
            if now - last_event_time < 1.5:  # le dernier événement reste affiché 1,5 s
                description = GESTURE_ACTIONS.get(last_event, (None, ""))[1]
                draw_text(frame, f">> {last_event}  {description}", 2, (0, 255, 0))
            cv2.imshow("Jarvis - test des gestes", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


# --- Actions ---

VK_MEDIA_NEXT, VK_MEDIA_PREV, VK_MEDIA_PLAY_PAUSE = 0xB0, 0xB1, 0xB3  # touches multimédia


def _actions():
    """Associe un nom de geste ou de mouvement à (action, description).
    L'action est une fonction, "ecouter" (géré par l'interface : Jarvis écoute le micro),
    ou None (simple affichage). Pour changer : renomme la clé ou l'action."""
    from jarvis_tools import _press, VK_VOLUME_UP, VK_VOLUME_DOWN, couper_son, verrouiller_pc
    return {
        # Gestes figés (à collecter sous ces noms)
        "main_ouverte": ("ecouter", "Jarvis t'écoute"),
        "poing": (couper_son, "Son coupé / rétabli"),
        "pouce_haut": (lambda: _press(VK_VOLUME_UP, 5), "Volume +10 %"),
        "pouce_bas": (lambda: _press(VK_VOLUME_DOWN, 5), "Volume -10 %"),
        "victoire": (lambda: _press(VK_MEDIA_PLAY_PAUSE), "Lecture / pause musique"),
        "rock": (verrouiller_pc, "PC verrouillé"),
        # Mouvements (reconnus sans entraînement)
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
    """Surveille la webcam en arrière-plan et appelle on_gesture(nom) pour chaque
    geste ou mouvement reconnu.

    Usage : mettre enabled = True puis appeler start(). Remettre enabled = False
    arrête la surveillance et éteint la webcam (le thread se termine)."""

    def __init__(self, on_gesture, on_error):
        self.on_gesture = on_gesture
        self.on_error = on_error
        self.enabled = False
        self.running = False  # True tant que le thread de surveillance tourne

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        tracker = None
        try:
            # Relu à chaque activation : un nouvel entraînement est pris en compte.
            # Sans modèle, seuls les mouvements (glisser, pincer) sont reconnus.
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
                tracker.close()  # éteint la webcam
            self.running = False
            if self.enabled:  # réactivé pendant que ce thread se terminait : on relance
                self.start()


def model_ready():
    """True si un modèle de gestes figés à jour existe."""
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
