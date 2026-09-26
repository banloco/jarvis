"""Cerveau commun de Jarvis, utilisé par les deux interfaces (jarvis.py et jarvis_ui.py).

Contenu :
- les réglages (modèle, tailles de mémoire, micro...) ;
- le micro (listen_voice) — la voix est dans jarvis_voice.py ;
- la mémoire : conversation récente (memory.json), faits (facts.json)
  et mémoire à long terme (voir jarvis_memory.py) ;
- ask_model : la boucle qui interroge le modèle et exécute ses outils.
"""
import threading
import queue
import json
import shutil
import subprocess
import time
import httpx
import numpy as np
import ollama
import speech_recognition as sr
import sounddevice as sd
from datetime import datetime
from pathlib import Path
from jarvis_memory import LongTermMemory

# --- Réglages ---

BASE_DIR = Path(__file__).resolve().parent   # fichiers de mémoire rangés à côté du code
MEMORY_FILE = BASE_DIR / "memory.json"       # conversation récente
FACTS_FILE = BASE_DIR / "facts.json"         # faits notés via l'outil « memoriser »
MODEL = "qwen2.5:3b"  # doit supporter les outils (qwen2.5, llama3.1...) ; qwen2.5:7b si le PC suit
THINK = None          # modèles qui « réfléchissent » avant de répondre (qwen3...) : False = plus rapide,
                      # True = plus réfléchi ; None = ne rien envoyer (modèles sans cette option)
MAX_CONTEXT = 24      # derniers messages envoyés au modèle (sa « mémoire de travail »),
                      # appels d'outils compris
MAX_STORED = 200      # messages conservés dans memory.json (l'archive complète est à part)
MAX_FACTS = 30        # faits récents toujours rappelés au modèle
MAX_MEMORIES = 5      # souvenirs anciens retrouvés par le sens à chaque question
MAX_TOOL_ROUNDS = 5   # appels d'outils enchaînés avant de forcer une réponse (anti-boucle)
KEEP_ALIVE = "30m"    # durée pendant laquelle Ollama garde les modèles en mémoire vive
OLLAMA_TIMEOUT = 180  # secondes sans aucune nouvelle d'Ollama avant d'abandonner (évite de rester figé)
MAX_TOOL_RESULT = 300 # caractères d'un résultat d'outil gardés dans l'historique (le reste alourdit tout)
SAMPLE_RATE = 16000   # qualité d'enregistrement (16 kHz suffit pour la voix)
FRAME = 1280          # taille d'un bloc audio : 1280 échantillons = 80 ms (format attendu par openWakeWord)
SILENCE_END = 1.0     # secondes de silence qui marquent la fin de la phrase
NO_SPEECH = 5.0       # abandon si personne ne parle pendant ce temps
MAX_COMMAND = 15.0    # durée maximale d'une demande
MIN_SPEECH_RMS = 300  # volume minimal considéré comme de la parole (monte-le si le micro capte trop de bruit)
# Reconnaissance vocale hors ligne (secours quand Google est injoignable) : petit modèle français
VOSK_MODEL = "vosk-model-small-fr-0.22"
VOSK_DIR = BASE_DIR / "models" / VOSK_MODEL
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL}.zip"

# Instructions permanentes données au modèle (le « prompt système »)
JARVIS_PERSONA = """Tu es Jarvis, l'assistant personnel de l'utilisateur, inspiré de l'IA de Tony Stark.
Tu es intelligent, direct, légèrement sarcastique mais loyal.
Tu t'exprimes en français sauf si l'utilisateur te parle en anglais.
Tu n'es pas un simple assistant — tu es une extension de son intelligence.
Ne te présente jamais comme une IA générique. Tu es Jarvis et tu appartiens à l'utilisateur.
Tu contrôles l'ordinateur de l'utilisateur grâce à tes outils : utilise-les dès qu'une demande
le nécessite (ouvrir une appli, chercher sur le web, météo, heure, fichiers, volume, mémoriser).
Pour une simple conversation, réponds directement sans outil.
N'invente jamais le résultat d'une action : fie-toi à ce que renvoie l'outil.
N'annonce jamais une action (« je recherche », « je lance »...) sans appeler l'outil
correspondant dans la même réponse. Tu as accès à la météo, à l'heure et au web :
ne prétends jamais le contraire. Si aucun outil ne permet l'action, dis-le franchement.
Tu as une mémoire à long terme : appuie-toi sur les souvenirs fournis quand ils sont utiles.
Tes réponses doivent être concises — maximum 3 phrases."""


# Client Ollama avec délai maximum : sans lui, un Ollama bloqué figeait Jarvis pour toujours.
# (Pendant le streaming, le délai s'applique entre deux morceaux de réponse.)
client = ollama.Client(timeout=OLLAMA_TIMEOUT)


def now_text():
    """Date et heure en toutes lettres : « samedi 26 septembre 2026, 14:05 »."""
    jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    mois = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
            "août", "septembre", "octobre", "novembre", "décembre"]
    now = datetime.now()
    return f"{jours[now.weekday()]} {now.day} {mois[now.month - 1]} {now.year}, {now:%H:%M}"


def ensure_ollama(on_status=None, wait=20):
    """Vérifie qu'Ollama répond ; sinon tente de le lancer et attend jusqu'à `wait` secondes.
    Retourne None si tout va bien, sinon un message d'erreur à afficher."""
    try:
        client.list()
        return None
    except Exception:
        pass
    exe = shutil.which("ollama")
    if not exe:
        return "Ollama est introuvable : installe-le depuis https://ollama.com"
    if on_status:
        on_status("Démarrage d'Ollama...")
    # Lancé détaché et sans fenêtre : il continue de tourner même si Jarvis se ferme
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    for _ in range(wait):
        time.sleep(1)
        try:
            client.list()
            return None
        except Exception:
            continue
    return "Ollama ne démarre pas. Lance-le à la main avec « ollama serve »."


# --- Fichiers JSON ---

def load_json(path, default):
    """Lit un fichier JSON ; renvoie `default` s'il est absent ou abîmé."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, data):
    """Écrit d'abord dans un fichier temporaire puis le renomme :
    un crash en pleine écriture ne corrompt jamais le fichier existant."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# --- Mémoire : conversation récente et faits ---

def load_history():
    """Conversation récente : messages {"role": "user" / "assistant" / "tool", "content": ...}.
    Les messages "assistant" peuvent contenir "tool_calls" (outils demandés) et sont suivis
    des messages "tool" (résultats) : le modèle voit ainsi des exemples où il agit vraiment."""
    history = load_json(MEMORY_FILE, [])
    # L'ancien format stockait le prompt système : on ne garde que le dialogue
    return [m for m in history if m.get("role") in ("user", "assistant", "tool")]


def recent_window(history):
    """Les MAX_CONTEXT derniers messages, en commençant toujours par une question de l'utilisateur
    (couper au milieu d'un échange laisserait un résultat d'outil sans sa demande)."""
    window = history[-MAX_CONTEXT:]
    while window and window[0]["role"] != "user":
        window = window[1:]
    return window


def save_history(history):
    del history[:-MAX_STORED]  # coupe les plus anciens (ils restent dans la mémoire long terme)
    save_json(MEMORY_FILE, history)


def load_facts():
    """Liste des faits notés, du plus ancien au plus récent."""
    return [v["content"] for v in load_json(FACTS_FILE, {}).values()]


def add_fact(fact):
    """Note un fait dans facts.json ET dans la mémoire à long terme."""
    facts = load_json(FACTS_FILE, {})
    facts[f"fact_{datetime.now():%Y%m%d%H%M%S%f}"] = {"content": fact, "date": datetime.now().isoformat()}
    save_json(FACTS_FILE, facts)
    remember(lambda memory: memory.add(fact, "fait"))


# --- Mémoire à long terme ---

_memory = None                    # instance unique, créée à la demande
_memory_lock = threading.Lock()   # évite de la créer deux fois (warm_up + 1re question)


def get_memory():
    """Mémoire à long terme, créée au premier appel. None si Ollama est indisponible."""
    global _memory
    with _memory_lock:
        if _memory is None:
            try:
                memory = LongTermMemory(BASE_DIR, KEEP_ALIVE)
                memory.import_existing(load_json(FACTS_FILE, {}).values(), load_history())
                _memory = memory
            except Exception:
                return None  # Ollama éteint ? nouvel essai au prochain appel
        return _memory


def remember(action):
    """Exécute action(mémoire) sans jamais faire échouer la conversation :
    si la mémoire long terme est en panne, Jarvis répond quand même, sans souvenirs."""
    memory = get_memory()
    if memory:
        try:
            return action(memory)
        except Exception:
            pass
    return None


# --- Micro ---

def volume(frame):
    """Volume moyen (RMS) d'un bloc audio : ~0 en silence, plusieurs centaines quand on parle."""
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


def mic_stream(frames_queue):
    """Ouvre le micro : chaque bloc de 80 ms est déposé dans frames_queue.
    À utiliser avec « with » (le micro est libéré à la sortie du bloc)."""
    return sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME,
                          callback=lambda data, *_: frames_queue.put(data[:, 0].copy()))


def read_frames(frames_queue):
    """Blocs audio du micro, un par un, au fur et à mesure qu'ils arrivent.
    (Pas de iter(queue.get, None) : il compare chaque tableau numpy à None, ce qui plante.)"""
    while True:
        yield frames_queue.get()


def record_until_silence(frames, noise_floor=None, no_speech=NO_SPEECH):
    """Lit des blocs audio jusqu'à ce que l'utilisateur ait fini de parler.

    frames      : itérable de blocs int16 de 80 ms (venant du micro)
    noise_floor : volume du bruit ambiant, s'il est connu ; sinon il est mesuré
                  sur les premiers blocs (on suppose qu'on ne parle pas pile au début)
    no_speech   : secondes d'attente maximum si personne ne commence à parler
    Retourne l'audio complet (np.ndarray), ou None si personne n'a parlé.
    """
    per_second = SAMPLE_RATE / FRAME  # 12,5 blocs par seconde
    chunks, heard, silent_frames = [], False, 0
    threshold = max(MIN_SPEECH_RMS, 3 * noise_floor) if noise_floor else None

    for frame in frames:
        chunks.append(frame)
        if threshold is None:  # calibrage sur les 4 premiers blocs (0,3 s)
            if len(chunks) == 4:
                threshold = max(MIN_SPEECH_RMS, 3 * float(np.median([volume(c) for c in chunks])))
            continue

        if volume(frame) > threshold:
            heard, silent_frames = True, 0
        else:
            silent_frames += 1

        if heard and silent_frames >= SILENCE_END * per_second:
            break  # fin de phrase
        if not heard and len(chunks) >= no_speech * per_second:
            return None  # personne n'a parlé
        if len(chunks) >= MAX_COMMAND * per_second:
            break  # sécurité : demande trop longue

    return np.concatenate(chunks) if heard else None


def transcribe(audio):
    """Transforme l'audio en texte : Google (plus précis, Internet nécessaire), et Vosk
    sur le PC si Google est injoignable. Retourne (texte, None) ou (None, message d'erreur)."""
    if audio is None:
        return None, "Rien entendu. Réessayez."
    try:
        # Le son est passé directement en mémoire, sans fichier temporaire (2 = octets par échantillon)
        data = sr.AudioData(audio.tobytes(), SAMPLE_RATE, 2)
        return sr.Recognizer().recognize_google(data, language="fr-FR"), None
    except sr.UnknownValueError:
        return None, "Je n'ai pas compris. Réessayez."
    except Exception:  # pas d'Internet, Google indisponible... -> reconnaissance hors ligne
        return transcribe_offline(audio)


_vosk = None  # modèle hors ligne, chargé au premier besoin (~2 s)


def download_offline_speech_model():
    """Télécharge le modèle Vosk français (~41 Mo) s'il n'est pas déjà là. À faire avec Internet."""
    import urllib.request
    import zipfile
    if VOSK_DIR.exists():
        return
    VOSK_DIR.parent.mkdir(parents=True, exist_ok=True)
    archive = VOSK_DIR.parent / f"{VOSK_MODEL}.zip"
    urllib.request.urlretrieve(VOSK_URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(VOSK_DIR.parent)
    archive.unlink()  # l'archive ne sert plus une fois décompressée


def transcribe_offline(audio):
    """Reconnaissance vocale sur le PC, sans Internet (Vosk). Moins précise que Google."""
    global _vosk
    if not VOSK_DIR.exists():
        return None, "Pas d'Internet, et la reconnaissance hors ligne n'est pas installée (python installer.py)."
    try:
        from vosk import KaldiRecognizer, Model, SetLogLevel
        if _vosk is None:
            SetLogLevel(-1)  # Vosk est très bavard par défaut
            _vosk = Model(str(VOSK_DIR))
        recognizer = KaldiRecognizer(_vosk, SAMPLE_RATE)
        recognizer.AcceptWaveform(audio.tobytes())
        text = json.loads(recognizer.FinalResult()).get("text", "").strip()
    except Exception as e:
        return None, f"Reconnaissance vocale indisponible : {e}"
    return (text, None) if text else (None, "Je n'ai pas compris. Réessayez.")


def listen_voice():
    """Écoute le micro jusqu'à la fin de la phrase, puis transcrit (bouton 🎤, mode terminal).
    Retourne (texte, None) si ça marche, (None, message d'erreur) sinon."""
    try:
        frames = queue.Queue()
        with mic_stream(frames):
            audio = record_until_silence(read_frames(frames))
    except Exception as e:
        return None, f"Micro indisponible : {e}"
    return transcribe(audio)


# --- Modèle ---

def warm_up(history=(), tools=(), on_error=None, on_status=None):
    """Prépare les modèles en arrière-plan dès le lancement.

    0. Vérifie qu'Ollama tourne (et le lance si besoin) ; sinon appelle on_error(message).
    1. Il ne suffit pas de charger le modèle : avant de répondre, il doit « lire » tout le
       début de la conversation (instructions, description des outils, historique), ce qui
       prend ~50 s sur processeur. On lui fait donc lire ce début dès maintenant avec une
       fausse question dont on ne garde pas la réponse (1 seul mot généré). Ollama garde
       cette lecture en mémoire : la 1re vraie question n'aura plus qu'à lire la fin."""
    def load():
        problem = ensure_ollama(on_status)
        if problem:
            if on_error:
                on_error(problem)
            return
        try:
            messages = build_prompt(list(history), "Bonjour")[:-1]  # sans souvenirs : début commun
            client.chat(model=MODEL, messages=messages + [{"role": "user", "content": "Bonjour"}],
                        tools=list(tools), keep_alive=KEEP_ALIVE, options={"num_predict": 1},
                        **({} if THINK is None else {"think": THINK}))
        except Exception:
            pass  # l'erreur sera signalée à la première vraie question
        get_memory()  # charge aussi le modèle d'embeddings et indexe l'existant
    thread = threading.Thread(target=load, daemon=True)
    thread.start()
    return thread  # thread.join() permet d'attendre la fin de la préparation


def build_prompt(history, user_input):
    """Assemble tout ce que le modèle voit :
    instructions + faits + conversation récente + (souvenirs pertinents + nouvelle question).

    L'ordre compte pour la vitesse : Ollama garde en mémoire le début de la conversation
    déjà lu et ne relit que ce qui a changé. Tout ce qui change à chaque question (les
    souvenirs retrouvés) est donc mis À LA FIN, dans le dernier message. Placés dans les
    instructions du début, ils obligeaient à tout relire : ~40 s au lieu de ~10 s."""
    system = JARVIS_PERSONA
    facts = load_facts()[-MAX_FACTS:]
    if facts:
        system += "\n\nCe que tu sais sur l'utilisateur:\n" + "".join(f"- {f}\n" for f in facts)

    # Souvenirs anciens liés à la question, sauf ceux déjà visibles par ailleurs
    # (échanges encore dans la conversation récente, faits déjà listés ci-dessus)
    window = recent_window(history)
    recent = {m["content"] for m in window if m["role"] == "user"}
    memories = remember(lambda memory: memory.search(
        user_input, MAX_MEMORIES,
        skip=lambda item: item.get("user") in recent or (item["kind"] == "fait" and item["text"] in facts)))
    # Dernier message : date du jour (+ souvenirs) + demande. La date change à chaque minute :
    # placée ici, à la fin, elle ne force pas à relire tout le début.
    content = f"[Nous sommes le {now_text()}]\n"
    if memories:
        content += ("[Souvenirs de conversations passées, à utiliser seulement s'ils sont utiles :]\n"
                    + "".join(f"- [{m['date'][:10] or 'date inconnue'}] {m['text'][:400]}\n" for m in memories))
    content += f"[Demande de l'utilisateur :]\n{user_input}"

    return [{"role": "system", "content": system}, *window, {"role": "user", "content": content}]


def run_tool(tools, call):
    """Exécute l'outil demandé par le modèle. Toute erreur est renvoyée au modèle
    sous forme de texte (il pourra l'expliquer à l'utilisateur) au lieu de faire planter Jarvis."""
    func = tools.get(call.function.name)
    if not func:
        return f"Outil inconnu : {call.function.name}"
    try:
        return str(func(**(call.function.arguments or {})))
    except Exception as e:
        return f"Erreur de l'outil {call.function.name} : {e}"


def ask_model(history, user_input, tools=(), on_token=None, on_status=None):
    """Pose la question au modèle et retourne sa réponse finale.

    Déroulé :
    1. le modèle reçoit la question et la liste des outils ;
    2. s'il demande un outil, on l'exécute et on lui renvoie le résultat ;
    3. on recommence jusqu'à ce qu'il réponde par du texte (MAX_TOOL_ROUNDS max) ;
    4. l'échange est ajouté à l'historique et à la mémoire à long terme.

    on_token(texte)  : appelé pour chaque morceau de réponse (affichage en direct)
    on_status(texte) : appelé pour indiquer l'étape en cours (« 🌐 Recherche web... »)
    """
    by_name = {f.__name__: f for f in tools}
    messages = build_prompt(history, user_input)
    reply = ""
    tool_steps = []  # demandes d'outils + résultats de ce tour, gardés dans l'historique

    # Le petit modèle renvoie parfois une réponse totalement vide (ni texte, ni outil) :
    # dans ce cas seulement, on lui repose la question une 2e fois (il a une part de hasard).
    for attempt in range(2):
        for _ in range(MAX_TOOL_ROUNDS):
            content, calls = "", []
            # stream=True : la réponse arrive morceau par morceau au lieu d'un bloc à la fin
            options = {} if THINK is None else {"think": THINK}
            try:
                for chunk in client.chat(model=MODEL, messages=messages, tools=list(tools),
                                         stream=True, keep_alive=KEEP_ALIVE, **options):
                    piece = chunk.message.content or ""
                    content += piece
                    if piece and on_token:
                        on_token(piece)
                    calls.extend(chunk.message.tool_calls or [])
            except httpx.TimeoutException:
                raise TimeoutError(f"Ollama ne répond plus depuis {OLLAMA_TIMEOUT} s. "
                                   "Réessaie, ou relance Ollama.") from None
            reply += content

            if not calls:
                break  # réponse texte finale : terminé
            # On garde la trace de la demande d'outil et de son résultat : le modèle s'appuie
            # dessus au tour suivant, et l'historique en garde l'exemple pour les prochaines fois
            # (format dictionnaire, pour pouvoir l'enregistrer dans memory.json)
            step = [{"role": "assistant", "content": content,
                     "tool_calls": [{"function": {"name": c.function.name, "arguments": c.function.arguments or {}}}
                                    for c in calls]}]
            for call in calls:
                if on_status:
                    on_status(getattr(by_name.get(call.function.name), "label", f"⚙️ {call.function.name}..."))
                step.append({"role": "tool", "tool_name": call.function.name, "content": run_tool(by_name, call)})
            messages += step  # le modèle voit le résultat complet pour cette réponse...
            # ... mais l'historique n'en garde qu'un extrait : une recherche web complète alourdirait
            # (et ralentirait) toutes les questions suivantes
            tool_steps += [dict(m, content=m["content"][:MAX_TOOL_RESULT]) if m["role"] == "tool" else m
                           for m in step]
            if on_status:
                on_status("Jarvis réfléchit...")
        if reply.strip() or tool_steps:
            break

    reply = reply.strip()
    if not reply:
        # Réponse vide : si un outil a tourné, le modèle a agi sans commenter ; sinon il a
        # simplement échoué, et prétendre « C'est fait » serait mentir.
        reply = "C'est fait." if tool_steps else "Je n'ai pas bien saisi. Pouvez-vous reformuler ?"
        if on_token:
            on_token(reply)
    history.append({"role": "user", "content": user_input})
    history.extend(tool_steps)
    history.append({"role": "assistant", "content": reply})
    save_history(history)
    # Mémoire à long terme : seulement les vraies conversations. Un échange qui a utilisé un
    # outil (météo, heure, web, ouvrir une appli...) est une info périmée ou une simple
    # commande : ressorti comme « souvenir », il faisait redonner la météo d'hier sans
    # rappeler l'outil. (Les faits notés avec « memoriser » sont enregistrés à part.)
    if not tool_steps:
        remember(lambda memory: memory.add_exchange(user_input, reply))
    return reply
