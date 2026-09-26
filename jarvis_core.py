"""Jarvis's brain, shared by both front ends (jarvis.py and jarvis_ui.py).

What lives here:
- the settings (model, memory sizes, microphone...);
- the microphone (listen_voice). The speaking side is in jarvis_voice.py;
- memory: the recent conversation (memory.json), saved facts (facts.json)
  and long-term memory (see jarvis_memory.py);
- ask_model: the loop that talks to the model and runs the tools it asks for.
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
from jarvis_config import USER_REF
from jarvis_memory import LongTermMemory

# --- Settings ---

BASE_DIR = Path(__file__).resolve().parent   # memory files sit next to the code
MEMORY_FILE = BASE_DIR / "memory.json"       # recent conversation
FACTS_FILE = BASE_DIR / "facts.json"         # facts saved with the "memoriser" tool
MODEL = "qwen2.5:3b"  # must support tools (qwen2.5, llama3.1...); try qwen2.5:7b if your PC can take it
THINK = None          # for models that "think" before answering (qwen3...): False = faster,
                      # True = more careful; None = don't send the option at all (most models)
MAX_CONTEXT = 24      # how many recent messages the model gets (its working memory),
                      # tool calls included
MAX_STORED = 200      # messages kept in memory.json (older ones live on in long-term memory)
MAX_FACTS = 30        # most recent facts, always given to the model
MAX_MEMORIES = 5      # old memories pulled back in by meaning, for each question
MAX_TOOL_ROUNDS = 5   # tool calls in a row before we force an answer (stops loops)
KEEP_ALIVE = "30m"    # how long Ollama keeps the models loaded in RAM
OLLAMA_TIMEOUT = 180  # seconds without a word from Ollama before we give up (instead of hanging)
MAX_TOOL_RESULT = 300 # characters of a tool result kept in the history (more just slows things down)
SAMPLE_RATE = 16000   # recording quality; 16 kHz is plenty for speech
FRAME = 1280          # one audio block = 1280 samples = 80 ms (the size openWakeWord expects)
SILENCE_END = 1.0     # this much silence means you've finished your sentence
NO_SPEECH = 5.0       # give up if nobody starts talking within this time
MAX_COMMAND = 15.0    # longest request we'll record
MIN_SPEECH_RMS = 300  # quietest sound we count as speech (raise it if your mic picks up a lot of noise)
# Offline speech recognition, used when Google can't be reached: a small French model
VOSK_MODEL = "vosk-model-small-fr-0.22"
VOSK_DIR = BASE_DIR / "models" / VOSK_MODEL
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL}.zip"

# The standing instructions for the model (the "system prompt"). Jarvis speaks French,
# so this stays in French: the model answers in the language it's instructed in.
JARVIS_PERSONA = f"""Tu es Jarvis, l'assistant personnel de {USER_REF}, inspiré de l'IA de Tony Stark.
Tu es intelligent, direct, légèrement sarcastique mais loyal.
Tu t'exprimes en français sauf si {USER_REF} te parle en anglais.
Tu n'es pas un simple assistant — tu es une extension de son intelligence.
Ne te présente jamais comme une IA générique. Tu es Jarvis et tu appartiens à {USER_REF}.
Tu contrôles l'ordinateur de {USER_REF} grâce à tes outils : utilise-les dès qu'une demande
le nécessite (ouvrir une appli, chercher sur le web, météo, heure, fichiers, volume, mémoriser).
Pour une simple conversation, réponds directement sans outil.
N'invente jamais le résultat d'une action : fie-toi à ce que renvoie l'outil.
N'annonce jamais une action (« je recherche », « je lance »...) sans appeler l'outil
correspondant dans la même réponse. Tu as accès à la météo, à l'heure et au web :
ne prétends jamais le contraire. Si aucun outil ne permet l'action, dis-le franchement.
Tu as une mémoire à long terme : appuie-toi sur les souvenirs fournis quand ils sont utiles.
Tes réponses doivent être concises — maximum 3 phrases."""


# Ollama client with a timeout. Without it, a stuck Ollama froze Jarvis forever.
# (While streaming, the timeout applies between two chunks of the answer.)
client = ollama.Client(timeout=OLLAMA_TIMEOUT)


def now_text():
    """Date and time spelled out in French: "samedi 26 septembre 2026, 14:05"."""
    jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    mois = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
            "août", "septembre", "octobre", "novembre", "décembre"]
    now = datetime.now()
    return f"{jours[now.weekday()]} {now.day} {mois[now.month - 1]} {now.year}, {now:%H:%M}"


def ensure_ollama(on_status=None, wait=20):
    """Make sure Ollama answers. If it doesn't, try to start it and wait up to `wait` seconds.
    Returns None when all is well, otherwise an error message to show."""
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
    # Started detached and without a window, so it keeps running after Jarvis closes
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


# --- JSON files ---

def load_json(path, default):
    """Read a JSON file, or return `default` if it's missing or damaged."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, data):
    """Write to a temporary file first, then rename it over the real one.
    That way a crash halfway through a write can never corrupt the existing file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# --- Memory: recent conversation and facts ---

def load_history():
    """The recent conversation: messages {"role": "user" / "assistant" / "tool", "content": ...}.
    "assistant" messages can carry "tool_calls" and are followed by the "tool" messages with
    the results. Keeping those means the model sees examples of itself actually doing things."""
    history = load_json(MEMORY_FILE, [])
    # The old format also stored the system prompt; we only keep the dialogue
    return [m for m in history if m.get("role") in ("user", "assistant", "tool")]


def recent_window(history):
    """The last MAX_CONTEXT messages, always starting with something the user said
    (cutting in the middle of an exchange would leave a tool result without its request)."""
    window = history[-MAX_CONTEXT:]
    while window and window[0]["role"] != "user":
        window = window[1:]
    return window


def save_history(history):
    del history[:-MAX_STORED]  # drop the oldest ones (they're still in long-term memory)
    save_json(MEMORY_FILE, history)


def load_facts():
    """Saved facts, oldest first."""
    return [v["content"] for v in load_json(FACTS_FILE, {}).values()]


def add_fact(fact):
    """Save a fact in facts.json and in long-term memory."""
    facts = load_json(FACTS_FILE, {})
    facts[f"fact_{datetime.now():%Y%m%d%H%M%S%f}"] = {"content": fact, "date": datetime.now().isoformat()}
    save_json(FACTS_FILE, facts)
    remember(lambda memory: memory.add(fact, "fait"))


# --- Long-term memory ---

_memory = None                    # single instance, created when first needed
_memory_lock = threading.Lock()   # so warm_up and the first question don't both create it


def get_memory():
    """Long-term memory, created on first use. None if Ollama isn't available."""
    global _memory
    with _memory_lock:
        if _memory is None:
            try:
                memory = LongTermMemory(BASE_DIR, KEEP_ALIVE)
                memory.import_existing(load_json(FACTS_FILE, {}).values(), load_history())
                _memory = memory
            except Exception:
                return None  # Ollama off? We'll try again next time
        return _memory


def remember(action):
    """Run action(memory) without ever breaking the conversation: if long-term memory
    is down, Jarvis still answers, just without his memories."""
    memory = get_memory()
    if memory:
        try:
            return action(memory)
        except Exception:
            pass
    return None


# --- Microphone ---

def volume(frame):
    """Average loudness (RMS) of an audio block: close to 0 in silence, a few hundred when talking."""
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


def mic_stream(frames_queue):
    """Open the microphone. Every 80 ms block goes into frames_queue.
    Use it with "with" so the mic is released when the block ends."""
    return sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME,
                          callback=lambda data, *_: frames_queue.put(data[:, 0].copy()))


def read_frames(frames_queue):
    """Microphone blocks, one at a time, as they arrive.
    (Not iter(queue.get, None): that compares each numpy array to None and crashes.)"""
    while True:
        yield frames_queue.get()


def record_until_silence(frames, noise_floor=None, no_speech=NO_SPEECH):
    """Read audio blocks until the user has finished talking.

    frames      : iterable of 80 ms int16 blocks (from the mic)
    noise_floor : background noise level if we already know it; otherwise we measure it
                  on the first few blocks (assuming nobody talks in the very first instant)
    no_speech   : how many seconds to wait at most for someone to start talking
    Returns the whole recording (np.ndarray), or None if nobody spoke.
    """
    per_second = SAMPLE_RATE / FRAME  # 12.5 blocks per second
    chunks, heard, silent_frames = [], False, 0
    threshold = max(MIN_SPEECH_RMS, 3 * noise_floor) if noise_floor else None

    for frame in frames:
        chunks.append(frame)
        if threshold is None:  # calibrate on the first 4 blocks (0.3 s)
            if len(chunks) == 4:
                threshold = max(MIN_SPEECH_RMS, 3 * float(np.median([volume(c) for c in chunks])))
            continue

        if volume(frame) > threshold:
            heard, silent_frames = True, 0
        else:
            silent_frames += 1

        if heard and silent_frames >= SILENCE_END * per_second:
            break  # end of the sentence
        if not heard and len(chunks) >= no_speech * per_second:
            return None  # nobody spoke
        if len(chunks) >= MAX_COMMAND * per_second:
            break  # safety net: request too long

    return np.concatenate(chunks) if heard else None


def transcribe(audio):
    """Turn the audio into text: Google first (more accurate, needs Internet), then Vosk
    on the PC if Google can't be reached. Returns (text, None) or (None, error message)."""
    if audio is None:
        return None, "Rien entendu. Réessayez."
    try:
        # The sound is handed over straight from memory, no temp file (2 = bytes per sample)
        data = sr.AudioData(audio.tobytes(), SAMPLE_RATE, 2)
        return sr.Recognizer().recognize_google(data, language="fr-FR"), None
    except sr.UnknownValueError:
        return None, "Je n'ai pas compris. Réessayez."
    except Exception:  # no Internet, Google down... -> offline recognition
        return transcribe_offline(audio)


_vosk = None  # offline model, loaded the first time we need it (~2 s)


def download_offline_speech_model():
    """Download the French Vosk model (~41 MB) if it isn't there yet. Needs Internet."""
    import urllib.request
    import zipfile
    if VOSK_DIR.exists():
        return
    VOSK_DIR.parent.mkdir(parents=True, exist_ok=True)
    archive = VOSK_DIR.parent / f"{VOSK_MODEL}.zip"
    urllib.request.urlretrieve(VOSK_URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(VOSK_DIR.parent)
    archive.unlink()  # no need to keep the zip once it's extracted


def transcribe_offline(audio):
    """Speech recognition on the PC, no Internet needed (Vosk). Less accurate than Google."""
    global _vosk
    if not VOSK_DIR.exists():
        return None, "Pas d'Internet, et la reconnaissance hors ligne n'est pas installée (python installer.py)."
    try:
        from vosk import KaldiRecognizer, Model, SetLogLevel
        if _vosk is None:
            SetLogLevel(-1)  # Vosk is very chatty by default
            _vosk = Model(str(VOSK_DIR))
        recognizer = KaldiRecognizer(_vosk, SAMPLE_RATE)
        recognizer.AcceptWaveform(audio.tobytes())
        text = json.loads(recognizer.FinalResult()).get("text", "").strip()
    except Exception as e:
        return None, f"Reconnaissance vocale indisponible : {e}"
    return (text, None) if text else (None, "Je n'ai pas compris. Réessayez.")


def listen_voice():
    """Listen until the end of the sentence, then transcribe (🎤 button, terminal mode).
    Returns (text, None) if it worked, (None, error message) otherwise."""
    try:
        frames = queue.Queue()
        with mic_stream(frames):
            audio = record_until_silence(read_frames(frames))
    except Exception as e:
        return None, f"Micro indisponible : {e}"
    return transcribe(audio)


# --- The model ---

def warm_up(history=(), tools=(), on_error=None, on_status=None):
    """Get the models ready in the background as soon as Jarvis starts.

    0. Check that Ollama is running (and start it if needed); otherwise call on_error(message).
    1. Loading the model isn't enough. Before answering, it has to "read" the whole start of
       the conversation (instructions, tool descriptions, history), which takes ~50 s on a CPU.
       So we make it read that now, with a dummy question whose answer we throw away (it only
       generates one word). Ollama keeps that reading cached, so your first real question only
       has to read the end."""
    def load():
        problem = ensure_ollama(on_status)
        if problem:
            if on_error:
                on_error(problem)
            return
        try:
            messages = build_prompt(list(history), "Bonjour")[:-1]  # no memories: just the shared start
            client.chat(model=MODEL, messages=messages + [{"role": "user", "content": "Bonjour"}],
                        tools=list(tools), keep_alive=KEEP_ALIVE, options={"num_predict": 1},
                        **({} if THINK is None else {"think": THINK}))
        except Exception:
            pass  # the error will show up on the first real question
        get_memory()  # also loads the embedding model and indexes what's already there
    thread = threading.Thread(target=load, daemon=True)
    thread.start()
    return thread  # thread.join() lets you wait until it's ready


def build_prompt(history, user_input):
    """Put together everything the model sees:
    instructions + facts + recent conversation + (relevant memories + the new question).

    The order matters for speed. Ollama caches the part of the conversation it has already
    read and only re-reads what changed. So everything that changes with each question
    (the memories we pull back) goes at the very end, in the last message. When they sat in
    the instructions at the top, it had to re-read everything: ~40 s instead of ~10 s."""
    system = JARVIS_PERSONA
    facts = load_facts()[-MAX_FACTS:]
    if facts:
        system += f"\n\nCe que tu sais sur {USER_REF}:\n" + "".join(f"- {f}\n" for f in facts)

    # Old memories related to the question, except the ones the model can already see
    # (exchanges still in the recent conversation, facts already listed above)
    window = recent_window(history)
    recent = {m["content"] for m in window if m["role"] == "user"}
    memories = remember(lambda memory: memory.search(
        user_input, MAX_MEMORIES,
        skip=lambda item: item.get("user") in recent or (item["kind"] == "fait" and item["text"] in facts)))
    # Last message: today's date (+ memories) + the request. The date changes every minute;
    # putting it here at the end means it doesn't force a re-read of everything above.
    content = f"[Nous sommes le {now_text()}]\n"
    if memories:
        content += ("[Souvenirs de conversations passées, à utiliser seulement s'ils sont utiles :]\n"
                    + "".join(f"- [{m['date'][:10] or 'date inconnue'}] {m['text'][:400]}\n" for m in memories))
    content += f"[Demande de {USER_REF} :]\n{user_input}"

    return [{"role": "system", "content": system}, *window, {"role": "user", "content": content}]


def run_tool(tools, call):
    """Run the tool the model asked for. Any error goes back to the model as text
    (so it can explain it to the user) instead of crashing Jarvis."""
    func = tools.get(call.function.name)
    if not func:
        return f"Outil inconnu : {call.function.name}"
    try:
        return str(func(**(call.function.arguments or {})))
    except Exception as e:
        return f"Erreur de l'outil {call.function.name} : {e}"


def ask_model(history, user_input, tools=(), on_token=None, on_status=None):
    """Ask the model and return its final answer.

    How it goes:
    1. the model gets the question and the list of tools;
    2. if it asks for a tool, we run it and send back the result;
    3. repeat until it answers with text (at most MAX_TOOL_ROUNDS times);
    4. the exchange is added to the history and to long-term memory.

    on_token(text)  : called for each piece of the answer (live display)
    on_status(text) : called to say what's going on ("🌐 Recherche web...")
    """
    by_name = {f.__name__: f for f in tools}
    messages = build_prompt(history, user_input)
    reply = ""
    tool_steps = []  # tool requests + results from this turn, kept in the history

    # The small model sometimes sends back a completely empty answer (no text, no tool).
    # Only in that case do we ask it a second time (there's some randomness in it).
    for attempt in range(2):
        for _ in range(MAX_TOOL_ROUNDS):
            content, calls = "", []
            # stream=True: the answer comes in piece by piece instead of all at the end
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
                break  # final text answer: we're done
            # Keep a record of the tool request and its result. The model relies on it in the
            # next round, and the history keeps it as an example for next time
            # (plain dicts, so it can be saved to memory.json)
            step = [{"role": "assistant", "content": content,
                     "tool_calls": [{"function": {"name": c.function.name, "arguments": c.function.arguments or {}}}
                                    for c in calls]}]
            for call in calls:
                if on_status:
                    on_status(getattr(by_name.get(call.function.name), "label", f"⚙️ {call.function.name}..."))
                step.append({"role": "tool", "tool_name": call.function.name, "content": run_tool(by_name, call)})
            messages += step  # the model sees the full result for this answer...
            # ...but the history only keeps an excerpt: a full web search would weigh down
            # (and slow down) every question after it
            tool_steps += [dict(m, content=m["content"][:MAX_TOOL_RESULT]) if m["role"] == "tool" else m
                           for m in step]
            if on_status:
                on_status("Jarvis réfléchit...")
        if reply.strip() or tool_steps:
            break

    reply = reply.strip()
    if not reply:
        # Empty answer. If a tool ran, the model did the thing without commenting; otherwise
        # it simply failed, and saying "C'est fait" would be a lie.
        reply = "C'est fait." if tool_steps else "Je n'ai pas bien saisi. Pouvez-vous reformuler ?"
        if on_token:
            on_token(reply)
    history.append({"role": "user", "content": user_input})
    history.extend(tool_steps)
    history.append({"role": "assistant", "content": reply})
    save_history(history)
    # Long-term memory only gets real conversations. An exchange that used a tool (weather,
    # time, web, opening an app...) is either stale info or just a command. When those came
    # back as "memories", Jarvis would repeat yesterday's weather without calling the tool.
    # (Facts saved with "memoriser" are stored separately.)
    if not tool_steps:
        remember(lambda memory: memory.add_exchange(user_input, reply))
    return reply
