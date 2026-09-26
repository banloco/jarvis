# J.A.R.V.I.S

My own take on Tony Stark's assistant, running on a regular Windows PC.

You talk to it ("Hey Jarvis"), it answers out loud and does things on your computer: checks
the weather, opens apps, plays music, sets reminders, reads your files, controls the volume.
You can also drive it with your hand in front of the webcam, through a full-screen
holographic display that's honestly the most fun part of the project.

I started this to learn, so the code is heavily commented. If you want to understand how a
voice assistant works under the hood (speech recognition, a local language model calling
tools, memory, hand tracking), reading it from top to bottom should get you there.

**A heads-up before you dive in: Jarvis speaks French.** His voice, his answers and the
window are in French. The code, comments and docs are in English. Making him bilingual is
on the wish list, and help is very welcome.

## What makes it different

- **Everything that matters runs on your machine.** The brain is a small language model
  ([Qwen 2.5](https://github.com/QwenLM/Qwen2.5), 3B) served by [Ollama](https://ollama.com).
  No paid API, no account, no key.
- **No graphics card needed.** It was built and tuned on a CPU-only laptop with 16 GB of RAM.
  A lot of the code exists just to make that fast enough; the comments explain the tricks.
- **Only a few things go online**: Google speech recognition, the voice (edge-tts), the
  weather and web search. Voice and listening both fall back to offline versions when
  there's no Internet.

## What it can do

- **Talk** with a memory: the recent conversation, facts you ask it to remember
  ("retiens que..."), and older memories it pulls back up when they're relevant.
- **Wake word** "Hey Jarvis", plus a conversation mode where you can keep going without
  saying it again. You can calibrate it to your voice.
- **23 tools**: web search, weather and forecast, time, a daily briefing, PC status, exact
  math, files (Documents, Downloads, Desktop), open and close apps, open websites, music
  (YouTube, play / pause / next), timers and reminders, volume, locking the PC.
- **Hand gestures**: poses you train on your own examples (fist, thumbs up...) and moves
  that work out of the box (swipe, pinch to change the volume).
- **Holographic display**: your webcam feed with glowing panels on top (weather, system,
  reminders, music, volume, apps). Point with your thumb and index, pinch to click.
- **An Iron Man style window** with an animated arc reactor, a tray icon, start with Windows,
  and a briefing the first time you open it each day.

## Getting started

You'll need **Windows 10 or 11**, **Python 3.14** (that's what I use), a microphone, and a
webcam if you want the gesture features.

```powershell
git clone https://github.com/banloco/jarvis.git
cd jarvis
python -m pip install -r requirements.txt
```

Then install [Ollama](https://ollama.com) and pull the two models:

```powershell
ollama pull qwen2.5:3b
ollama pull paraphrase-multilingual
```

Optional, but nice: a desktop shortcut, starting with Windows, and offline speech
recognition (~41 MB):

```powershell
python installer.py            # python installer.py --retirer undoes it
```

### Tell Jarvis your name

Create a `config.json` next to the code (there's a `config.example.json` to copy):

```json
{"name": "Tony"}
```

Without it, Jarvis just says "Bonjour" and shows your messages as "Vous".

## Using it

| Command | What it does |
|---|---|
| `python jarvis_ui.py` | The window (start here) |
| `python jarvis.py` | Terminal mode, simpler |
| `python jarvis_holo.py` | The holographic display on its own (`--fenetre` to keep it in a window) |
| `python jarvis_wake.py calibrer` | Tunes "Hey Jarvis" to your voice |
| `python jarvis_gestures.py` | Record, train and test your own gestures (details at the top of the file) |

In the window, type your question, click 🎤, or just say "Hey Jarvis". The **ÉCOUTE**,
**GESTES** and **INTERFACE HOLO** buttons turn on the wake word, gestures and the
holographic display.

In the holographic display, the cursor sits between your thumb and index finger. Pinch to
click, pinch a panel and move to drag it around, pinch the reactor in the middle to talk
to Jarvis, and press Esc to leave.

## Your data stays yours

Memories, facts, reminders, gesture examples, voice calibration and `config.json` are saved
in the project folder and **ignored by Git** (see `.gitignore`). They never end up on GitHub.

## Finding your way around the code

| File | What's in it |
|---|---|
| `jarvis_core.py` | Settings, memory, microphone, speech recognition, talking to the model |
| `jarvis_tools.py` | The tools the model can call (how to add one is explained at the top) |
| `jarvis_memory.py` | Long-term memory with embeddings |
| `jarvis_reminders.py` | Timers and reminders |
| `jarvis_voice.py` | The voice (edge-tts, offline fallback) |
| `jarvis_wake.py` | "Hey Jarvis" and conversation mode |
| `jarvis_gestures.py` | Hand gestures (MediaPipe + a small neural network) |
| `jarvis_holo.py` | The holographic display |
| `jarvis_ui.py`, `jarvis_hud.py` | The window and its visuals |
| `jarvis_config.py` | Your personal settings (`config.json`) |
| `installer.py` | Shortcuts and the offline model |

The comments at the top of each file explain the technical decisions, and quite a few of
them mention traps I've already fallen into. Please read them before changing things; it'll
save you some of the evenings it cost me.

## Want to help?

Yes please. Start with [CONTRIBUTING.md](CONTRIBUTING.md). Good first ideas: English support,
Mac/Linux support (volume and shortcuts are Windows-only for now), interrupting Jarvis by
voice while he's talking, and anything in the issues.

## Thanks

This stands on the shoulders of some great free projects:
[Ollama](https://ollama.com) and [Qwen 2.5](https://github.com/QwenLM/Qwen2.5) ·
[MediaPipe](https://github.com/google-ai-edge/mediapipe) ·
[openWakeWord](https://github.com/dscripka/openWakeWord) · [Vosk](https://alphacephei.com/vosk/) ·
[edge-tts](https://github.com/rany2/edge-tts) · [wttr.in](https://wttr.in) ·
[HaGRID](https://github.com/hukenovs/hagrid) (CC BY-SA 4.0, optional import).

## License

[MIT](LICENSE). Use it, change it, share it; just keep the license notice.
