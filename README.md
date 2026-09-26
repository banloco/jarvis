# J.A.R.V.I.S — assistant personnel façon Iron Man

Un assistant vocal pour Windows, inspiré de l'IA de Tony Stark : on lui parle (« Hey Jarvis »),
il répond à voix haute et agit sur le PC. Il se pilote aussi **à la main devant la webcam**,
avec une interface holographique plein écran.

**Local et gratuit** : le cerveau est un petit modèle de langage qui tourne sur ton PC
(via [Ollama](https://ollama.com)), **sans carte graphique** et sans aucune API payante.
Seules la reconnaissance vocale Google, la voix, la météo et la recherche web passent par
Internet, avec des solutions de secours hors ligne pour la voix et l'écoute.

Le code est commenté en français, pour pouvoir apprendre en le lisant.

## Ce qu'il sait faire

- **Conversation** en français, avec une mémoire : conversation récente, faits retenus
  (« retiens que... ») et souvenirs anciens retrouvés par le sens.
- **Réveil vocal** « Hey Jarvis », mode conversation (enchaîner sans le redire), calibrage
  sur ta voix.
- **23 outils** : recherche web, météo et prévisions, heure, briefing du jour, état du PC,
  calcul exact, fichiers (Documents, Téléchargements, Bureau), ouvrir / fermer une appli,
  ouvrir un site, musique (YouTube, commandes lecture / suivant), minuteurs et rappels,
  volume, verrouillage.
- **Gestes de la main** (webcam) : gestes appris sur tes exemples (poing, pouce levé...) et
  mouvements (glisser, pincer pour régler le volume).
- **Interface holographique** : ta webcam en fond, des panneaux lumineux (météo, système,
  rappels, musique, volume, applis) que tu pilotes en pinçant pouce et index.
- **Fenêtre style Iron Man** avec réacteur arc animé, icône près de l'horloge, démarrage avec
  Windows et briefing au premier lancement de la journée.

## Installation

Prérequis : **Windows 10 ou 11**, **Python 3.14** (testé), un micro ; une webcam pour les gestes.

```powershell
git clone https://github.com/banloco/jarvis.git
cd jarvis
python -m pip install -r requirements.txt
```

Installe ensuite [Ollama](https://ollama.com), puis les deux modèles :

```powershell
ollama pull qwen2.5:3b
ollama pull paraphrase-multilingual
```

Enfin (facultatif) : raccourci sur le Bureau, démarrage avec Windows et reconnaissance vocale
hors ligne (~41 Mo) :

```powershell
python installer.py            # python installer.py --retirer pour tout enlever
```

### Ton prénom

Crée un fichier `config.json` à côté du code (modèle : `config.example.json`) :

```json
{"prenom": "Tony"}
```

Sans ce fichier, Jarvis dit simplement « Bonjour » et t'appelle « Vous » dans le chat.

## Utilisation

| Commande | Rôle |
|---|---|
| `python jarvis_ui.py` | La fenêtre (conseillé) |
| `python jarvis.py` | Mode terminal, plus simple |
| `python jarvis_holo.py` | L'interface holographique seule (`--fenetre` pour ne pas être en plein écran) |
| `python jarvis_wake.py calibrer` | Adapte le réveil « Hey Jarvis » à ta voix |
| `python jarvis_gestures.py` | Collecter / entraîner / tester tes gestes (détails dans le fichier) |

Dans la fenêtre : tape ta question, clique sur 🎤, ou dis « Hey Jarvis ». Les boutons
**ÉCOUTE**, **GESTES** et **INTERFACE HOLO** activent le réveil vocal, les gestes et
l'interface holographique.

**Interface holographique** : le curseur suit le point entre ton pouce et ton index.
Pincer = cliquer ; pincer un panneau puis bouger = le déplacer ; pincer le réacteur au
centre = parler à Jarvis ; Échap = quitter.

## Tes données restent chez toi

Mémoire, faits, rappels, exemples de gestes, calibrage de la voix et `config.json` sont
enregistrés dans le dossier du projet et **ignorés par Git** (voir `.gitignore`) : ils ne
partent jamais sur GitHub.

## Organisation du code

| Fichier | Rôle |
|---|---|
| `jarvis_core.py` | Réglages, mémoire, micro, reconnaissance vocale, dialogue avec le modèle |
| `jarvis_tools.py` | Les outils que le modèle peut appeler (ajouter un outil : voir l'en-tête du fichier) |
| `jarvis_memory.py` | Mémoire à long terme par embeddings |
| `jarvis_reminders.py` | Minuteurs et rappels |
| `jarvis_voice.py` | Voix (edge-tts, secours hors ligne) |
| `jarvis_wake.py` | Réveil « Hey Jarvis » et mode conversation |
| `jarvis_gestures.py` | Gestes de la main (MediaPipe + petit réseau de neurones) |
| `jarvis_holo.py` | Interface holographique pilotée à la main |
| `jarvis_ui.py`, `jarvis_hud.py` | Fenêtre et éléments visuels |
| `jarvis_config.py` | Réglages personnels (`config.json`) |
| `installer.py` | Raccourcis et modèle hors ligne |

Les commentaires en haut de chaque fichier expliquent les choix techniques et les pièges
déjà rencontrés : à lire avant de modifier le code.

## Contribuer

Les contributions sont bienvenues : voir [CONTRIBUTING.md](CONTRIBUTING.md).

## Remerciements

[Ollama](https://ollama.com) et [Qwen 2.5](https://github.com/QwenLM/Qwen2.5) ·
[MediaPipe](https://github.com/google-ai-edge/mediapipe) ·
[openWakeWord](https://github.com/dscripka/openWakeWord) · [Vosk](https://alphacephei.com/vosk/) ·
[edge-tts](https://github.com/rany2/edge-tts) · [wttr.in](https://wttr.in) ·
[HaGRID](https://github.com/hukenovs/hagrid) (CC BY-SA 4.0, import facultatif).

## Licence

[MIT](LICENSE) : libre d'utiliser, modifier et redistribuer, en gardant la mention de licence.
