# Contribuer à Jarvis

Merci de ton aide ! Quelques règles simples pour que tout le monde s'y retrouve.

## Avant de commencer

1. Lis le README et les commentaires en haut des fichiers que tu modifies : ils expliquent
   l'architecture et les choix techniques (beaucoup d'essais ont déjà été faits et mesurés).
2. Pour une grosse modification, ouvre d'abord une **issue** pour en discuter.

## Façon de travailler

- Crée une **branche** à partir de `main` (`git switch -c ma-fonction`), puis ouvre une
  **pull request**. On ne pousse pas directement sur `main`.
- **Français** partout : commentaires, messages affichés, messages de commit.
- Code **commenté** et simple : le projet sert aussi à apprendre.
- **Gratuit et local** : pas d'API payante ni de service qui demande une clé.
- **Aucune donnée personnelle** dans les commits : prénom, ville, mémoire, exemples de
  gestes... restent dans les fichiers ignorés (`config.json`, `memory.json`, etc.).
  Vérifie `git status` avant chaque commit.

## Tester

- **Ne touche jamais à tes vraies données pendant un test** : redirige `core.BASE_DIR`,
  `core.MEMORY_FILE`, `core.FACTS_FILE`, `jarvis_ui.STATE_FILE` et `tools.reminders` vers un
  dossier temporaire : `ask_model` écrit dans l'historique et dans la mémoire à long terme.
- **Mesure avant de conclure** : temps de réponse, précision, images par seconde. Donne les
  chiffres dans la pull request, même décevants.
- Décris dans la pull request ce que tu as vérifié en vrai (micro, webcam, fenêtre).

## Ajouter un outil

Voir l'en-tête de `jarvis_tools.py` : une fonction typée, une docstring claire qui dit
**quand** l'utiliser (et quand ne pas l'utiliser), et le décorateur `@tool("libellé...")`.
