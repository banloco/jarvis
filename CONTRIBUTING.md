# Contributing to Jarvis

First off, thanks for wanting to help. Here's how we keep things smooth.

## Before you start

1. Skim the README and the comments at the top of the files you'll touch. They explain
   how everything fits together, and a lot of things have already been tried and measured.
2. For anything big, open an **issue** first so we can talk it through.

## How we work

- Create a **branch** from `main` (`git switch -c my-feature`) and open a **pull request**.
  Nobody pushes straight to `main`.
- **Code, comments, docs and commit messages are in English.** Jarvis himself speaks French,
  so anything the user sees or hears stays in French for now (that includes the system
  prompt and the tool docstrings, which the model reads).
- Keep the code **simple and commented**. People read this project to learn.
- **Free and local**: no paid APIs, no services that need a key.
- **No personal data in commits**: names, cities, memories, gesture examples... all stay in
  ignored files (`config.json`, `memory.json`, etc.). Check `git status` before committing.

## Testing

- **Never touch your real data while testing.** Point `core.BASE_DIR`, `core.MEMORY_FILE`,
  `core.FACTS_FILE`, `jarvis_ui.STATE_FILE` and `tools.reminders` at a temp folder:
  `ask_model` writes to both the history and long-term memory.
- To test the window without the model, stub it out: `jarvis_ui.warm_up = lambda *a, **k: None`.
- **Measure before you conclude**: response time, accuracy, frames per second. Put the
  numbers in your pull request, even if they're not great.
- In the pull request, say what you actually tried for real (mic, webcam, window).
- If you touch a tool docstring, re-test how the model uses that tool: those docstrings
  are part of the prompt.

## Adding a tool

See the top of `jarvis_tools.py`: a function with typed parameters, a clear docstring that
says **when** to use it (and when not to), and the `@tool("label...")` decorator.
