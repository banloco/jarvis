"""Jarvis in terminal mode. Run: python jarvis.py

Press Enter on an empty line to talk into the mic, or just type your message. "exit" quits.
All the smarts are in jarvis_core.py; this file only handles the back-and-forth.
"""
from rich.console import Console
from rich.panel import Panel
from jarvis_config import USER_LABEL
from jarvis_core import load_history, listen_voice, ask_model, warm_up
from jarvis_voice import Speaker
from jarvis_tools import TOOLS
from jarvis_reminders import reminders

console = Console()   # colored output in the terminal
speaker = Speaker()


def speak(text):
    """Show the answer in a box, then say it out loud."""
    console.print(Panel(text, title="[cyan]Jarvis[/cyan]", border_style="cyan"))
    speaker.say(text, wait=True)  # wait until he's done, so the mic doesn't record him


def listen():
    """Listen to the mic and return what was understood, or None."""
    console.print("[cyan]Jarvis écoute...[/cyan]")
    text, error = listen_voice()
    if text:
        console.print(f"[green]{USER_LABEL} > {text}[/green]")
    else:
        console.print(f"[dim]{error}[/dim]")
    return text


def respond(history, user_input):
    """Ask Jarvis while showing a spinner. Errors are turned into answers."""
    try:
        with console.status("[cyan]Jarvis réfléchit...[/cyan]") as status:
            return ask_model(history, user_input, TOOLS, on_status=lambda text: status.update(f"[cyan]{text}[/cyan]"))
    except ConnectionError:
        return "Ollama ne répond pas. Lancez-le avec `ollama serve`."
    except Exception as e:
        return f"Erreur : {e}"


def main():
    history = load_history()
    # Warm up the model during the greeting (and start Ollama if it isn't running)
    warm_up(history, TOOLS, on_error=lambda msg: console.print(f"[red]{msg}[/red]"))
    console.print(Panel("JARVIS — Système vocal en ligne", style="bold cyan"))
    speak("Jarvis en ligne. Je vous écoute.")
    # Reminders get announced in the terminal and out loud, even while you're typing
    reminders.on_due = lambda message, late: speak(
        f"Rappel manqué (il y a {late} min) : {message}" if late else f"Rappel : {message}")
    reminders.start()

    while True:
        try:
            console.print("\n[dim]Appuyez sur Entrée pour parler ou tapez votre message :[/dim]")
            mode = input().strip()

            if mode.lower() in ["exit", "quit", "bye"]:
                speak("Système en veille. À bientôt.")
                break

            user_input = mode or listen()  # nothing typed -> listen to the mic
            if user_input:
                speak(respond(history, user_input))

        except (KeyboardInterrupt, EOFError):
            speak("Arrêt forcé. Mémoire sauvegardée.")
            break


if __name__ == "__main__":
    main()
