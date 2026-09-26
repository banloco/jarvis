"""Jarvis en mode terminal. Lancer : python jarvis.py

Entrée vide = parler au micro ; sinon taper son message. « exit » pour quitter.
Toute l'intelligence est dans jarvis_core.py ; ce fichier ne gère que le dialogue.
"""
from rich.console import Console
from rich.panel import Panel
from jarvis_core import load_history, listen_voice, ask_model, warm_up
from jarvis_voice import Speaker
from jarvis_tools import TOOLS

console = Console()   # affichage coloré dans le terminal
speaker = Speaker()


def speak(text):
    """Affiche la réponse dans un cadre puis la prononce."""
    console.print(Panel(text, title="[cyan]Jarvis[/cyan]", border_style="cyan"))
    speaker.say(text, wait=True)  # attendre la fin pour ne pas s'enregistrer soi-même


def listen():
    """Écoute le micro et retourne le texte compris, ou None."""
    console.print("[cyan]Jarvis écoute...[/cyan]")
    text, error = listen_voice()
    if text:
        console.print(f"[green]l'utilisateur > {text}[/green]")
    else:
        console.print(f"[dim]{error}[/dim]")
    return text


def respond(history, user_input):
    """Interroge Jarvis avec une animation d'attente ; les erreurs deviennent des réponses."""
    try:
        with console.status("[cyan]Jarvis réfléchit...[/cyan]") as status:
            return ask_model(history, user_input, TOOLS, on_status=lambda text: status.update(f"[cyan]{text}[/cyan]"))
    except ConnectionError:
        return "Ollama ne répond pas. Lancez-le avec `ollama serve`."
    except Exception as e:
        return f"Erreur : {e}"


def main():
    history = load_history()
    # Prépare le modèle pendant l'accueil (et lance Ollama s'il ne tourne pas)
    warm_up(history, TOOLS, on_error=lambda msg: console.print(f"[red]{msg}[/red]"))
    console.print(Panel("JARVIS — Système vocal en ligne", style="bold cyan"))
    speak("Jarvis en ligne. Je vous écoute.")

    while True:
        try:
            console.print("\n[dim]Appuyez sur Entrée pour parler ou tapez votre message :[/dim]")
            mode = input().strip()

            if mode.lower() in ["exit", "quit", "bye"]:
                speak("Système en veille. À bientôt.")
                break

            user_input = mode or listen()  # rien tapé -> on écoute le micro
            if user_input:
                speak(respond(history, user_input))

        except (KeyboardInterrupt, EOFError):
            speak("Arrêt forcé. Mémoire sauvegardée.")
            break


if __name__ == "__main__":
    main()
