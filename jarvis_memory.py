"""Mémoire à long terme de Jarvis.

Principe :
- Chaque échange (question + réponse) et chaque fait est archivé, sans limite.
- Pour chaque texte, un petit modèle (EMBED_MODEL) calcule une « empreinte de sens » :
  un vecteur de nombres. Deux textes qui parlent de la même chose ont des vecteurs proches,
  même s'ils n'utilisent pas les mêmes mots (« mon chien » ≈ « Rex, mon berger allemand »).
- À chaque question, on calcule son vecteur et on ressort les souvenirs les plus proches.

Fichiers créés à côté du code :
- long_term.json                    : les souvenirs en texte (lisible, modifiable à la main)
- long_term.<modèle>.npy            : les vecteurs correspondants (binaire, recalculable)
"""
import json
import threading
import numpy as np
import ollama
from datetime import datetime
from jarvis_config import USER_LABEL

# Modèle multilingue : il sépare bien les phrases françaises liées / sans rapport
# (nomic-embed-text, testé, donnait des scores presque identiques partout en français).
EMBED_MODEL = "paraphrase-multilingual"
MIN_SCORE = 0.4    # similarité minimale (-1 à 1) pour qu'un souvenir soit jugé pertinent
BATCH = 32         # textes envoyés à Ollama par appel
client = ollama.Client(timeout=60)  # délai maximum : un Ollama bloqué ne doit pas figer Jarvis


class LongTermMemory:
    def __init__(self, base_dir, keep_alive="30m"):
        # Les textes (lisibles) et les vecteurs (binaires) sont stockés séparément.
        # Le fichier de vecteurs porte le nom du modèle : en changer force un recalcul.
        self.items_file = base_dir / "long_term.json"
        self.vectors_file = base_dir / f"long_term.{EMBED_MODEL}.npy"
        self.keep_alive = keep_alive
        self.lock = threading.Lock()  # add() peut être appelé depuis plusieurs threads
        try:
            self.items = json.loads(self.items_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.items = []
        try:
            self.vectors = np.load(self.vectors_file)
        except (FileNotFoundError, ValueError):
            self.vectors = None
        # Vecteurs absents ou désynchronisés (crash, changement de modèle) : on recalcule
        if self.items and (self.vectors is None or len(self.vectors) != len(self.items)):
            self.vectors = self._embed([i["text"] for i in self.items])
            self._save()

    def _embed(self, texts):
        """Transforme des textes en vecteurs normalisés (une ligne par texte).
        Normalisés = de longueur 1, donc un simple produit scalaire donne la similarité."""
        chunks = []
        for start in range(0, len(texts), BATCH):
            batch = texts[start:start + BATCH]
            resp = client.embed(model=EMBED_MODEL, input=batch, keep_alive=self.keep_alive)
            chunks.append(np.array(resp["embeddings"], dtype=np.float32))
        vectors = np.vstack(chunks)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def _save(self):
        # Écriture via un fichier temporaire : un crash ne corrompt jamais l'archive
        tmp = self.items_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.items_file)
        with open(self.vectors_file, "wb") as f:
            np.save(f, self.vectors)

    def add_many(self, entries):
        """Archive plusieurs souvenirs d'un coup.
        entries : liste de dicts {"text", "kind" ("fait" ou "conversation"), "date", ...}."""
        if not entries:
            return
        vectors = self._embed([e["text"] for e in entries])  # hors verrou : c'est l'étape lente
        with self.lock:
            self.items.extend(entries)
            self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
            self._save()

    def add(self, text, kind, **extra):
        """Archive un souvenir daté de maintenant. extra : champs libres en plus."""
        self.add_many([{"text": text, "kind": kind, "date": datetime.now().isoformat(timespec="minutes"), **extra}])

    def add_exchange(self, user_input, reply):
        """Archive un échange. Le champ "user" sert à ne pas ressortir un échange
        qui est déjà dans la conversation en cours."""
        self.add(f"{USER_LABEL} : {user_input}\nJarvis : {reply}", "conversation", user=user_input)

    def search(self, query, k=5, skip=lambda item: False):
        """Retourne jusqu'à k souvenirs proches de la question, du plus au moins pertinent.
        skip(item) -> True pour ignorer un souvenir (ex : déjà affiché ailleurs)."""
        if self.vectors is None:
            return []
        # Similarité entre la question et chaque souvenir, en une seule opération
        scores = self.vectors @ self._embed([query])[0]
        results = []
        for idx in np.argsort(scores)[::-1]:  # du meilleur score au moins bon
            if scores[idx] < MIN_SCORE or len(results) >= k:
                break
            if not skip(self.items[idx]):
                results.append(self.items[idx])
        return results

    def import_existing(self, facts, history):
        """Au tout premier lancement : archive les faits (facts.json) et les
        conversations (memory.json) enregistrés avant l'existence de cette mémoire."""
        if self.items:
            return
        entries = [{"text": f["content"], "kind": "fait", "date": f.get("date", "")[:16]} for f in facts]
        # On reforme les paires question / réponse de l'historique
        for user, assistant in zip(history, history[1:]):
            if user["role"] == "user" and assistant["role"] == "assistant":
                entries.append({"text": f"{USER_LABEL} : {user['content']}\nJarvis : {assistant['content']}",
                                "kind": "conversation", "date": "", "user": user["content"]})
        self.add_many(entries)
