"""Jarvis's long-term memory.

The idea:
- Every exchange (question + answer) and every fact gets archived, with no limit.
- For each text, a small model (EMBED_MODEL) computes a "meaning fingerprint": a vector of
  numbers. Two texts about the same thing end up with close vectors, even when they don't
  share any words ("my dog" ≈ "Rex, my German shepherd").
- For each new question we compute its vector and bring back the closest memories.

Files created next to the code:
- long_term.json          : the memories as text (readable, you can edit it by hand)
- long_term.<model>.npy   : the matching vectors (binary, can always be recomputed)
"""
import json
import threading
import numpy as np
import ollama
from datetime import datetime
from jarvis_config import USER_LABEL

# A multilingual model: it tells related and unrelated French sentences apart nicely.
# (We tried nomic-embed-text first; in French it gave almost the same score to everything.)
EMBED_MODEL = "paraphrase-multilingual"
MIN_SCORE = 0.4    # minimum similarity (-1 to 1) for a memory to count as relevant
BATCH = 32         # texts sent to Ollama per call
client = ollama.Client(timeout=60)  # with a timeout, so a stuck Ollama can't freeze Jarvis


class LongTermMemory:
    def __init__(self, base_dir, keep_alive="30m"):
        # Texts (readable) and vectors (binary) are stored separately.
        # The vector file is named after the model, so switching models forces a recompute.
        self.items_file = base_dir / "long_term.json"
        self.vectors_file = base_dir / f"long_term.{EMBED_MODEL}.npy"
        self.keep_alive = keep_alive
        self.lock = threading.Lock()  # add() can be called from several threads
        try:
            self.items = json.loads(self.items_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.items = []
        try:
            self.vectors = np.load(self.vectors_file)
        except (FileNotFoundError, ValueError):
            self.vectors = None
        # Vectors missing or out of sync (a crash, a new model): recompute them
        if self.items and (self.vectors is None or len(self.vectors) != len(self.items)):
            self.vectors = self._embed([i["text"] for i in self.items])
            self._save()

    def _embed(self, texts):
        """Turn texts into normalized vectors (one row per text).
        Normalized means length 1, so a plain dot product gives the similarity."""
        chunks = []
        for start in range(0, len(texts), BATCH):
            batch = texts[start:start + BATCH]
            resp = client.embed(model=EMBED_MODEL, input=batch, keep_alive=self.keep_alive)
            chunks.append(np.array(resp["embeddings"], dtype=np.float32))
        vectors = np.vstack(chunks)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def _save(self):
        # Written through a temp file, so a crash never corrupts the archive
        tmp = self.items_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.items_file)
        with open(self.vectors_file, "wb") as f:
            np.save(f, self.vectors)

    def add_many(self, entries):
        """Archive several memories at once.
        entries: list of dicts {"text", "kind" ("fait" or "conversation"), "date", ...}."""
        if not entries:
            return
        vectors = self._embed([e["text"] for e in entries])  # outside the lock: this is the slow part
        with self.lock:
            self.items.extend(entries)
            self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
            self._save()

    def add(self, text, kind, **extra):
        """Archive one memory dated now. extra: any additional fields."""
        self.add_many([{"text": text, "kind": kind, "date": datetime.now().isoformat(timespec="minutes"), **extra}])

    def add_exchange(self, user_input, reply):
        """Archive an exchange. The "user" field lets us avoid bringing back an exchange
        that's already in the current conversation."""
        self.add(f"{USER_LABEL} : {user_input}\nJarvis : {reply}", "conversation", user=user_input)

    def search(self, query, k=5, skip=lambda item: False):
        """Return up to k memories close to the question, most relevant first.
        skip(item) -> True to leave a memory out (e.g. it's already shown elsewhere)."""
        if self.vectors is None:
            return []
        # Similarity between the question and every memory, in a single operation
        scores = self.vectors @ self._embed([query])[0]
        results = []
        for idx in np.argsort(scores)[::-1]:  # best score first
            if scores[idx] < MIN_SCORE or len(results) >= k:
                break
            if not skip(self.items[idx]):
                results.append(self.items[idx])
        return results

    def import_existing(self, facts, history):
        """On the very first run: archive the facts (facts.json) and conversations
        (memory.json) that were saved before this memory existed."""
        if self.items:
            return
        entries = [{"text": f["content"], "kind": "fait", "date": f.get("date", "")[:16]} for f in facts]
        # Rebuild the question / answer pairs from the history
        for user, assistant in zip(history, history[1:]):
            if user["role"] == "user" and assistant["role"] == "assistant":
                entries.append({"text": f"{USER_LABEL} : {user['content']}\nJarvis : {assistant['content']}",
                                "kind": "conversation", "date": "", "user": user["content"]})
        self.add_many(entries)
