# graph/rag/rag_network.py
import os
import json
import urllib.request

def fetch_ollama_vector(text: str) -> list:
    """Sends prompt coordinates straight to nomic-embed-text."""
    payload = {
        "model": "nomic-embed-text:latest",
        "prompt": text,
        "options": {"keep_alive": 0}  # VRAM safe drop
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["embedding"]
    except Exception:
        return []
