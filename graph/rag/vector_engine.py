# graph/rag/vector_engine.py
import os
import pickle
from .rag_parser import RagDocumentParser
from .rag_network import fetch_ollama_vector
from .rag_math import compute_top_k  # Note: matches your rag_maths file name

class LocalVectorDB:
    def __init__(self):
        # Read absolute target path from .env with a secure fallback
        self.db_path = os.getenv("DB_PATH", r"E:\VectorDB\index.bin")
        self.file_parser = RagDocumentParser()
        
        # Ensure directory structure exists safely
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.registry = self._load_database()

    def _load_database(self):
        if os.path.exists(self.db_path) and os.path.getsize(self.db_path) > 0:
            try:
                with open(self.db_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[RAG Engine] Binary parse warning, starting fresh: {e}")
                return []
        return []

    def _save_database(self):
        try:
            with open(self.db_path, "wb") as f:
                pickle.dump(self.registry, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[RAG Engine] Failed writing binary cache to disk: {e}")

    def ingest_local_file(self, file_path: str):
        chunks = self.file_parser.process_file_to_chunks(file_path)
        if not chunks: 
            return

        # Core Deduplication Guard Layer: Hash existing texts to skip API overhead entirely
        existing_contents = {record["content"] for record in self.registry}
        new_entries = []

        for chunk in chunks:
            clean_chunk = chunk.strip()
            if not clean_chunk or clean_chunk in existing_contents: 
                continue
            
            # Request local text embedding vector
            vector = fetch_ollama_vector(clean_chunk)
            if not vector: 
                continue

            new_entries.append({
                "content": clean_chunk,
                "source": os.path.basename(file_path),
                "embedding": vector
            })

        if new_entries:
            self.registry.extend(new_entries)
            self._save_database()

    def query_similarity(self, query_text: str, k: int = 2, lambda_mult: float = 0.5) -> list:
        """Queries the store using dynamic MMR to ensure diverse contextual opinions."""
        query_vec = fetch_ollama_vector(query_text)
        if not query_vec or not self.registry:
            return []
        return compute_top_k(query_vec, self.registry, k, lambda_mult=lambda_mult)
