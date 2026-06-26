import os
import pickle
import numpy as np
from langchain_ollama import OllamaEmbeddings
from .rag_parser import RagDocumentParser
from .rag_maths import compute_top_k

# Sentinel scope for chunks that are not tied to any thread.
GLOBAL_SCOPE = "__GLOBAL__"


class LocalVectorDB:
    def __init__(self):
        self.db_path = os.getenv("DB_PATH", r"E:\VectorDB\index.bin")
        self.file_parser = RagDocumentParser()
        self.embeddings_engine = OllamaEmbeddings(
            model="nomic-embed-text:latest",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            keep_alive=600
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.registry = []
        self.vectors_matrix = None
        self._load_database()

    def _load_database(self):
        if os.path.exists(self.db_path) and os.path.getsize(self.db_path) > 0:
            try:
                with open(self.db_path, "rb") as f:
                    data = pickle.load(f)
                    # Support legacy structures safely
                    if isinstance(data, dict) and "registry" in data:
                        self.registry = data["registry"]
                        self.vectors_matrix = data["matrix"]
                    else:
                        self.registry = data
                        if data:
                            valid_vectors = [r["embedding"] for r in data if "embedding" in r]
                            if valid_vectors:
                                self.vectors_matrix = np.array(valid_vectors, dtype=np.float32)
                    # Backward compat: ensure every loaded entry has a scope.
                    for record in self.registry:
                        if isinstance(record, dict) and "scope" not in record:
                            record["scope"] = GLOBAL_SCOPE
            except Exception as e:
                print(f"[RAG Engine] Binary cache reading warning: {e}")

    def _save_database(self):
        try:
            with open(self.db_path, "wb") as f:
                # Save both structures to maintain matrix persistence
                pickle.dump({"registry": self.registry, "matrix": self.vectors_matrix}, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[RAG Engine] Disk write warning: {e}")

    def ingest_local_file(self, file_path: str, scope: str = GLOBAL_SCOPE):
        chunks = self.file_parser.process_file_to_chunks(file_path)
        if not chunks: return
        existing_contents = {record["content"] for record in self.registry}
        chunks_to_embed = [c.strip() for c in chunks if c.strip() and c.strip() not in existing_contents]
        if not chunks_to_embed: return
        try:
            vectors = self.embeddings_engine.embed_documents(chunks_to_embed)
            new_entries = []
            new_vectors = []
            for text, vec in zip(chunks_to_embed, vectors):
                if vec:
                    new_entries.append({
                        "content": text,
                        "source": os.path.basename(file_path),
                        "scope": scope,
                    })
                    new_vectors.append(vec)
            if new_entries:
                new_v_arr = np.array(new_vectors, dtype=np.float32)
                self.vectors_matrix = np.vstack([self.vectors_matrix, new_v_arr]) if self.vectors_matrix is not None else new_v_arr
                self.registry.extend(new_entries)
                self._save_database()
        except Exception as e:
            print(f"[RAG Engine] Batch vector processing exception: {e}")

    def query_similarity(self, query_text: str, k: int = 2, lambda_mult: float = 0.5, scope: str | None = None) -> list:
        if not self.registry or self.vectors_matrix is None: return []
        # Optional scope filtering: restrict the registry and matrix to matching rows
        # before delegating to the SIMD top-k math.
        if scope is None:
            registry = self.registry
            matrix = self.vectors_matrix
        else:
            keep_idx = [i for i, r in enumerate(self.registry) if r.get("scope", GLOBAL_SCOPE) == scope]
            if not keep_idx:
                return []
            registry = [self.registry[i] for i in keep_idx]
            matrix = self.vectors_matrix[keep_idx]
        try:
            query_vec = self.embeddings_engine.embed_query(query_text)
            return compute_top_k(query_vec, registry, matrix, k, lambda_mult=lambda_mult)
        except Exception: return []

    def delete_by_source(self, scope: str, source: str) -> int:
        """Remove every chunk whose scope AND source match.

        Used by the per-thread quota enforcer to drop the oldest upload(s) when
        the thread's chunk total exceeds its quota. Returns the number of
        chunks removed. Rebuilds vectors_matrix in place; persists immediately.
        """
        if not self.registry:
            return 0
        keep_idx = [
            i for i, r in enumerate(self.registry)
            if not (r.get("scope", GLOBAL_SCOPE) == scope and r.get("source") == source)
        ]
        deleted = len(self.registry) - len(keep_idx)
        if deleted == 0:
            return 0
        self.registry = [self.registry[i] for i in keep_idx]
        if keep_idx and self.vectors_matrix is not None and len(self.vectors_matrix) >= len(keep_idx):
            self.vectors_matrix = self.vectors_matrix[keep_idx]
        else:
            self.vectors_matrix = None
        self._save_database()
        return deleted
