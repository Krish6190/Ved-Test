import os
import pickle
import numpy as np
from langchain_ollama import OllamaEmbeddings
from .rag_parser import RagDocumentParser
from .rag_maths import compute_top_k

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
            except Exception as e:
                print(f"[RAG Engine] Binary cache reading warning: {e}")

    def _save_database(self):
        try:
            with open(self.db_path, "wb") as f:
                # Save both structures to maintain matrix persistence
                pickle.dump({"registry": self.registry, "matrix": self.vectors_matrix}, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[RAG Engine] Disk write warning: {e}")

    def ingest_local_file(self, file_path: str):
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
                    new_entries.append({"content": text, "source": os.path.basename(file_path)})
                    new_vectors.append(vec)
            if new_entries:
                new_v_arr = np.array(new_vectors, dtype=np.float32)
                self.vectors_matrix = np.vstack([self.vectors_matrix, new_v_arr]) if self.vectors_matrix is not None else new_v_arr
                self.registry.extend(new_entries)
                self._save_database()
        except Exception as e:
            print(f"[RAG Engine] Batch vector processing exception: {e}")

    def query_similarity(self, query_text: str, k: int = 2, lambda_mult: float = 0.5) -> list:
        if not self.registry or self.vectors_matrix is None: return []
        try:
            query_vec = self.embeddings_engine.embed_query(query_text)
            return compute_top_k(query_vec, self.registry, self.vectors_matrix, k, lambda_mult=lambda_mult)
        except Exception: return []
