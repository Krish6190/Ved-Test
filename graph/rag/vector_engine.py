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
        self.db_path = os.getenv("DB_PATH", "data/vectordb/index.bin")
        self.file_parser = RagDocumentParser()
        self.embeddings_engine = self._new_embeddings_engine()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.registry = []
        self.vectors_matrix = None
        self._load_database()

    @staticmethod
    def _new_embeddings_engine() -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model="nomic-embed-text:latest",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            keep_alive=600,
        )

    def reset_embeddings_engine(self) -> None:
        """Create a fresh embeddings engine (e.g., after hibernate).

        OllamaEmbeddings is stateless across calls, but creating a fresh
        instance after a flush ensures any cached state is dropped and the
        next embed call starts a clean request to Ollama.
        """
        self.embeddings_engine = self._new_embeddings_engine()

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
                    # Backward compat: ensure every loaded entry has the expected keys.
                    for record in self.registry:
                        if isinstance(record, dict):
                            if "scope" not in record:
                                record["scope"] = GLOBAL_SCOPE
                            # Older records (pre-chunker-aware ingest) won't have these.
                            record.setdefault("chunker", "text")
                            record.setdefault("layer", "body")
                            record.setdefault("name", "text")
                            record.setdefault("lineno", 0)
                            record.setdefault("end_lineno", 0)
            except Exception as e:
                print(f"[RAG Engine] Binary cache reading warning: {e}")

    def _save_database(self):
        try:
            with open(self.db_path, "wb") as f:
                # Save both structures to maintain matrix persistence
                pickle.dump({"registry": self.registry, "matrix": self.vectors_matrix}, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[RAG Engine] Disk write warning: {e}")

    def ingest_local_file(self, file_path: str, scope: str = GLOBAL_SCOPE, chunker: str = "text", source: str | None = None) -> bool:
        """Read a local file and ingest its chunks into the vector DB.

        Args:
            file_path: Path to the source file on disk.
            scope: Registry scope tag (typically the thread id or GLOBAL_SCOPE).
            chunker: "text" uses RagDocumentParser.process_file_to_chunks (the
                original behavior). "ast" uses graph.rag.code_chunker.chunk_file
                for AST-aware 2-layer chunking (signatures + bodies).
            source: Registry `source` label stored on every chunk. If None, we
                fall back to os.path.basename(file_path). Callers should pass an
                explicit basename so delete_by_source can evict by source later.

        Returns:
            True if at least one new chunk was successfully committed to the
            vector DB, False otherwise (file unreadable, chunker failed with
            no fallback available, embedding failed, or the file was already
            fully indexed).

        Robustness: when the requested chunker fails (e.g. AST chunker hits
        a syntax error on a non-Python file or a syntax-invalid snippet),
        we transparently fall back to the text chunker so the file still
        gets indexed. Previously the AST path returned silently with
        zero chunks committed, and `index_workspace` marked the file as
        indexed anyway -- producing an empty RAG result for that file.
        """
        records: list = []
        ast_failed = False

        if chunker == "ast":
            try:
                import importlib.util
                import pathlib
                module_path = pathlib.Path(__file__).resolve().parent / "code_chunker.py"
                spec = importlib.util.spec_from_file_location("ved_ast_chunker", module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load chunker spec from {module_path}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                chunk_file = getattr(module, "chunk_file")
                raw_chunks = chunk_file(file_path)
            except Exception as e:
                print(f"[RAG Engine] AST chunking failed for {file_path}: {e}; falling back to text chunker")
                ast_failed = True
                raw_chunks = None
            if raw_chunks is not None:
                for c in raw_chunks:
                    content = (c.get("content") or "").strip()
                    if not content:
                        continue
                    records.append({
                        "content": content,
                        "source": source if source is not None else os.path.basename(file_path),
                        "scope": scope,
                        "chunker": "ast",
                        "layer": c.get("layer", "body"),
                        "name": c.get("name", ""),
                        "lineno": int(c.get("lineno", 0) or 0),
                        "end_lineno": int(c.get("end_lineno", 0) or 0),
                    })

        # Fall back to text chunker when AST failed or produced zero
        # records. We also fall back if AST succeeded but yielded nothing
        # (e.g. an empty source file with only docstrings/comments that
        # the AST layer dropped).
        if (chunker == "ast" and (ast_failed or not records)) or chunker == "text":
            try:
                chunks = self.file_parser.process_file_to_chunks(file_path)
            except Exception as e:
                print(f"[RAG Engine] Text chunking failed for {file_path}: {e}")
                chunks = []
            for c in chunks:
                content = c.strip()
                if not content:
                    continue
                records.append({
                    "content": content,
                    "source": source if source is not None else os.path.basename(file_path),
                    "scope": scope,
                    "chunker": "text",
                    "layer": "body",
                    "name": "text",
                    "lineno": 0,
                    "end_lineno": 0,
                })

        if not records:
            print(f"[RAG Engine] No chunks produced for {file_path}; skipping commit")
            return False
        existing_contents = {record["content"] for record in self.registry}
        records_to_embed = [r for r in records if r["content"] not in existing_contents]
        if not records_to_embed:
            return False
        try:
            vectors = self.embeddings_engine.embed_documents([r["content"] for r in records_to_embed])
        except Exception as e:
            print(f"[RAG Engine] Batch vector processing exception for {file_path}: {e}")
            return False
        new_entries = []
        new_vectors = []
        for rec, vec in zip(records_to_embed, vectors):
            if vec:
                new_entries.append(rec)
                new_vectors.append(vec)
        if not new_entries:
            print(f"[RAG Engine] Embedder returned no usable vectors for {file_path}")
            return False
        new_v_arr = np.array(new_vectors, dtype=np.float32)
        self.vectors_matrix = np.vstack([self.vectors_matrix, new_v_arr]) if self.vectors_matrix is not None else new_v_arr
        self.registry.extend(new_entries)
        try:
            self._save_database()
        except Exception as e:
            print(f"[RAG Engine] Disk write failed for {file_path}: {e}")
            return False
        return True

    def ingest_text(self, text: str, scope: str = GLOBAL_SCOPE, source: str = "raw_text", chunker: str = "text"):
        """Embed raw text into the vector DB without going through the disk path.

        Used by ThreadFileStore.add_text and the GUI paste path. The text is
        split via the RagDocumentParser's text splitter. Each resulting chunk
        is stored as a registry record with `chunker` (default "text") and
        `layer="body"` so retrieval treats them like other text chunks.

        `source` is required (no basename fallback because there is no file).
        """
        if not text or not text.strip():
            return
        chunks = self.file_parser.text_splitter.split_raw_text(text)
        records = []
        for c in chunks:
            content = c.strip()
            if not content:
                continue
            records.append({
                "content": content,
                "source": source,
                "scope": scope,
                "chunker": chunker,
                "layer": "body",
                "name": "text",
                "lineno": 0,
                "end_lineno": 0,
            })
        if not records:
            return
        existing_contents = {record["content"] for record in self.registry}
        records_to_embed = [r for r in records if r["content"] not in existing_contents]
        if not records_to_embed:
            return
        try:
            vectors = self.embeddings_engine.embed_documents([r["content"] for r in records_to_embed])
            new_entries = []
            new_vectors = []
            for rec, vec in zip(records_to_embed, vectors):
                if vec:
                    new_entries.append(rec)
                    new_vectors.append(vec)
            if new_entries:
                new_v_arr = np.array(new_vectors, dtype=np.float32)
                self.vectors_matrix = np.vstack([self.vectors_matrix, new_v_arr]) if self.vectors_matrix is not None else new_v_arr
                self.registry.extend(new_entries)
                self._save_database()
        except Exception as e:
            print(f"[RAG Engine] Text ingest exception: {e}")

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
