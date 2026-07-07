import os
import threading
from tkinter import filedialog
from langchain_core.messages import HumanMessage
from .gui_render_engine import VedGuiRenderEngine

class VedRagWorker(VedGuiRenderEngine):
    def __init__(self, root):
        super().__init__(root)
        self.chatbot = None  # Instantiated and wired up securely by ui/gui.py

    def _trigger_file_attachment(self):
        """Open file dialog (multi-select) and STAGE chosen files as chips above the input.
        Files are sent + ingested when the user presses Enter with the prompt (or alone)."""
        supported_extensions = [(
            "Parsable Assets",
            "*.txt *.md *.pdf *.docx *.doc "
            "*.py *.js *.jsx *.ts *.tsx "
            "*.java *.go *.rs *.rb *.php *.cs "
            "*.cpp *.c *.h *.hpp *.swift *.kt *.scala "
            "*.sh *.bash *.zsh *.ps1 "
            "*.html *.css *.scss *.xml *.svg "
            "*.json *.yaml *.yml *.toml *.csv *.sql "
            "*.log *.zip"
        )]
        chosen_paths = filedialog.askopenfilenames(
            parent=self.root, title="Attach Files to Active Thread",
            filetypes=supported_extensions,
        )
        if not chosen_paths:
            return
        for path in chosen_paths:
            if path not in self.pending_attachments:
                self.pending_attachments.append(path)
        self._render_attachment_chips()

    def _ingest_payload(self, data_payload: str, is_raw_file: bool = True):
        """Synchronously ingest a file path or pasted text into the RAG index.
        Returns (added_count, error_message_or_None). Logs to terminal — not the GUI."""
        from graph.rag import rag_db
        from graph.rag.mixer import GLOBAL_SCOPE

        thread_id = None
        if self.chatbot and getattr(self.chatbot, "_active_thread_id", None):
            thread_id = self.chatbot._active_thread_id

        try:
            if is_raw_file:
                if thread_id and self.chatbot and getattr(self.chatbot, "_thread_files", None):
                    # Per-thread quota tracking + LRU eviction.
                    filename = os.path.basename(data_payload)
                    chunker = self.chatbot._rag_chunker() if self.chatbot else "text"
                    entry = self.chatbot._thread_files.add(thread_id, data_payload, filename=filename, chunker=chunker)
                    evicted = entry.pop("evicted", [])
                    print(f"[Ingest] {entry['filename']}: +{entry['chunk_count']} chunks (thread {thread_id[:8]})", flush=True)
                    if evicted:
                        print(f"[Quota] Evicted {len(evicted)} oldest upload(s) to make room: {evicted}", flush=True)
                    return entry["chunk_count"], None
                # No active thread — fall back to direct RAG ingest with global scope.
                old_count = len(rag_db.registry)
                rag_db.ingest_local_file(data_payload, scope=GLOBAL_SCOPE, chunker="text", source=os.path.basename(data_payload))
                added = len(rag_db.registry) - old_count
                print(f"[Ingest] {os.path.basename(data_payload)}: +{added} chunks (no thread, scope=global)", flush=True)
                return added, None

            # Pasted-text path: legacy, no quota tracking.
            from graph.rag.rag_network import fetch_ollama_vector
            chunks = rag_db.file_parser.text_splitter.split_raw_text(data_payload)
            new_nodes = 0
            existing_entries = {record["content"] for record in rag_db.registry}
            scope = thread_id or GLOBAL_SCOPE
            for chunk in chunks:
                clean = chunk.strip()
                if not clean or clean in existing_entries: continue
                vec = fetch_ollama_vector(clean)
                if vec:
                    rag_db.registry.append({
                        "content": clean,
                        "source": "DirectUIClipboardPaste",
                        "scope": scope,
                        "embedding": vec,
                    })
                    new_nodes += 1
            if new_nodes > 0:
                rag_db._save_database()
            print(f"[Ingest] {len(data_payload)} chars pasted: +{new_nodes} chunks (scope={scope})", flush=True)
            return new_nodes, None
        except Exception as e:
            print(f"[Ingest Failure]: {e}", flush=True)
            return 0, str(e)

    def _process_rag_ingest_pipeline(self, data_payload: str, is_raw_file: bool = True):
        """Backwards-compatible async wrapper for the legacy filepath / paste flows.
        Runs _ingest_payload in a background thread and logs to terminal."""
        def worker():
            self._ingest_payload(data_payload, is_raw_file)
        threading.Thread(target=worker, daemon=True).start()

    def _extract_real_human_prompt(self, heavy_payload: str) -> str:
        """Uses the pre-warmed ChatOllama instance running on active hardware to bypass disk reload latency."""
        try:
            if not self.chatbot or self.chatbot._hibernating:
                return "Analyze and summarize this reference material."
                
            active_llm = self.chatbot._get_llm()
            if not active_llm:
                return "Analyze and summarize this reference material."

            extraction_prompt = (
                "You are a structural parser. The user has pasted a massive block of text containing data "
                "along with their actual command or question. Analyze the text below and extract ONLY the "
                "specific instruction, question, or command the human wants executed. Do not return any "
                "of the raw data, code blocks, or document text. If no explicit question is found, return "
                "'Analyze and summarize this reference material.'\n\n"
                f"RAW PASTED PAYLOAD:\n{heavy_payload}"
            )
            
            # Direct invocation bypasses response loop streams and context memory structures
            res = active_llm.invoke([HumanMessage(content=extraction_prompt)])
            extracted_question = res.content.strip()
            return extracted_question if extracted_question else "Process reference material."
        except Exception:
            return "Analyze and summarize this provided context chunk payload."
