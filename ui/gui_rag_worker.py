import threading
from tkinter import filedialog
from langchain_core.messages import HumanMessage
from .gui_render_engine import VedGuiRenderEngine

class VedRagWorker(VedGuiRenderEngine):
    def __init__(self, root):
        super().__init__(root)
        self.chatbot = None  # Instantiated and wired up securely by ui/gui.py

    def _trigger_file_attachment(self):
        """Launches a native file dialog limited strictly to code and document extensions."""
        supported_extensions = [("Parsable Assets", "*.txt;*.py;*.md;*.json;*.js;*.cpp;*.h")]
        chosen_path = filedialog.askopenfilename(
            parent=self.root, title="Select Local Asset for RAG Ingestion", filetypes=supported_extensions
        )
        if chosen_path:
            self._process_rag_ingest_pipeline(chosen_path, is_raw_file=True)

    def _process_rag_ingest_pipeline(self, data_payload: str, is_raw_file: bool = True):
        """Spins up an isolated non-blocking worker thread to parse files or large paste blocks."""
        self._append_text("[System: Parsing incoming data layout to database index...]\n", "#89b4fa")
        
        def async_ingest_worker():
            from graph.rag import rag_db
            try:
                old_count = len(rag_db.registry)
                if is_raw_file:
                    rag_db.ingest_local_file(data_payload)
                else:
                    from graph.rag.rag_network import fetch_ollama_vector
                    chunks = rag_db.file_parser.text_splitter.split_raw_text(data_payload)
                    new_nodes = 0
                    existing_entries = {record["content"] for record in rag_db.registry}
                    
                    for chunk in chunks:
                        clean = chunk.strip()
                        if not clean or clean in existing_entries: continue
                        vec = fetch_ollama_vector(clean)
                        if vec:
                            rag_db.registry.append({"content": clean, "source": "DirectUIClipboardPaste", "embedding": vec})
                            new_nodes += 1
                    if new_nodes > 0: rag_db._save_database()
                        
                new_count = len(rag_db.registry)
                self._append_text(f"[System Index Sync Complete: Added {new_count - old_count} unique offline nodes.]\n", "#a6e3a1")
            except Exception as e:
                self._append_text(f"[System Ingest Failure]: {e}\n", "#f38ba8")
                
        threading.Thread(target=async_ingest_worker, daemon=True).start()

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
