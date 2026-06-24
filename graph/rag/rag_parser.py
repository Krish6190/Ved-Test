import os
from .rag_splitter import RagTextSplitter

class RagDocumentParser:
    def __init__(self):
        # Connect our raw string splitter pipeline directly
        self.text_splitter = RagTextSplitter()

    def process_file_to_chunks(self, file_path: str) -> list:
        """Reads a target hard file and converts it into a structural token array."""
        if not os.path.exists(file_path):
            print(f"[RAG Error] Parser target path missing: {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        except Exception as e:
            print(f"[RAG Error] File data stream read failure: {e}")
            return []

        # Hand off clean string data straight to our splitter layout layer
        return self.text_splitter.split_raw_text(raw_text)
