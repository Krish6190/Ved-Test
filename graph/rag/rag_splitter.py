from langchain_text_splitters import RecursiveCharacterTextSplitter

class RagTextSplitter:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_raw_text(self, text: str) -> list:
        """Processes an input string into smaller, overlapping character arrays."""
        if not text or not text.strip():
            return []
        return self.splitter.split_text(text)
