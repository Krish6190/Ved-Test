from pathlib import Path

class DiskMemoryManager:
    """Handles loading, saving, and capping long-term memories on the hard drive."""
    
    def __init__(self, project_root: Path):
        self.file_path = project_root / "pinned_memories.txt"
        self.saved_memories = self._load_from_disk()

    def _load_from_disk(self) -> list:
        """Reads your saved facts from the text file on startup."""
        if not self.file_path.exists():
            return []
        try:
            lines = self.file_path.read_text(encoding="utf-8").splitlines()
            return [line.strip() for line in lines if line.strip()]
        except Exception:
            return []

    def save_to_disk(self) -> bool:
        """Physically locks your current memories onto your hard drive."""
        try:
            text_content = "\n".join(self.saved_memories)
            self.file_path.write_text(text_content, encoding="utf-8")
            return True
        except Exception as e:
            print(f"Failed to write memories to hard drive: {e}")
            return False

    def add_pin(self, pinned_block: str) -> str:
        """Appends a new pinned block if under the 8-slot limit, then updates disk."""
        if len(self.saved_memories) >= 8:
            return (
                "[WARNING]: Your long-term memory is full (8/8 slots occupied)!\n"
                "Please use the command '/unpin_all' to empty your saves before locking new ones."
            )
        
        self.saved_memories.append(pinned_block)
        self.save_to_disk()
        return "Pinned previous message exchange successfully into local disk memory."

    def clear_all(self) -> str:
        """Wipes out all stored memories from RAM and your hard drive."""
        self.saved_memories = []
        self.save_to_disk()
        return "[Success] All long-term saved memories have been wiped from your hard drive."

    def get_formatted_list(self) -> str:
        """Returns a numbered string list of all your pinned memories."""
        if not self.saved_memories:
            return "You have no messages locked in long-term memory."
        formatted = "\n".join([f"{i+1}. {m}" for i, m in enumerate(self.saved_memories)])
        return f"Current Long-Term Disk Storage ({len(self.saved_memories)}/8):\n{formatted}"

    def unpin_by_index(self, index_str: str) -> str:
        """Deletes a single specific memory slot using its list number."""
        try:
            # Convert user text input into a standard Python index number (1-based to 0-based)
            idx = int(index_str.strip()) - 1
            
            if idx < 0 or idx >= len(self.saved_memories):
                return f"[Error] Invalid slot number. Choose between 1 and {len(self.saved_memories)}."
            
            removed_item = self.saved_memories.pop(idx)
            self.save_to_disk() # Stamp the deletion onto your hard drive
            return f"[Success] Removed from long-term memory: '{removed_item}'"
            
        except ValueError:
            return "[Error] Please provide a valid number. Usage: /unpin 2"