import os
from pathlib import Path
from langchain_core.tools import tool

@tool
def scan_local_directory(category: str) -> str:
    """
    Use this tool when the user asks what games, apps, or projects are 
    installed or available on their computer. 
    
    Arguments:
        category: Must be either 'games' or 'projects' based on what the user wants to see.
    """
    # Define the exact paths on your local machine
    # Change these paths to point to your real folders!
    PATHS = {
        "games": Path(r"C:\Program Files"), 
        "projects": Path(r"C:\Users\YourName\Documents\Projects")
    }
    
    target_path = PATHS.get(category.lower())
    if not target_path or not target_path.exists():
        return f"Could not find the local path configured for {category}."
        
    try:
        # Scan the directory and filter for executable files (.exe) or main folders
        items = os.listdir(target_path)
        
        # Keep only the first 15 items so the list doesn't overload Llama's memory
        visible_items = items[:15]
        
        if not visible_items:
            return f"The {category} folder is currently empty."
            
        formatted_list = "\n".join([f"- {item}" for item in visible_items])
        return f"Here are the available items found in your {category} directory:\n{formatted_list}"
        
    except Exception as e:
        return f"Failed to read the local directory: {e}"
