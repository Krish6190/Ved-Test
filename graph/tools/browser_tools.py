import webbrowser
from langchain_core.tools import tool

@tool
def open_browser_url(url: str) -> str:
    """
    currently not functional just return not available
    CRITICAL: ONLY run this tool if the user explicitly orders or demands 
    to open, browse, visit, or launch a live website link or URL right now.
    """
    try:
        return "tool not available"
        # webbrowser.open(url)
        # return f"Successfully opened web browser to {url}."
    except Exception as e:
        return f"Failed to open browser: {e}"
