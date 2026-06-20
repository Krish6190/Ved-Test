import subprocess
from langchain_core.tools import tool

@tool
def run_os_app(app_name: str) -> str:
    """
    CRITICAL: This tool doesnt work rt now
    just return not available
    ONLY run this tool if the user explicitly orders or demands 
    to open, launch, or start a local computer app or game right now.
    """
    try:
        return "tool not available"
        # if "notepad" in app_name.lower():
        #     subprocess.Popen(["notepad.exe"])
        #     return "Successfully launched Notepad."
        # return f"App '{app_name}' is not configured in local paths."
    except Exception as e:
        return f"Failed to open app: {e}"
