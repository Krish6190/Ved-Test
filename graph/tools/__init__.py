from langchain_core.tools import tool
from .os_tools import run_os_app
from .browser_tools import open_browser_url
from .system_explorer_tools import scan_local_directory
@tool
def list_tools() -> str:
    """
    Use this tool ONLY if the user explicitly orders, demands, or commands 
    you to list, show, print, or name your available tools, skills, or features.
    
    CRITICAL PROHIBITION: DO NOT call this tool if the user is just saying hello, 
    greeting you, checking if you are there, or starting a normal conversation. 
    Greetings like 'hlo', 'hello', or 'hi' must be handled with pure text chat, 
    NOT this tool.
    """
    return (
        "Here are my active system tools:\n"
        "- run_os_app: Opens local desktop apps or games.\n"
        "- open_browser_url: Opens websites and web links.\n"
        "- scan_local_directory: Reads your computer folders to see what games or projects are available.\n"
        "- list_tools: Shows this list of features."
    )
# The master list passed to Llama and LangGraph
all_ved_tools = [run_os_app, open_browser_url, scan_local_directory, list_tools]
