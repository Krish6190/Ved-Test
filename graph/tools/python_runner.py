# tools/python_runner.py
import os
import sys
import re
import subprocess
import tkinter as tk
from tkinter import messagebox
from langchain_core.messages import HumanMessage, AIMessage

def execute_safe_python_block(state, config) -> dict:
    """
    Isolates generated Python code blocks, prompts the user for absolute 
    permission via a native popup window, and safely compiles the output.
    """
    # 1. Locate the most recent assistant or user message turn
    last_msg_text = ""
    for msg in reversed(state.messages):
        if hasattr(msg, "content") and msg.content:
            last_msg_text = msg.content
            break
            
    # 2. Extract markdown code snippets securely via regex boundaries
    code_match = re.search(r"```python\s*([\s\S]*?)```", last_msg_text)
    if not code_match:
        code_match = re.search(r"```\s*([\s\S]*?)```", last_msg_text)
        
    if code_match:
        raw_code = code_match.group(1).strip()
    else:
        raw_code = last_msg_text.strip()

    # Safety boundary check: Guard against system logs, placeholders, or tiny snippets
    if not raw_code or raw_code.startswith("[System") or len(raw_code.split()) < 2:
        feedback = HumanMessage(content="SYSTEM NOTICE:\nNo clear executable code block was isolated. Terminal execution aborted.")
        return {"messages": [feedback], "route_intent": "", "mode": state.mode}

    # 3. INTERLOCK PERMISSION POPUP: Render safely on top of all windows
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main blank window frame
        root.attributes("-topmost", True)  # Force popup to the front
        
        user_choice = messagebox.askyesno(
            title="⚠️ Code Execution Approval Requested",
            message=(
                f"Ved is requesting permission to run this generated Python code:\n\n"
                f"----------------------------------------\n"
                f"{raw_code[:600]}\n"
                f"{'... [Truncated for visibility]' if len(raw_code) > 600 else ''}\n"
                f"----------------------------------------\n\n"
                f"Do you authorize running this script on your machine?"
            ),
            parent=root
        )
        root.destroy()
    except Exception:
        user_choice = False  # Secure fallback: deny if UI initialization fails

    if not user_choice:
        feedback = HumanMessage(content="SYSTEM TOOL BLOCK NOTICE:\nThe user explicitly denied execution authorization for this code block. Explain this restriction calmly.")
        return {"messages": [feedback], "route_intent": "", "mode": state.mode}

    # 4. Push live status update text to the UI thread streaming queue
    try:
        token_queue = config["configurable"]["token_queue"]
    except (KeyError, TypeError):
        token_queue = None
    
    if token_queue:
        token_queue.put(f"\n[System Status: Running script execution block...]\n")

    # Establish sandboxed temporary runtime file paths
    temp_script_path = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "ved_runtime_exec.py")
    os.makedirs(os.path.dirname(temp_script_path), exist_ok=True)
    
    terminal_output = ""
    try:
        with open(temp_script_path, "w", encoding="utf-8") as f:
            f.write(raw_code)
            
        # Execute the temporary script using the current python environment
        res = subprocess.run(
            [sys.executable, "-u", temp_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10  # Hardware watchdog limit to prevent lockups
        )
        terminal_output = res.stdout
    except subprocess.TimeoutExpired:
        terminal_output = "Execution Failure Error: Terminal process terminated due to a strict hardware timeout gate (10s limit exceeded)."
    except Exception as e:
        terminal_output = f"Execution Failure Error: {str(e)}"
    finally:
        if os.path.exists(temp_script_path):
            try:
                os.remove(temp_script_path)
            except Exception:
                pass

    if not terminal_output.strip():
        terminal_output = "Process execution completed successfully but returned an empty response layout from standard output."

    if token_queue:
        token_queue.put(f"[System Execution Complete: Received data bytes from subprocess.]\n")

    feedback_message = HumanMessage(
        content=(
            "SYSTEM FILE SYSTEM TOOL CALL NOTICE:\n"
            "The script execution returned the logs below. Analyze these logs and provide a brief summary of the execution results to the user.\n\n"
            f"TERMINAL RAW LOG OUTPUT:\n{terminal_output.strip()}"
        )
    )
    
    return {"messages": [feedback_message], "route_intent": "", "mode": state.mode}
