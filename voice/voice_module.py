import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import threading
import tkinter as tk
from faster_whisper import WhisperModel
import openwakeword
from openwakeword.model import Model as OWWModel
from piper import PiperVoice
from dotenv import load_dotenv
load_dotenv()

from .audio_loop import _unified_audio_loop
from .audio_processors import _process_captured_audio, _process_initial_prompt_logic, _process_confirmation_logic
from .audio_utils import _say_reply, trigger_reread, toggle_listening, _clear_input_box, _reset_to_wake_word

class VoiceSystem:
    def __init__(self, root, input_frame, input_entry, send_command):
        self.root = root
        self.input_frame = input_frame
        self.input_entry = input_entry
        self.send_command = send_command
        
        # Core State System
        self.is_running = True
        self.current_state = "WAKE_WORD"  # States: WAKE_WORD, RECORDING, CONFIRMATION, PLAYING
        self.wake_phrase = "alexa"
        self.pending_text = ""

        base_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(base_dir, os.getenv("voice_file"))
        json_path = os.path.join(base_dir, os.getenv("voice_json"))
        
        self.piper_model = PiperVoice.load(onnx_path, json_path)
        self.oww_model = OWWModel(wakeword_models=[self.wake_phrase], vad_threshold=0.25)
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)
        
        # UI Button Setup
        self.mic_button = tk.Button(
            self.input_frame, text="🎙", bg="#12131b", fg="#a6adc8", bd=0,
            activebackground="#1e1e2e", activeforeground="#b4befe",
            font=("ONE DAY", 12), cursor="hand2", width=3, justify="center" 
        )
        self.mic_button.pack(side="right", padx=(0,5), pady=5)
        self.mic_button.config(command=self.toggle_listening)
        
        # Single background processing thread
        threading.Thread(target=self._unified_audio_loop, daemon=True).start()

    # Bind imported core behaviors cleanly onto the class architecture
    _say_reply = _say_reply
    trigger_reread = trigger_reread
    toggle_listening = toggle_listening
    _clear_input_box = _clear_input_box
    _unified_audio_loop = _unified_audio_loop
    _process_captured_audio = _process_captured_audio
    _process_initial_prompt_logic = _process_initial_prompt_logic
    _process_confirmation_logic = _process_confirmation_logic
    _reset_to_wake_word = _reset_to_wake_word
