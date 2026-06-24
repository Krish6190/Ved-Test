import tkinter as tk
import numpy as np

def _process_captured_audio(self, chunks, talked, fs, stream):
    self.root.after(0, lambda: self.mic_button.config(text="🎙", fg="#a6adc8"))
    if not chunks or not talked:
        if self.current_state == "CONFIRMATION":
            self._process_confirmation_logic("timeout_keep_text", stream)
        else:
            self.current_state = "WAKE_WORD"
            if stream.read_available > 0:
                stream.read(stream.read_available)
        return
    try:
        audio_data = np.concatenate(chunks, axis=0).flatten().astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(
            audio_data, 
            beam_size=3, 
            language="en", 
            vad_filter=True, 
            vad_parameters=dict(min_speech_duration_ms=50),
        )
        text = "".join([segment.text for segment in segments]).strip()
        print(f"[Voice] Recognized text: {text}")
        if self.current_state == "CONFIRMATION":
            self._process_confirmation_logic(text, stream)
        else:
            self._process_initial_prompt_logic(text)
    except Exception as e:
        print(f"[Voice Error] {e}")
        self.current_state = "WAKE_WORD"
        if stream.read_available > 0:
            stream.read(stream.read_available)

def _process_initial_prompt_logic(self, text):
    if not text:
        self.current_state = "WAKE_WORD"
        return

    def update_ui():
        self._clear_input_box()
        if isinstance(self.input_entry, tk.Text):
            self.input_entry.insert("1.0", text)
        else:
            self.input_entry.insert(0, text)
    self.root.after(0, update_ui)
    self.pending_text = text
    
    self._say_reply(f"You said: {text}. Should I send this?")
    self.current_state = "CONFIRMATION"

def _process_confirmation_logic(self, text, stream):
    words = text.lower().replace(',', '').replace('.', '').split() if text else []
    
    confirm_keywords = ["yes", "send", "submit", "execute", "run", "go", "do it", "enter", "correct"]
    cancel_keywords = ["no", "wrong", "stop", "cancel", "don't"]
    repeat_keywords = ["repeat", "again", "say again"]
    
    has_confirm = any(word in words for word in confirm_keywords)
    has_cancel = any(word in words for word in cancel_keywords)
    has_repeat = any(word in words for word in repeat_keywords)

    if text == "timeout_keep_text":
        print("[Voice] Confirmation timed out. Keeping text on input bar without sending.")
        self._say_reply("Timed out. Keeping text.")
        self._reset_to_wake_word(stream)
    elif has_confirm:
        print("[Voice] Confirmation positive match heard. Executing pending graph turn...")
        self._say_reply("Sending.")
        self.root.after(0, lambda: self.send_command(None))
        self._reset_to_wake_word(stream)
    elif has_cancel:
        print("[Voice] Execution canceled by user request. Returning to sleep mode.")
        self.root.after(0, self._clear_input_box)
        self._say_reply("Canceled.")
        self._reset_to_wake_word(stream)
    elif has_repeat:
        print("[Voice] Repeat requested. Re-auditing text prompt...")
        self.root.after(0, lambda: self._say_reply(f"You said: {self.pending_text}. Execute?"))
        if stream.read_available > 0:
            stream.read(stream.read_available)
        self.current_state = "RECORDING"
    else:
        print("[Voice] Confirmation clear choice not heard or timed out. Dropping locks.")
        self._say_reply("No response.")
        self._reset_to_wake_word(stream)
