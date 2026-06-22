import os
import threading
import time
import tkinter as tk
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import openwakeword
from openwakeword.model import Model as OWWModel
from piper import PiperVoice
from dotenv import load_dotenv
load_dotenv()

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

        print("[Voice Engine] Initializing verbal feedback engine...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(base_dir, os.getenv("voice_file"))
        json_path = os.path.join(base_dir, os.getenv("voice_json"))
        
        print("[Voice Engine] Loading AI model... please wait.")
        self.piper_model = PiperVoice.load(onnx_path, json_path)
        print("[Voice Engine] Piper TTS loaded successfully.")

        print("[Voice Engine] Initializing memory-efficient openWakeWord monitor...")
        self.oww_model = OWWModel(wakeword_models=[self.wake_phrase])

        print("[Voice Engine] Loading AI model... please wait.")
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)
        print("[Voice Engine] Model loaded successfully.")
        
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

    def _say_reply(self, text):
        """Plays speech using Piper, silently forcing master hardware level modifications without overlays."""
        previous_state = self.current_state
        self.current_state = "PLAYING"
        import ctypes
        try:
            Ole32 = ctypes.windll.ole32
            Ole32.CoInitialize(None)
            winmm = ctypes.WinDLL('winmm.dll')
            orig_vol = ctypes.c_uint32()
            winmm.waveOutGetVolume(0, ctypes.byref(orig_vol))
            winmm.waveOutSetVolume(0, 0xFFFFFFFF)
        except Exception:
            orig_vol = None
        if self.piper_model:
            try:
                raw_bytes = b""
                for chunk in self.piper_model.synthesize(text):
                    raw_bytes += chunk.audio_int16_bytes
                audio_data = np.frombuffer(raw_bytes, dtype=np.int16)
                silence_padding = np.zeros(7200, dtype=np.int16)
                silence_trailing = np.zeros(4800, dtype=np.int16)
                full_audio = np.concatenate([silence_padding, audio_data, silence_trailing])
                sd.play(full_audio, 16000)
                sd.wait() 
            except Exception as e:
                print(f"[Audio Error] {e}")
        if orig_vol is not None:
            try:
                winmm.waveOutSetVolume(0, orig_vol)
            except Exception:
                pass
        try:
            Ole32.CoUninitialize()
        except Exception:
            pass
        self.current_state = previous_state

    def trigger_reread(self):
        """Resets text fields and instantly drops back into prompt recording."""
        self.root.after(0, self._clear_input_box)
        self._say_reply("Try again.")
        time.sleep(0.2)
        self.current_state = "RECORDING"

    def toggle_listening(self):
        """Handles manual button clicks cleanly."""
        if self.current_state in ["RECORDING", "CONFIRMATION"]:
            self.mic_button.config(text="🎙️", fg="#a6adc8")
            self.current_state = "WAKE_WORD"
            print("[Voice] Listening stopped manually.")
        else:
            self.mic_button.config(text="🛑", fg="#f38ba8")
            self._say_reply("Yes? What's up?")
            self.current_state = "RECORDING"

    def _clear_input_box(self):
        if hasattr(self.input_entry, "delete"):
            if isinstance(self.input_entry, tk.Text):
                self.input_entry.delete("1.0", tk.END)
            else:
                self.input_entry.delete(0, tk.END)

    def _unified_audio_loop(self):
        """Single microphone loop to avoid hardware streaming clashes."""
        fs = 16000
        chunk_size = 1280  
        
        recorded_chunks = []
        silence_threshold = 150      
        if self.current_state == "CONFIRMATION":
            pause_limit_chunks = 6    # Fast cut-off for quick yes/no answers
        else:
            pause_limit_chunks = 10  # Longer pause allowed for initial prompt to accommodate natural speech patterns    
        silent_chunk_counter = 0
        has_started_talking = False

        print(f"[Wake Engine] Background listening active. Say '{self.wake_phrase}'...")

        with sd.InputStream(samplerate=fs, channels=1, dtype='int16', blocksize=chunk_size) as stream:
            while self.is_running:
                if self.current_state == "PLAYING":
                    if stream.read_available > 0:
                        stream.read(stream.read_available)
                    time.sleep(0.05)
                    continue

                try:
                    audio_chunk, overflowed = stream.read(chunk_size)
                    audio_data = audio_chunk.flatten()
                    volume_score = np.abs(audio_data).mean()
                    
                    # 1. Listening for Wake Word Mode
                    if self.current_state == "WAKE_WORD":
                        prediction = self.oww_model.predict(audio_data)
                        wake_score = prediction[self.wake_phrase]
                        
                        if wake_score > 0.65 and volume_score > 15:
                            print(f"[Wake Engine] Wake word triggered! (Confidence: {wake_score:.2f})")
                            self.root.after(0, lambda: self.mic_button.config(text="🛑", fg="#f38ba8"))
                            beep_sample_rate = 16000
                            beep_duration = 0.1
                            t = np.linspace(0, beep_duration, int(beep_sample_rate * beep_duration), False)
                            beep_signal = np.sin(440 * t * 2 * np.pi) * 3000  # 440Hz Sine Wave signal
                            sd.play(beep_signal.astype(np.int16), beep_sample_rate)
                            sd.wait()

                            if stream.read_available > 0:
                                stream.read(stream.read_available)
                                
                            recorded_chunks = []
                            silent_chunk_counter = 0
                            has_started_talking = False
                            print("[Voice] Actively recording your prompt...")
                            self.current_state = "RECORDING"
                        continue

                    # 2. Recording Mode
                    if self.current_state in ["RECORDING", "CONFIRMATION"]:
                        recorded_chunks.append(audio_chunk)
                        
                        if not has_started_talking:
                            if volume_score > silence_threshold:
                                has_started_talking = True
                        else:
                            if volume_score < silence_threshold:
                                silent_chunk_counter += 1
                            else:
                                silent_chunk_counter = 0 
                            
                            if silent_chunk_counter >= pause_limit_chunks or len(recorded_chunks) >= 150:
                                print("[Voice] Pause detected. Stopping capture early.")
                                # Clone chunks to process safely, then clear buffers instantly
                                current_chunks = list(recorded_chunks)
                                current_talked = has_started_talking
                                
                                # ⚡ Reset all state metrics BEFORE running heavy audio functions
                                recorded_chunks = []
                                silent_chunk_counter = 0
                                has_started_talking = False
                                
                                self._process_captured_audio(current_chunks, current_talked, fs, stream)

                except Exception as e:
                    time.sleep(0.05)

    def _process_captured_audio(self, chunks, talked, fs, stream):
        self.root.after(0, lambda: self.mic_button.config(text="🎙️", fg="#a6adc8"))
        if not chunks or not talked:
            if self.current_state == "CONFIRMATION":
                print("[Voice] Silence detected. Assuming default confirmation ('Yes, keep text').")
                self._process_confirmation_logic("yes", stream)
            else:
                self.current_state = "WAKE_WORD"
                # FIX: Clear mic data so returning to wake word doesn't self-trigger
                if stream.read_available > 0:
                    stream.read(stream.read_available)
                print(f"[Wake Engine] Background listening active. Say '{self.wake_phrase}'...")
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
        
        print(f"[System] Auditing command text. Repeating back: '{text}'")
        self._say_reply(f"You said: {text}. Execute?")
        self.current_state = "CONFIRMATION"

    def _process_confirmation_logic(self, text, stream):
        # Normalize the incoming words
        words = text.lower().replace(',', '').replace('.', '').split() if text else []
        
        confirm_keywords = ["yes", "send", "submit", "execute", "run", "go", "do it", "enter", "correct"]
        cancel_keywords = ["no", "wrong", "stop", "cancel", "don't"]
        
        has_confirm = any(word in words for word in confirm_keywords)
        has_cancel = any(word in words for word in cancel_keywords)

        if has_confirm:
            print("[Voice] Command verified. Executing send_command.")
            self.root.after(0, lambda: self.send_command(None))
            self._reset_to_wake_word(stream)
            
        elif has_cancel:
            print("[Voice] Execution canceled by user request. Returning to sleep mode.")
            self.root.after(0, self._clear_input_box)
            self._say_reply("Canceled.")
            self._reset_to_wake_word(stream)
            
        else:
            # ⏱️ 4-Second Silence Gate: If no clear yes/no keyword is heard, say "No response" and exit
            print("[Voice] Confirmation clear choice not heard or timed out. Dropping locks.")
            self._say_reply("No response.")
            self._reset_to_wake_word(stream)

    def _reset_to_wake_word(self, stream):
        """Helper to cleanly clear caches and reset state back to listening."""
        time.sleep(0.2)
        if stream.read_available > 0:
            stream.read(stream.read_available)
        if hasattr(self.oww_model, "reset"):
            self.oww_model.reset()
        self.current_state = "WAKE_WORD"
        print(f"[Wake Engine] Background listening active. Say '{self.wake_phrase}'...")
