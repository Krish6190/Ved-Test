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
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")
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
        """Plays speech safely using Piper, adding audio-awaken padding without breaking arguments."""
        previous_state = self.current_state
        self.current_state = "PLAYING"
        
        if self.piper_model:
            try:
                raw_bytes = b""
                # FIX: Removed the broken argument completely so it never crashes
                for chunk in self.piper_model.synthesize(text):
                    raw_bytes += chunk.audio_int16_bytes
                
                audio_data = np.frombuffer(raw_bytes, dtype=np.int16)
                
                # Kept your silent padding to wake up the Windows audio driver
                silence_padding = np.zeros(7200, dtype=np.int16)
                silence_trailing = np.zeros(4800, dtype=np.int16)
                full_audio = np.concatenate([silence_padding, audio_data, silence_trailing])
                
                sd.play(full_audio, 16000)
                sd.wait() 
            except Exception as e:
                print(f"[Audio Error] {e}")
        
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
                            self.mic_button.config(text="🛑", fg="#f38ba8")
                            
                            self._say_reply("Yes? What's up?")
                            
                            # Clean stream cache before recording your prompt
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
                                self._process_captured_audio(recorded_chunks, has_started_talking, fs, stream)
                                
                                recorded_chunks = []
                                silent_chunk_counter = 0
                                has_started_talking = False

                except Exception as e:
                    time.sleep(0.05)

    def _process_captured_audio(self, chunks, talked, fs, stream):
        filename = "temp_voice.wav"
        self.mic_button.config(text="🎙️", fg="#a6adc8")
        
        # DEFAULT TO YES ON SILENCE OPTIMIZATION
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
            audio_data = np.concatenate(chunks, axis=0)
            sf.write(filename, audio_data, fs)
            segments, info = self.model.transcribe(filename, beam_size=3, language="en", vad_filter=True, vad_parameters=dict(min_speech_duration_ms=50))
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
        finally:
            if os.path.exists(filename):
                try: os.remove(filename)
                except: pass

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
        # Your custom question string updates
        self._say_reply("That's all?... Yes or No?")
        print("[System] Waiting for confirmation. Listening loop starting shortly...")
        self.current_state = "CONFIRMATION"

    def _process_confirmation_logic(self, text, stream):
        if not text:
            words = ["yes"]
        else:
            words = text.lower().replace(',', '').replace('.', '').split()
        if not words:
            words = ["yes"]
        if "yes" in words:
            print("[Voice] Confirmation accepted.")
            send_sounds = ["send", "sen", "sand", "sent", "saying", "sound", "enter", "go", "input", "submit"]
            has_send_keyword = any(sound in words for sound in send_sounds)
            if has_send_keyword and self.pending_text:
                print("[Voice] Send keyword detected. Executing send_command.")
                self.root.after(0, lambda: self.send_command(None))
            else:
                print("[Voice] Just confirmation detected. Text remains in entry box.")
            time.sleep(0.2)
            if stream.read_available > 0:
                stream.read(stream.read_available)
            if hasattr(self.oww_model, "reset"):
                self.oww_model.reset()
            self.current_state = "WAKE_WORD"
            print(f"[Wake Engine] Background listening active. Say '{self.wake_phrase}'...")
        elif "no" in words or "wrong" in words:
            print("[Voice] Correction requested. Restarting record chain.")
            self.root.after(0, self.trigger_reread)
        else:
            print("[Voice] Unrecognized choice. Demanding explicit response...")
            self._say_reply("That's all?... Yes or No?")
            self.current_state = "CONFIRMATION"