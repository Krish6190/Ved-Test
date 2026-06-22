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

class VoiceSystem:
    def __init__(self, root, input_frame, input_entry, send_command):
        self.root = root
        self.input_frame = input_frame
        self.input_entry = input_entry
        self.send_command = send_command
        self.is_listening = False
        self.is_wake_word_active = True 
        self.wake_phrase = "alexa"
        self.awaiting_confirmation = False
        self.pending_text = ""

        print("[Voice Engine] Initializing verbal feedback engine...")
        self.tts_engine = "en-GB-SoniaNeural"  

        print("[Voice Engine] Loading AI model... please wait.")
        try:
            self.piper_model = PiperVoice.load("path_to_piper_model.onnx", "path_to_model.json")
            print("[Voice Engine] Piper TTS loaded successfully.")
        except Exception as e:
            print(f"[Voice Engine Warning] Could not load Piper models ({e}). TTS fallback active.")
            self.piper_model = None
        print("[Voice Engine] Initializing memory-efficient openWakeWord monitor...")
        openwakeword.utils.download_models() # Safely downloads official free models if missing
        self.oww_model = OWWModel(wakeword_models=[self.wake_phrase])

        print("[Voice Engine] Loading AI model... please wait.")
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("[Voice Engine] Model loaded successfully.")
        
        self.mic_button = tk.Button(
            self.input_frame, text="🎙", bg="#12131b", fg="#a6adc8", bd=0,
            activebackground="#1e1e2e", activeforeground="#b4befe",
            font=("ONE DAY", 12), cursor="hand2",
            width=3, justify="center" 
        )
        self.mic_button.pack(side="right", padx=(0,5), pady=5)
        self.mic_button.config(command=self.toggle_listening)
        
        threading.Thread(target=self._wake_word_monitor_loop, daemon=True).start()


    def _say_reply(self, text):
        """Helper to say a word out loud safely using natural Edge AI voices."""
        def run_offline_tts():
            if self.piper_model:
                audio_stream = self.piper_model.synthesize_stream(text)
                for audio_bytes in audio_stream:
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    sd.play(audio_data, 16000)
                    sd.wait() 
            else:
                # Kept your original logic string variable as a fallback structure
                print(f"[TTS Fallback Engine] Piper missing. Attempting fallback print output: {text}")
                time.sleep(1.0)
                
        threading.Thread(target=run_offline_tts, daemon=True).start()
            
    def trigger_reread(self):
        """Clears text boxes and restarts recording immediately for corrections."""
        if hasattr(self.input_entry, "delete"):
            if isinstance(self.input_entry, tk.Text):
                self.input_entry.delete("1.0", tk.END)
            else:
                self.input_entry.delete(0, tk.END)
                
        threading.Thread(target=self._say_reply, args=("Try again.",), daemon=True).start()
        time.sleep(0.6)
        self.start_listening()

    def toggle_listening(self):
        if self.is_listening:
            self.stop_listening()
        else:
            self._say_reply("Yes, What's up?")
            time.sleep(0.8) 
            self.start_listening()

    def start_listening(self):
        if self.is_listening:
            return
        time.sleep(0.6)
        self.is_wake_word_active = False
        self.is_listening = True
        self.mic_button.config(text="🛑", fg="#f38ba8")
        print("[Voice] Actively recording your prompt...")
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop_listening(self):
        if not self.is_listening:
            return
        self.is_listening = False
        self.mic_button.config(text="🎙️", fg="#a6adc8")
 
        print("[Voice] Listening stopped.")
        self.root.after(0, lambda: self.input_entry.focus_set())

        def re_arm():
            if not self.awaiting_confirmation:
                self.is_wake_word_active = True
                print("[Wake Engine] Background listening re-armed.")
            else:
                print("[Wake Engine] Armed sleep kept: pending confirmation.")
        self.root.after(300, re_arm)
    
    def _wake_word_monitor_loop(self):
        """Runs silently in the background checking for the wake phrase."""

        fs = 16000
        chunk_size=1280 
        print(f"[Wake Engine] Background listening active. Say '{self.wake_phrase}'...")

        with sd.InputStream(samplerate=fs, channels=1, dtype='int16', blocksize=chunk_size) as stream:
            while True:
                if self.is_listening or not self.is_wake_word_active:
                    time.sleep(0.5)
                    continue
                    
                try:
                    audio_chunk, overflowed = stream.read(chunk_size)
                    audio_data = audio_chunk.flatten()
                    prediction = self.oww_model.predict(audio_data)
                    wake_score = prediction[self.wake_phrase]
                    
                    if wake_score > 0.5:
                        print(f"[Wake Engine] Wake word triggered! (Confidence: {wake_score:.2f})")
                        self.is_wake_word_active = False  
                        self._say_reply("Yes, whats up?")
                        time.sleep(0.8)
                        
                        self.root.after(0, self.start_listening)
                        continue
                except Exception as e:
                    self.is_wake_word_active = True
                    print(f"[Wake Engine Warning] Streaming loop error: {e}")
                    time.sleep(0.1)
                    pass

    def _listen_loop(self):
        fs = 16000  # Sample rate
        filename = "temp_voice.wav"

        chunk_size = 1600  # 0.1 seconds of audio data
        recorded_chunks = []

        silence_threshold = 400      # Lower this number if it cuts you off too early
        pause_limit_chunks = 20       # 8 chunks = 0.8 seconds of continuous quiet silence
        silent_chunk_counter = 0
        has_started_talking = False
        max_duration_chunks = 150
        try:
            time.sleep(0.1)
            with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
                for _ in range(max_duration_chunks):
                    if not self.is_listening:
                        break
                        
                    # Read 0.1 seconds of mic raw sound waves data
                    chunk, overflowed = stream.read(chunk_size)
                    recorded_chunks.append(chunk)
                    
                    # Calculate the audio amplitude volume strength score
                    volume_score = np.abs(chunk).mean()
                    
                    # VAD State Machine routing logic
                    if not has_started_talking:
                        if volume_score > silence_threshold:
                            has_started_talking = True
                    else:
                        if volume_score < silence_threshold:
                            silent_chunk_counter += 1
                        else:
                            silent_chunk_counter = 0 # Reset pause tracking if you continue talking
                        
                        # SMART CUT: If you stop speaking for 0.8 seconds, break out of loop immediately
                        if silent_chunk_counter >= pause_limit_chunks:
                            print("[Voice] Pause detected. Stopping capture early.")
                            break

            if not recorded_chunks or not has_started_talking:
                print("[Voice] No speech detected after wake command.")
                self.root.after(0, self.stop_listening)
                return

            audio_data = np.concatenate(recorded_chunks, axis=0)
            sf.write(filename, audio_data, fs)
            
            # Using faster-whisper to transcribe your main instruction phrase
            segments, info = self.model.transcribe(filename, beam_size=3, language="en", vad_filter=True, vad_parameters=dict(min_speech_duration_ms=50))
            text = "".join([segment.text for segment in segments])
            
            print(f"[Voice] Recognized text: {text}")
            self.root.after(0, self._handle_recognized_text, text)
                
        except Exception as e:
            print(f"[Voice Error] {e}")
            self.root.after(0, self.stop_listening)
        finally:
            if os.path.exists(filename):
                try: os.remove(filename)
                except: pass        

    def _handle_recognized_text(self, text):
        """Cleans up the text and directly drops it into the input entry box."""
        clean_text = text.strip()
        if not clean_text:
            self.stop_listening()
            return

        if self.awaiting_confirmation:
            # Clean out all commas and periods so "Yes, send." becomes "yes send"
            lowercase_text = clean_text.lower().replace(',', '').replace('.', '')
            words = lowercase_text.split()
            
            if words and words[0] == "yes":
                print("[Voice] Confirmation accepted. Transmitting command...")
                self.awaiting_confirmation = False
                self.stop_listening()
                
                # Broad list of words Whisper outputs when it mishears "send"
                send_sounds = ["send", "sen", "sand", "sent", "saying", "sound"]
                
                # Check if you said "yes send" (or any sound-alike)
                has_send_keyword = any(sound in words for sound in send_sounds)
                
                if has_send_keyword and self.pending_text:
                    self.send_command(None)
                return
                
            # Match keywords for corrections/mistakes
            elif "no" in words or "wrong" in words:
                print("[Voice] Correction requested. Restarting record chain.")
                self.awaiting_confirmation = False
                self.root.after(0, self.trigger_reread)
                return
                
            # If you said something completely different during confirmation, ask again
            else:
                threading.Thread(target=self._say_reply, args=("Please say yes or no.",), daemon=True).start()
                self.root.after(200, self.start_listening)
                return
        # Update the UI entry field with the clean text
        if hasattr(self.input_entry, "delete"):
            if isinstance(self.input_entry, tk.Text):
                self.input_entry.delete("1.0", tk.END)
                self.input_entry.insert("1.0", clean_text)
            else:
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, clean_text)

        # Stop the mic loop and reset the button colors
        self.pending_text = clean_text
        self.awaiting_confirmation = True
        self.is_listening = False
        self.mic_button.config(text="🎙️", fg="#a6adc8")
        self._say_reply("Is that all?")
        self.root.after(100, self.start_listening)
