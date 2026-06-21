import os
import threading
import time
import tkinter as tk
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import Static.font
class VoiceSystem:
    """Manages speech recognition without using PyAudio."""
    
    def __init__(self, root, input_frame, input_entry, send_command):
        self.root = root
        self.input_frame = input_frame
        self.input_entry = input_entry
        self.send_command = send_command
        self.is_listening = False
        self.is_wake_word_active = True # Controls whether the background listener is on
        self.wake_phrase = "ved"

        self.mic_button = tk.Button(
            self.input_frame, text="🎙", bg="#12131b", fg="#a6adc8", bd=0,
            activebackground="#1e1e2e", activeforeground="#b4befe",
            font=("ONE DAY", 12), cursor="hand2",
            width=3, justify="center" # <--- Forces a centered, fixed-width square box
        )
        self.mic_button.pack(side="right", padx=(0,5), pady=5)
        self.mic_button.config(command=self.toggle_listening)
        threading.Thread(target=self._wake_word_monitor_loop, daemon=True).start()

    def toggle_listening(self):
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()

    def start_listening(self):
        if self.is_listening:
            return
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
            self.is_wake_word_active = True
            print("[Wake Engine] Background listening re-armed.")

        self.root.after(300, re_arm)
    
    def _wake_word_monitor_loop(self):
        """Runs silently in the background checking for the wake phrase."""

        fs = 16000
        chunk_duration = 2.5 # Keeps a rolling 2.5-second audio window buffer
        recognizer = sr.Recognizer()
        
        print(f"[Wake Engine] Background listening active. Say '{self.wake_phrase}'...")

        while True:
            # If the user clicks the button or Ved is already processing a prompt, pause wake checking
            if self.is_listening or not self.is_wake_word_active:
                time.sleep(0.5)
                continue
                
            try:
                audio_data = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=1, dtype='int16')
                sd.wait()
                
                volume_score = np.abs(audio_data).mean()
                if volume_score < 400: # Silence gate threshold
                    continue
                    
                filename = "wake_chunk.wav"
                sf.write(filename, audio_data, fs)
                
                with sr.AudioFile(filename) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio).lower()
                    
                    # List of common words Google outputs when it mishears "Ved"
                    ved_sounds = ["vade","ved", "bed", "bade", "said", "red", "head", "then", "lead", "dead"]
                    
                    spoken_words = text.split()
                    wake_detected = False
                    
                    if spoken_words:
                        first_word = spoken_words[0]
                        if first_word in ved_sounds:
                            wake_detected = True
                    
                    if wake_detected:
                        print(f"[Wake Engine] Wake word triggered! (Heard: '{text}')")
                        self.root.after(0, self.start_listening)

                if os.path.exists(filename):
                    os.remove(filename)
                    
            except Exception:
                pass

    def _listen_loop(self):
        fs = 16000  # Sample rate
        filename = "temp_voice.wav"

        chunk_size = 1600  # 0.1 seconds of audio data
        recorded_chunks = []

        silence_threshold = 400      # Lower this number if it cuts you off too early
        pause_limit_chunks = 8       # 8 chunks = 0.8 seconds of continuous quiet silence
        silent_chunk_counter = 0
        has_started_talking = False
        max_duration_chunks = 100
        try:
            # Open a live background microphone audio stream block reader
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
            
            recognizer = sr.Recognizer()
            with sr.AudioFile(filename) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                print(f"[Voice] Recognized text: {text}")
                self.root.after(0, self._handle_recognized_text, text)
                
        except Exception as e:
            print(f"[Voice Error] {e}")
            self.root.after(0, self.stop_listening)
        finally:
            if os.path.exists("temp_voice.wav"):
                try: os.remove("temp_voice.wav")
                except: pass        

    def _handle_recognized_text(self, text):
        """Processes words, checks for trigger keywords, and fixes keyboard focus."""
        clean_text = text.strip()
        lowercase_text = clean_text.lower()
        
        enter_sounds = ["enter", "winter", "center", "inter", "hunter", "after"]
        send_sounds = ["send", "and", "end", "sent", "spend"]
        # Check if the sentence ends with your specific trigger keywords
        should_auto_send = False
        words = lowercase_text.split()
        if words:
            last_word = words[-1]
            if last_word in enter_sounds or last_word in send_sounds:
                should_auto_send = True
                orig_words = clean_text.split()
                clean_text = " ".join(orig_words[:-1])

        if hasattr(self.input_entry, "delete"):
            if isinstance(self.input_entry, tk.Text):
                self.input_entry.delete("1.0", tk.END)
                self.input_entry.insert("1.0", clean_text)
            else:
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, clean_text)

        self.stop_listening()

        if should_auto_send and clean_text:
            self.send_command(None)
