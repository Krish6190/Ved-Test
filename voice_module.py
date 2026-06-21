import threading
import tkinter as tk
import os
import numpy as np
class VoiceSystem:
    """Manages speech recognition without using PyAudio."""
    
    def __init__(self, root, input_frame, input_entry, send_command):
        self.root = root
        self.input_frame = input_frame
        self.input_entry = input_entry
        self.send_command = send_command
        self.is_listening = False
        
        self.mic_button = tk.Button(
            self.input_frame, text="🎙️", bg="#12131b", fg="#a6adc8", bd=0,
            activebackground="#1e1e2e", activeforeground="#b4befe",
            font=("Segoe UI", 12), cursor="hand2"
        )
        self.mic_button.pack(side="right", padx=(0,5), pady=5)
        self.mic_button.config(command=self.toggle_listening)

    def toggle_listening(self):
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()

    def start_listening(self):
        if self.is_listening:
            return
        self.is_listening = True
        self.mic_button.config(text="🛑", fg="#f38ba8")
        print("[Voice] Listening started...")
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop_listening(self):
        if not self.is_listening:
            return
        self.is_listening = False
        self.mic_button.config(text="🎙️", fg="#a6adc8")
        print("[Voice] Listening stopped.")
        self.root.after(0, lambda: self.input_entry.focus_set())

    def _listen_loop(self):
        try:
            import sounddevice as sd
            import soundfile as sf
            import speech_recognition as sr
        except ImportError:
            print("[Voice Error] Missing packages. Run: python -m pip install sounddevice soundfile numpy speech_recognition")
            self.root.after(0, self.stop_listening)
            return

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

            if not recorded_chunks:
                return

            # Flatten array segments together into a standard master wav sound track file
            audio_data = np.concatenate(recorded_chunks, axis=0)
            sf.write(filename, audio_data, fs)
            
            # Read the audio file using SpeechRecognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(filename) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                print(f"[Voice] Recognized text: {text}")
                self.root.after(0, self._handle_recognized_text, text)
                
        except Exception as e:
            print(f"[Voice Error] {e}")
        finally:
            # Clean up the temporary file safely
            if os.path.exists("temp_voice.wav"):
                try: os.remove("temp_voice.wav")
                except: pass        
        self.root.after(0, self.stop_listening)

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
            last_word = words[-1] # Check the absolute last word spoken
            
            # 1. Check if you tried to say "enter"
            if last_word in enter_sounds:
                should_auto_send = True
                # Safely slice off the phonetic word from the UI textbox display
                orig_words = clean_text.split()
                clean_text = " ".join(orig_words[:-1])
                
            # 2. Check if you tried to say "send"
            elif last_word in send_sounds:
                should_auto_send = True
                orig_words = clean_text.split()
                clean_text = " ".join(orig_words[:-1])

        # Update text grid contents
        self.input_entry.delete("1.0", tk.END)
        self.input_entry.insert("1.0", clean_text)
        
        # Snap typing cursor focus inside the box immediately
        self.input_entry.focus_set()
        
        # Only submit if you explicitly spoke the command words
        if should_auto_send and clean_text:
            self.send_command(None)
