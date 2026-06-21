import os
import threading
import time
import tkinter as tk
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from sqlalchemy import text
from Static import font
import edge_tts  
import asyncio 

class VoiceSystem:
    """Manages speech recognition without using PyAudio."""
    
    def __init__(self, root, input_frame, input_entry, send_command):
        self.root = root
        self.input_frame = input_frame
        self.input_entry = input_entry
        self.send_command = send_command
        self.is_listening = False
        self.is_wake_word_active = True 
        self.wake_phrase = "nova"

        self.awaiting_confirmation = False
        self.pending_text = ""

        print("[Voice Engine] Initializing verbal feedback engine...")
        self.tts_engine = "en-GB-SoniaNeural"  

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
        def run_async_tts():
            async def speak():
                communicate = edge_tts.Communicate(text, self.tts_engine)
                await communicate.save("reply_temp.mp3")
                
                data, fs = sf.read("reply_temp.mp3")
                sd.play(data, fs)
                sd.wait()
                
                if os.path.exists("reply_temp.mp3"):
                    try: os.remove("reply_temp.mp3")
                    except: pass
            asyncio.run(speak())
        threading.Thread(target=run_async_tts, daemon=True).start()
            
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
            threading.Thread(target=self._say_reply, args=("Yes, What's up?",), daemon=True).start()
            time.sleep(0.4) # Small window so the mic doesn't record its own "Yes?"
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
        chunk_duration = 1.2 # Keeps a rolling 2.5-second audio window buffer
        
        print(f"[Wake Engine] Background listening active. Say '{self.wake_phrase}'...")

        while True:
            # If the user clicks the button or Ved is already processing a prompt, pause wake checking
            if self.is_listening or not self.is_wake_word_active:
                time.sleep(0.5)
                continue
                
            try:
                audio_data = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=1, dtype='float32')
                sd.wait()
                
                volume_score = np.abs(audio_data).mean()
                if volume_score < 0.01: # Silence gate threshold
                    continue
                self.is_wake_word_active = False  
                filename = "wake_chunk.wav"
                sf.write(filename, audio_data, fs)
                
                segments, info = self.model.transcribe(filename, beam_size=3, language="en")
                text = "".join([segment.text for segment in segments]).lower().strip()
                
                wake_detected = False
                
                # Clean out all punctuation marks from the AI text
                clean_text = text.replace('.', '').replace(',', '').replace('?', '').replace('!', '')
                clean_words = clean_text.split()
                
                if len(clean_words) <= 2 and "nova" in clean_words:
                    wake_detected = True
                    
                if wake_detected:
                    print(f"[Wake Engine] Wake word triggered! (Heard: '{text}')")
                    if os.path.exists(filename):
                        try: os.remove(filename)
                        except: pass

                    # Speak out loud first to let you know she is listening
                    self._say_reply("Yes, whats up?")
                    
                    # Open the active microphone loop
                    self.root.after(0, self.start_listening)
                    continue

                if os.path.exists(filename):
                    os.remove(filename)
                
                # If no wake word was found, turn the monitor engine back on
                self.is_wake_word_active = True

            except Exception as e:
                self.is_wake_word_active = True
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
