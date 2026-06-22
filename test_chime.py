import os
import winsound
import time

def play_my_downloaded_engine():
    # Looks for the real audio file you downloaded in your folder
    audio_filename = "turbo_engine_short.wav"
    
    if not os.path.exists(audio_filename):
        print(f"[Error] Could not find '{audio_filename}' in your directory!")
        print("Make sure you manually downloaded it from Pixabay, converted it to a WAV, and placed it here.")
        return

    try:
        print("[Audio Test] Playing your custom Pixabay engine roar asset...")
        # Uses native Windows drivers to stream your real V8 wave file cleanly
        winsound.PlaySound(audio_filename, winsound.SND_FILENAME | winsound.SND_ASYNC)
        
        # Keeps the test alive for 4 seconds so you can hear the roar
        time.sleep(4.0)
        print("[Audio Test] Playback complete.")
        
    except Exception as err:
        print(f"[Driver Error] Windows failed to parse the file structure: {err}")

if __name__ == "__main__":
    play_my_downloaded_engine()
