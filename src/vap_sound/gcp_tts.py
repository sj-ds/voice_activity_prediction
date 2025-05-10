import threading
import subprocess
import time
from google.cloud import texttospeech


class SafeMicTTS:
    def __init__(self):
        self.tts_client = texttospeech.TextToSpeechClient()
        self.playback_process = None
        self.lock = threading.Lock()
        self.playing = False
        self.play_start_time = 0

    def stop_audio(self):
        with self.lock:
            if self.playing and self.playback_process and self.playback_process.poll() is None:
                print("[TTS] Interrupting playback.")
                self.playback_process.terminate()
                self.playback_process = None
                self.playing = False

    def play_audio(self, text: str):
        if not text.strip():
            return

        def _speak():
            try:
                synthesis_input = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US",
                    name="en-US-Wavenet-D"
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                response = self.tts_client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )

                with open("temp.mp3", "wb") as out:
                    out.write(response.audio_content)

                print(f"[TTS] Playing: {text}")
                with self.lock:
                    self.playing = True
                    self.play_start_time = time.time()
                    self.playback_process = subprocess.Popen(
                        ["ffplay", "-nodisp", "-autoexit", "temp.mp3"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )

                self.playback_process.wait()

            finally:
                with self.lock:
                    self.playing = False
                    self.playback_process = None

        threading.Thread(target=_speak, daemon=True).start()