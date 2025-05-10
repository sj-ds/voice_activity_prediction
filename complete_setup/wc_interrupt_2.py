import sounddevice as sd
import queue
import threading
import time
import subprocess
from google.cloud import speech, texttospeech
from vertexai.preview.generative_models import GenerativeModel

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024


# ---------------------------
# Google Cloud STT
# ---------------------------
class GCPSpeechToText:
    def __init__(self):
        self.client = speech.SpeechClient()

    def streaming_transcribe(self, requests, sample_rate=16000):
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="en-US",
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )
        return self.client.streaming_recognize(
            config=streaming_config,
            requests=requests
        )


# ---------------------------
# Vertex AI (Gemini)
# ---------------------------
class ChatLLM:
    def __init__(self):
        self.model = GenerativeModel("gemini-2.0-flash-001")

        # Manually injecting system prompt as first message
        system_instruction = "Act as a chatbot and do not give answers in more than 50 words."

        self.chat = self.model.start_chat(history=[])
        self.chat.send_message(system_instruction)

    def stream_message(self, user_input: str):
        try:
            response_stream = self.chat.send_message(user_input, stream=True)
            for chunk in response_stream:
                yield chunk.text
        except Exception as e:
            yield f"Error: {str(e)}"




# ---------------------------
# Google TTS + ffplay (Interruptible)
# ---------------------------
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


# ---------------------------
# Microphone Stream to STT
# ---------------------------
class MicStreamToSTT:
    def __init__(self, rate=16000, chunk=1024):
        self.rate = rate
        self.chunk = chunk
        self.q = queue.Queue(maxsize=20)
        self.running = False

    def start(self):
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.rate,
            channels=1,
            dtype='int16',
            blocksize=self.chunk,
            callback=self._callback,
        )
        self.stream.start()
        print("[Mic] Microphone stream started.")

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[Mic] Status: {status}")
        try:
            self.q.put_nowait(bytes(indata))
        except queue.Full:
            print("[Warning] Mic queue full. Dropping audio.")

    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
        print("[Mic] Microphone stream stopped.")

    def generator(self):
        while self.running:
            try:
                chunk = self.q.get(timeout=0.5)
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                print("[Warning] Sending silent audio to avoid timeout.")
                yield speech.StreamingRecognizeRequest(audio_content=b'\0' * self.chunk * 2)


# ---------------------------
# Main Loop
# ---------------------------
def run_speech_chat_loop():
    tts = SafeMicTTS()
    mic_stream = MicStreamToSTT()
    llm = ChatLLM()

    mic_stream.start()

    try:
        while True:
            print("[STT] Starting new streaming session...")
            audio_gen = mic_stream.generator()
            gcp_stt = GCPSpeechToText()
            responses = gcp_stt.streaming_transcribe(audio_gen, sample_rate=SAMPLE_RATE)

            try:
                for response in responses:
                    if not response.results:
                        continue

                    result = response.results[0]

                    # ✅ Only stop playback if actual text was recognized
                    if result.alternatives and result.alternatives[0].transcript.strip():
                        tts.stop_audio()

                    if result.is_final:
                        transcript = result.alternatives[0].transcript.strip()
                        print(f"\n[STT Final] {transcript}")

                        print("[Gemini] Generating response...\n")
                        response_text = ""
                        for chunk in llm.stream_message(transcript):
                            print(chunk, end='', flush=True)
                            response_text += chunk
                        print("\n" + "-" * 50)

                        tts.play_audio(response_text)

                    else:
                        interim = result.alternatives[0].transcript
                        print(f"[STT Interim] {interim}", end='\r')

            except Exception as stream_error:
                print(f"[STT Error] Restarting stream: {stream_error}")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[System] Exiting.")
    finally:
        mic_stream.stop()
        tts.stop_audio()


if __name__ == "__main__":
    run_speech_chat_loop()
