import sounddevice as sd
import queue
import threading
import time
import subprocess
from google.cloud import speech, texttospeech
from vertexai.preview.generative_models import GenerativeModel

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

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

class ChatLLM:
    def __init__(self):
        self.model = GenerativeModel("gemini-2.0-flash-001")
        self.chat = self.model.start_chat(history=[])

    def stream_message(self, user_input: str):
        try:
            response_stream = self.chat.send_message(user_input, stream=True)
            for chunk in response_stream:
                yield chunk.text
        except Exception as e:
            yield f"Error: {str(e)}"

class SafeMicTTS:
    def __init__(self):
        self.tts_client = texttospeech.TextToSpeechClient()
        self.running = False

    def start_mic_loop(self):
        """Dummy placeholder to match old interface if needed."""
        print("[System] Mic started via STT stream.")

    def stop_mic_loop(self):
        self.running = False
        print("[System] Mic stopped.")

    def play_audio(self, text: str):
        if not text.strip():
            return

        def _speak():
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
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "temp.mp3"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        threading.Thread(target=_speak, daemon=True).start()

class MicStreamToSTT:
    def __init__(self, rate=16000, chunk=1024):
        self.rate = rate
        self.chunk = chunk
        self.q = queue.Queue()
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
        self.q.put(bytes(indata))

    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
        print("[Mic] Microphone stream stopped.")

    def generator(self):
        while self.running:
            chunk = self.q.get()
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

def run_speech_chat_loop():
    mic_stream = MicStreamToSTT()
    gcp_stt = GCPSpeechToText()
    llm = ChatLLM()
    tts = SafeMicTTS()

    mic_stream.start()
    tts.start_mic_loop()

    try:
        audio_gen = mic_stream.generator()
        responses = gcp_stt.streaming_transcribe(audio_gen, sample_rate=SAMPLE_RATE)

        for response in responses:
            if not response.results:
                continue
            result = response.results[0]

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

    except Exception as e:
        print(f"[Error] {e}")
    finally:
        mic_stream.stop()
        tts.stop_mic_loop()

if __name__ == "__main__":
    run_speech_chat_loop()
