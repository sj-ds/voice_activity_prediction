import sounddevice as sd
import queue
import threading
import time
import subprocess
import pickle
import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np

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
        self.chat = self.model.start_chat(history=[])
        # Simulated system prompt
        self.chat.send_message("Act as a chatbot and do not give answers in more than 50 words.")

    def stream_message(self, user_input: str):
        try:
            response_stream = self.chat.send_message(user_input, stream=True)
            for chunk in response_stream:
                yield chunk.text
        except Exception as e:
            yield f"Error: {str(e)}"


# ---------------------------
# Google TTS + ffplay
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
# VAP Model
# ---------------------------
class VAPModel(nn.Module):
    def __init__(self, input_dim=40, lstm_hidden_dim=256, num_heads=8,
                 transformer_layers=4, lstm_layers=2, output_dim=1):
        super(VAPModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=transformer_layers
        )
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return torch.sigmoid(x)

    @staticmethod
    def load_model_pickle(path="vap_model_6.pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)
        model.eval()
        return model


class PredictionAverager:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.chunks_per_second = sample_rate // chunk_size
        self.predictions = []
        self.chunk_count = 0

    def update(self, prediction):
        self.predictions.append(prediction)
        self.chunk_count += 1
        if self.chunk_count >= self.chunks_per_second:
            avg = np.mean(self.predictions)
            print(f"[VAP] 1-second average prediction: {avg:.4f}")
            self.predictions = []
            self.chunk_count = 0
            return avg
        return None


# ---------------------------
# Microphone Stream
# ---------------------------
class MicStreamToSTT:
    def __init__(self, rate=16000, chunk=1024, mirror_queue=None):
        self.rate = rate
        self.chunk = chunk
        self.q = queue.Queue(maxsize=20)
        self.mirror_q = mirror_queue
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
        data = bytes(indata)
        try:
            self.q.put_nowait(data)
            if self.mirror_q:
                self.mirror_q.put_nowait(data)
        except queue.Full:
            pass

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
# VAP Monitoring Thread
# ---------------------------
def start_vap_monitoring(tts: SafeMicTTS, mic_queue: queue.Queue, model_path="vap_model_6.pkl"):
    model = VAPModel.load_model_pickle(model_path)
    averager = PredictionAverager()
    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
    )

    def _vap_thread():
        print("[VAP] Monitoring thread started.")
        while True:
            try:
                chunk = mic_queue.get(timeout=1)
            except queue.Empty:
                continue

            waveform = torch.from_numpy(
                np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            ).unsqueeze(0)

            mfcc = mfcc_transform(waveform).squeeze(0).T.unsqueeze(0)

            with torch.no_grad():
                output = model(mfcc)
                prediction = output.mean().item()
                avg = averager.update(prediction)
                if avg is not None and avg == 0.35:
                    print(f"[VAP] Threshold hit: {avg:.3f} — stopping TTS.")
                    tts.stop_audio()

    threading.Thread(target=_vap_thread, daemon=True).start()


# ---------------------------
# Main Chat Loop
# ---------------------------
def run_speech_chat_loop():
    tts = SafeMicTTS()
    vap_mirror_q = queue.Queue(maxsize=50)
    mic_stream = MicStreamToSTT(mirror_queue=vap_mirror_q)
    llm = ChatLLM()

    start_vap_monitoring(tts, vap_mirror_q)

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
