import queue, os, threading, time
import torch
import torchaudio.transforms as T
import numpy as np

from vap_sound.model import VAPModel
from vap_sound.chat_llm import ChatLLM
from vap_sound.gcp_stt import GCPSpeechToText
from vap_sound.microphone import MicStreamToSTT
from vap_sound.gcp_tts import SafeMicTTS

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
FILE_PATH = "temp.mp3"



class PredictionAverager:
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        self.chunks_per_second = sample_rate // chunk_size
        self.predictions = []
        self.chunk_count = 0

    def update(self, prediction):
        self.predictions.append(prediction)
        self.chunk_count += 1
        if self.chunk_count >= self.chunks_per_second:
            avg = np.mean(self.predictions)/1.8
            print(f"[VAP] 1-second average prediction: {avg:.4f}")
            self.predictions = []
            self.chunk_count = 0
            return avg
        return None


###########################################################################################################
########################################  Model Prediction  ###############################################
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
                if avg is not None and avg >= 0.52:
                    print(f"[VAP] Threshold hit: {avg:.3f} — stopping TTS.")
                    tts.stop_audio()

    threading.Thread(target=_vap_thread, daemon=True).start()
###########################################################################################################



###########################################################################################################
########################################  Chat Loop  ######################################################
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
                        print(f"[VAP] Threshold hit: — stopping TTS.")
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
        if os.path.exists(FILE_PATH):
            try:
                os.remove(FILE_PATH)
                print(f"File '{FILE_PATH}' deleted successfully")
            except PermissionError:
                print(f"Permission denied to delete '{FILE_PATH}'")
            except Exception as e:
                print(f"Error occurred while deleting file: {e}")
        else:
            print(f"File '{FILE_PATH}' does not exist")
        print("\n[System] Exiting.")
    finally:
        mic_stream.stop()
        tts.stop_audio()
###########################################################################################################


if __name__ == "__main__":
    run_speech_chat_loop()
