import queue
import sounddevice as sd
from google.cloud import speech

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