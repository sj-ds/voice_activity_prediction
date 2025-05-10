import asyncio
import websockets
# from microphone_stream import MicrophoneStream  # Save your class in this file or include it above


import pyaudio

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024


class MicrophoneStream:
    """Opens a recording stream as a generator yielding audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = None

    def __enter__(self):
        """Open the microphone stream."""
        self._audio_stream = self._audio_interface.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
        )
        return self

    def __exit__(self, type, value, traceback):
        """Close the microphone stream."""
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self._audio_interface.terminate()

    def generator(self):
        """Generates audio chunks for streaming."""
        while True:
            data = self._audio_stream.read(self._chunk, exception_on_overflow=False)
            yield data


WS_URI = "ws://localhost:8000/ws/audio"

async def stream_microphone():
    async with websockets.connect(WS_URI) as websocket:
        with MicrophoneStream(rate=16000, chunk=1024) as mic_stream:
            print("Streaming audio to server...")
            try:
                for chunk in mic_stream.generator():
                    await websocket.send(chunk)
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        print(response)
                    except asyncio.TimeoutError:
                        continue
            except KeyboardInterrupt:
                await websocket.send(b"__END__")
                print("Stopped by user.")

if __name__ == "__main__":
    asyncio.run(stream_microphone())

