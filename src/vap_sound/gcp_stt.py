from google.cloud import speech
    

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
