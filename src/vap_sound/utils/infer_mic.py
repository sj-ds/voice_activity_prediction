import pyaudio
import torch
import torchaudio.transforms as T
import numpy as np
import time
import os

# Real-time inference from microphone, streaming to a function
def infer_from_mic_stream(model, callback_function, sample_rate=16000, chunk_size=1024, n_mfcc=40, hop_length=160):
# def infer_from_mic_stream(model, sample_rate=16000, chunk_size=1024):
    """
    Streams microphone audio to a callback function for real-time inference.

    Args:
        model: The PyTorch model for inference.
        callback_function: A function that will receive the model's prediction.
        sample_rate: The audio sample rate.
        chunk_size: The size of audio chunks to process.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk_size)
    mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc,
                            melkwargs={"n_fft": 400, "hop_length": hop_length, "n_mels": n_mfcc})

    try:
        while True:
            audio_data = stream.read(chunk_size)
            waveform = torch.from_numpy(np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0).unsqueeze(0)
            mfcc = mfcc_transform(waveform).squeeze(0).T.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                output = model(mfcc)
                prediction = output.squeeze(0).cpu().numpy()
                array = np.array(prediction)
                mean = np.mean(array)
                avg = mean.item()
                # print(avg)
                # print(prediction)
                callback_function(avg)  # Call the provided function with the prediction

    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()




LOG_FILE = os.environ.get("MODEL_PREDICTION_FILE")

def write_to_file_pred(prediction):
    log_file = LOG_FILE
    
    # Create the log file if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("PREDICTION LOG\n")  # Add a header

    # Append new prediction with timestamp
    # cur_time = time.strftime("%Y-%m-%d %H:%M:%S")  # Readable timestamp
    cur_time = time.time()
    with open(log_file, "a") as f:  # Append mode
        f.write(f"{cur_time} Prediction: {prediction}\n")


def remove_log_file():
    """Removes the log file after execution."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print(f"Deleted log file: {LOG_FILE}")


# infer_from_mic_stream(model)


