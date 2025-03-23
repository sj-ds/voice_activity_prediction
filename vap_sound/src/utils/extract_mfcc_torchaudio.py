import torchaudio


# Define the VAP Model with MFCC Encoder
def extract_mfcc(audio_path, sample_rate, n_mfcc):
    "extract_mfcc function called"
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resample(waveform)
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, 
                                                melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40})
    mfcc = mfcc_transform(waveform)
    "extract_mfcc function completed"
    return mfcc.squeeze(0).T  # Transpose to match time-first format