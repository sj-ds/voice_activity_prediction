import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.models as models
import pandas as pd
import os
from glob import glob

# Load pre-trained CPC model
cpc_model = models.CPC(input_dim=1, enc_dim=256, ar_dim=256)
cpc_model.eval()

# Function to extract CPC features and save to CSV
def extract_and_save_cpc_features(data_dir, output_csv, sample_rate=16000):
    audio_files = glob(os.path.join(data_dir, "*.mp3"))
    feature_list = []
    
    for file in audio_files:
        waveform, sr = torchaudio.load(file)
        if sr != sample_rate:
            resampler = T.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        with torch.no_grad():
            cpc_features = cpc_model(waveform.unsqueeze(0))

        feature_list.append({
            "audio_path": file,
            "features": cpc_features.squeeze(0).flatten().tolist()  # Flatten features
        })

    # Save extracted features to CSV
    df = pd.DataFrame(feature_list)
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")


