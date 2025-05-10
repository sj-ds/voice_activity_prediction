from speechbrain.pretrained import VAD
import torchaudio
import torchaudio.transforms as T
import torch
import pandas as pd

class MFCC_EXTRACTION():
    def __init__(self):
        # Load pre-trained SpeechBrain VAD
        self.vad_model = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")

    def extract_mfcc_features(self, audio_files, target_sample_rate= 16000, future_offset = 10, n_mfcc = 40, hop_length = 160):  # <--- ADD self
        mfcc_transform = T.MFCC(
            sample_rate=target_sample_rate, 
            n_mfcc=n_mfcc, 
            melkwargs={"n_fft": 400, "hop_length": hop_length, "n_mels": 40}
        )
        data_list = []

        for file in audio_files:
            # Load audio file
            waveform, sr = torchaudio.load(file)

            # Ensure the audio is in the correct sample rate (16kHz for VAD)
            if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
                waveform = resampler(waveform)

            waveform = waveform.float()
            mfcc = mfcc_transform(waveform).squeeze(0).T  # Shape: (time_steps, n_mfcc)
            waveform = waveform.unsqueeze(0)  # Shape: (1, channel, num_samples)

            speech_probs = self.vad_model.get_speech_prob_chunk(waveform.squeeze(0))
            speech_probs = torch.tensor(speech_probs).squeeze(0)

            binary_labels = (speech_probs > 0.5).float()
            min_length = min(mfcc.shape[0], binary_labels.shape[0])
            if min_length == 0:
                continue

            mfcc = mfcc[:min_length, :]
            binary_labels = binary_labels[:min_length]

            y_projected = torch.zeros_like(binary_labels)
            if min_length > future_offset:
                y_projected[:-future_offset] = binary_labels[future_offset:]
            else:
                y_projected = binary_labels

            data_list.append([mfcc.tolist(), y_projected.tolist()])

        df = pd.DataFrame(data_list, columns=['features', 'labels'])
        return df