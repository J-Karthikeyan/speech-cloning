# dataset.py

import os
import torch
import torchaudio
from torch.utils.data import Dataset
from phoneme_embeddings import get_xphonebert_embeddings
from text2phonemesequence import Text2PhonemeSequence
from transformers import AutoTokenizer, AutoModel
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

class TacotronDataset(Dataset):
    def __init__(self, audio_dir, transcript_dir, sampling_rate=16000, file_ids=None):
        self.audio_dir = audio_dir
        self.transcript_dir = transcript_dir

        if file_ids is not None:
            self.file_ids = file_ids
        else:
            self.file_ids = sorted([
                f.split('.')[0] for f in os.listdir(audio_dir)
                if f.endswith('.wav')
            ])

        # Audio pre-processing
        self.sampling_rate = sampling_rate
        self.mel_transform = torch.nn.Sequential(
            MelSpectrogram(sample_rate=sampling_rate, n_fft=1024, hop_length=256, n_mels=80),
            AmplitudeToDB()
        )

        # XPhoneBERT setup
        self.phoneme_converter = Text2PhonemeSequence(language="eng-us", is_cuda=False)
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
        self.xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base").to("cpu")
        self.xphonebert.eval()

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]

        # Load text
        with open(os.path.join(self.transcript_dir, file_id + ".txt"), 'r') as f:
            text = f.read().strip()

        # Phoneme embeddings
        phoneme_embedding = get_xphonebert_embeddings(
            text=text,
            text2phone_model=self.phoneme_converter,
            tokenizer=self.tokenizer,
            model=self.xphonebert
        ).squeeze(0)  # Shape: [seq_len, emb_dim]

        # Load audio
        waveform, sr = torchaudio.load(os.path.join(self.audio_dir, file_id + ".wav"))
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            waveform = resampler(waveform)

        # Convert to Mel Spectrogram
        mel_spec = self.mel_transform(waveform).squeeze(0).transpose(0, 1)  # Shape: [time, n_mels]

        mel_tensor = mel_spec.clone().detach().float()
        stop_token = torch.zeros(mel_tensor.size(0), dtype=torch.float32)
        stop_token[-1] = 1.0

        return phoneme_embedding.clone().detach().float(), mel_tensor, stop_token
