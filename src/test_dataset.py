# test_dataset.py

from dataset import TacotronDataset

dataset = TacotronDataset(
    audio_dir="../data/VCTK-Corpus/VCTK-Corpus/wav48/p250",
    transcript_dir="../data/VCTK-Corpus/VCTK-Corpus/txt/p250"
)

phoneme_emb, mel_spec = dataset[0]
print("Phoneme Embedding Shape:", phoneme_emb.shape)  # [seq_len, 768]
print("Mel Spectrogram Shape:", mel_spec.shape)       # [frames, 80]
