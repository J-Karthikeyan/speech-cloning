import torch
from torch.nn.utils.rnn import pad_sequence

def tacotron_collate_fn(batch):
    """
    batch: list of tuples -> (phoneme_embeddings [T, 768], mel_spec [M, 80], stop_token [M])
    """
    # print(batch)

    phoneme_seqs, mel_specs, stop_tokens = zip(*batch)

    # Pad phoneme sequences [T, 768] => [B, max_T, 768]
    phoneme_seqs = [seq for seq in phoneme_seqs]
    phoneme_lens = [seq.shape[0] for seq in phoneme_seqs]
    padded_phonemes = pad_sequence(phoneme_seqs, batch_first=True)  # [B, max_T, 768]

    # Pad mel spectrograms [M, 80] => [B, max_M, 80]
    mel_specs = [mel for mel in mel_specs]
    mel_lens = [mel.shape[0] for mel in mel_specs]
    padded_mels = pad_sequence(mel_specs, batch_first=True)  # [B, max_M, 80]

    # Pad stop tokens [M] => [B, max_M]
    stop_tokens = [s for s in stop_tokens]
    padded_stops = pad_sequence(stop_tokens, batch_first=True)  # [B, max_M]

    return padded_phonemes, padded_mels, padded_stops
