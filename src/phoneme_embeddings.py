# phoneme_embeddings.py
import torch
from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence

def get_xphonebert_embeddings(text, text2phone_model, tokenizer, model):
    # Determine the device the model is on
    device = next(model.parameters()).device

    phonemes = text2phone_model.infer_sentence(text)

    # Tokenize - Creates tensors on CPU by default
    input_ids_dict = tokenizer(phonemes, return_tensors="pt")

    # --- Add this block to move tensors to the correct device ---
    input_ids_on_device = {
        key: tensor.to(device)
        for key, tensor in input_ids_dict.items()
    }
    # --- End of added block ---

    with torch.inference_mode():
        # Pass the tensors that are now on the correct device
        outputs = model(**input_ids_on_device)

    return outputs.last_hidden_state
