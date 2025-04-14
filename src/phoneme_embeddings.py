import torch
from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence

def get_xphonebert_embeddings(text, text2phone_model, tokenizer, model):
    phonemes = text2phone_model.infer_sentence(text)
    input_ids = tokenizer(phonemes, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**input_ids)
    return outputs.last_hidden_state  
