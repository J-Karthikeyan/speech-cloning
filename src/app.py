from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence
from phoneme_embeddings import get_xphonebert_embeddings
from speaker_embeddings import get_speaker_embedding

'''
xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
text2phone_model = Text2PhonemeSequence(language='eng-us', is_cuda=True)

text = "The quick brown fox jumps over the lazy dog."
embeddings = get_xphonebert_embeddings(text, text2phone_model, tokenizer, xphonebert)
print(embeddings.shape)
'''

wav_path = "Ses04F_impro01_F009.wav"  
embedding = get_speaker_embedding(wav_path)

print(f"Speaker embedding shape: {embedding.shape}")
print(f"First 5 values: {embedding[:5]}")
