# inference.py

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.io.wavfile import write as write_wav # Or use soundfile

# Your own project imports
from model import Tacotron2 # Make sure model.py is accessible
from dataset import TacotronDataset # Only needed for phoneme tools setup
from phoneme_embeddings import get_xphonebert_embeddings

# Dependencies for phoneme conversion (reuse from dataset)
from text2phonemesequence import Text2PhonemeSequence
from transformers import AutoTokenizer, AutoModel

# --- Vocos Vocoder Loading ---
from vocos import Vocos # <-- Import Vocos

def load_vocos_vocoder(device, model_name="patriotyk/vocos-mel-hifigan-compat-44100khz"):
    """Loads a Vocos vocoder from Hugging Face."""
    print(f"Loading Vocos model: {model_name}...")
    try:
        # Load the model and move it to the specified device
        vocoder = Vocos.from_pretrained(model_name).to(device)
        print("Vocos vocoder loaded successfully.")
        # Vocos model handles eval mode internally when using decode,
        # but setting it explicitly doesn't hurt.
        vocoder.eval()
        return vocoder
    except Exception as e:
        print(f"Error loading Vocos model {model_name}: {e}")
        print("Please ensure you have installed 'vocos' (`pip install vocos`) and have internet access.")
        exit(1)


# --- Tacotron 2 Inference Function ---
# (Keep your existing tacotron_inference function exactly as it is)
def tacotron_inference(model, text_embeddings, device, max_decoder_steps=1000, stop_threshold=0.5):
    """
    Generates mel spectrogram from text embeddings using Tacotron 2 autoregressively.
    """
    model.eval()
    print("Starting Tacotron 2 inference...")

    encoder_outputs = model.encoder(text_embeddings)

    # Initialize decoder states
    B = text_embeddings.size(0) # Should be 1 for inference
    mel_dim = model.decoder.mel_proj.out_features
    encoder_dim = model.encoder.lstm.hidden_size * 2 # Since bidirectional
    decoder_dim = model.decoder.attention_rnn.hidden_size

    h_att, c_att = torch.zeros(B, decoder_dim).to(device), torch.zeros(B, decoder_dim).to(device)
    h_dec, c_dec = torch.zeros(B, decoder_dim).to(device), torch.zeros(B, decoder_dim).to(device)
    context = torch.zeros(B, encoder_dim).to(device)
    attention_weights_cumulative = torch.zeros(B, encoder_outputs.size(1)).to(device)
    mel_input = torch.zeros(B, mel_dim).to(device) # Initial "GO" frame

    mel_outputs = []
    alignments = []
    stop_tokens = []

    with torch.no_grad():
        for t in range(max_decoder_steps):
            prenet_out = model.decoder.prenet(mel_input)
            att_input = torch.cat((prenet_out, context), dim=-1)
            h_att, c_att = model.decoder.attention_rnn(att_input, (h_att, c_att))
            context, attn_weights = model.decoder.attention_layer(h_att, encoder_outputs, attention_weights_cumulative)

            dec_input = torch.cat((h_att, context), dim=-1)
            h_dec, c_dec = model.decoder.decoder_rnn(dec_input, (h_dec, c_dec))

            proj_input = torch.cat((h_dec, context), dim=-1)
            mel_frame = model.decoder.mel_proj(proj_input) # Shape: (B, mel_dim)
            stop_pred = model.decoder.stop_proj(proj_input) # Shape: (B, 1)

            mel_outputs.append(mel_frame.unsqueeze(1)) # Append as (B, 1, mel_dim)
            alignments.append(attn_weights.unsqueeze(1)) # Store alignments as (B, 1, T_in)
            stop_tokens.append(stop_pred)
            attention_weights_cumulative = attention_weights_cumulative + attn_weights.detach()
            mel_input = mel_frame # Keep using (B, mel_dim) as input to prenet

            if torch.sigmoid(stop_pred.squeeze()) > stop_threshold:
                print(f"Stop condition met at step {t+1}")
                break
        else: # Executed if loop finishes without break
             print(f"Warning: Max decoder steps ({max_decoder_steps}) reached.")

    if not mel_outputs:
        print("Warning: No mel frames generated.")
        return torch.zeros(B, 0, mel_dim).to(device), np.zeros((0, encoder_outputs.size(1)))

    mel_outputs = torch.cat(mel_outputs, dim=1) # -> (B, T_out, mel_dim)
    alignments = torch.cat(alignments, dim=1) # -> (B, T_out, T_in)
    alignments_plot = alignments.squeeze(0).cpu().numpy()
    mel_outputs_postnet = mel_outputs + model.postnet(mel_outputs)

    print("Tacotron 2 inference finished.")
    return mel_outputs_postnet, alignments_plot # Return postnet output

# --- Plotting Function ---
# (Keep your existing plot_attention function exactly as it is)
def plot_attention(alignment, text, output_path):
    """Plots the attention alignment."""
    print(f"Plotting attention alignment to {output_path}...")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none', cmap='viridis')

    fig.colorbar(im, ax=ax)
    xlabel = 'Encoder Steps (Phonemes)'
    ylabel = 'Decoder Steps (Mel Frames)'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Attention Alignment: {text[:80]}...") # Display part of the text
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() # Close the plot to free memory
    print("Attention plot saved.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tacotron 2 Inference")
    parser.add_argument("--tacotron_checkpoint", type=str, required=True,
                        help="Path to the Tacotron 2 model checkpoint (.pt file).")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize.")
    parser.add_argument("--output_wav", type=str, required=True,
                        help="Path to save the output WAV file.")
    parser.add_argument("--output_attn", type=str, default="attention.png",
                        help="Path to save the attention plot PNG (optional).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('cuda' or 'cpu'). Auto-detects if not set.")
    parser.add_argument("--max_decoder_steps", type=int, default=1500,
                        help="Maximum number of decoder steps.")
    parser.add_argument("--stop_threshold", type=float, default=0.5,
                        help="Threshold for stop token prediction.")
    # --- Important: Update default SR ---
    parser.add_argument("--vocoder_sr", type=int, default=44100, # <-- Set to 44100
                        help="Expected sampling rate of the vocoder output (default: 44100 for patriotyk/vocos-mel-hifigan-compat-44100khz).")

    args = parser.parse_args()

    # --- Setup Device ---
    if args.device:
        DEVICE = torch.device(args.device)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Load Tacotron 2 Model ---
    print("Loading Tacotron 2 model...")
    # Ensure model dimensions match your trained model
    model = Tacotron2(input_dim=768, mel_dim=80).to(DEVICE) # Correct mel_dim=80
    try:
        checkpoint = torch.load(args.tacotron_checkpoint, map_location=DEVICE)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {args.tacotron_checkpoint}")
    except FileNotFoundError:
        print(f"Error: Tacotron checkpoint not found at {args.tacotron_checkpoint}")
        exit(1)
    except Exception as e:
         print(f"Error loading Tacotron state_dict: {e}")
         print("Ensure the model definition matches the checkpoint.")
         exit(1)
    model.eval()

    # --- Load Phoneme Conversion Tools ---
    print("Loading phoneme conversion tools...")
    try:
        phoneme_converter = Text2PhonemeSequence(language="eng-us", is_cuda=(DEVICE.type == 'cuda'))
        tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
        xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base").to(DEVICE)
        xphonebert.eval()
    except Exception as e:
        print(f"Error loading phoneme tools: {e}")
        exit(1)

    # --- Load Vocoder using Vocos ---
    # Use the new function to load the vocoder
    vocoder = load_vocos_vocoder(DEVICE) # No need to pass model name if using default

    # --- Prepare Input Text ---
    print(f"Processing text: \"{args.text}\"")
    try:
        with torch.no_grad():
             text_embeddings = get_xphonebert_embeddings(
                 text=args.text,
                 text2phone_model=phoneme_converter,
                 tokenizer=tokenizer,
                 model=xphonebert
             ).to(DEVICE)
    except Exception as e:
        print(f"Error getting phoneme embeddings: {e}")
        exit(1)

    # --- Run Inference ---
    with torch.no_grad():
        mel_spec_postnet, alignment = tacotron_inference(
            model,
            text_embeddings,
            DEVICE,
            args.max_decoder_steps,
            args.stop_threshold
        )

        # Prepare mel spectrogram for vocoder (needs shape [B, mel_dim, T_out])
        # mel_spec_postnet is already [B, T_out, mel_dim] -> [1, T_out, 80]
        # Vocos expects [B, mel_dim, T_out] -> [1, 80, T_out]
        mel_for_vocoder = mel_spec_postnet.transpose(1, 2) # Swap time and mel dim

        # Run Vocoder
        print("Running Vocoder...")
        # Use the decode method provided by Vocos
        waveform = vocoder.decode(mel_for_vocoder)

    # --- Save Output ---
    output_waveform = waveform.squeeze().cpu().detach().numpy()

    # --- Crucial: Check Sampling Rate ---
    # Ensure the saving SR matches the vocoder's output SR
    output_sr = args.vocoder_sr
    print(f"Saving audio with sampling rate: {output_sr}")
    # If you were using a different vocoder model name, you might need to
    # dynamically determine the SR from the vocoder object if possible,
    # but for this specific model, 44100 is correct.

    try:
        write_wav(args.output_wav, output_sr, output_waveform)
        print(f"Audio saved to {args.output_wav}")
    except Exception as e:
        print(f"Error saving WAV file: {e}")

    # --- Plot Attention (Optional) ---
    if args.output_attn:
        try:
            plot_attention(alignment, args.text, args.output_attn)
        except Exception as e:
            print(f"Error plotting attention: {e}")

    print("Inference complete.")
