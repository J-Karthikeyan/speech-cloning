import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm
from model import Tacotron2
from dataset import TacotronDataset  
from torch.utils.data import DataLoader
from collate_fn import tacotron_collate_fn
from sklearn.model_selection import train_test_split

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
SAVE_EVERY = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Tacotron2(input_dim=768, mel_dim=80).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train():
    best_val_loss = float('inf') # Keep track of best validation loss for saving maybe

    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train() # Set model to training mode
        train_epoch_mel_loss = 0
        train_epoch_stop_loss = 0
        num_train_batches = len(train_loader)

        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)

        for batch in progress_bar_train:
            text_emb, mel_target, stop_target = batch

            text_emb = text_emb.to(DEVICE)
            mel_target = mel_target.to(DEVICE)
            stop_target = stop_target.to(DEVICE)

            optimizer.zero_grad()
            mel_out, mel_postnet, stop_pred = model(text_emb, mel_target) # Teacher forcing

            # Calculate losses
            mel_loss = mel_loss_fn(mel_out, mel_target) + mel_loss_fn(mel_postnet, mel_target)
            stop_loss = stop_token_loss_fn(stop_pred, stop_target)
            loss = mel_loss + stop_loss

            # Backpropagate and update weights
            loss.backward()
            # Optional: Gradient clipping (helps prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses
            train_epoch_mel_loss += mel_loss.item()
            train_epoch_stop_loss += stop_loss.item()

            # Update progress bar
            progress_bar_train.set_postfix({
                "Mel Loss": f"{mel_loss.item():.4f}",
                "Stop Loss": f"{stop_loss.item():.4f}",
                "Total Loss": f"{loss.item():.4f}"
            })

        avg_train_mel_loss = train_epoch_mel_loss / num_train_batches
        avg_train_stop_loss = train_epoch_stop_loss / num_train_batches
        avg_train_total_loss = avg_train_mel_loss + avg_train_stop_loss

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        val_epoch_mel_loss = 0
        val_epoch_stop_loss = 0
        num_val_batches = len(val_loader)

        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False)

        with torch.inference_mode(): # Disable gradient calculations
            for batch in progress_bar_val:
                text_emb, mel_target, stop_target = batch

                text_emb = text_emb.to(DEVICE)
                mel_target = mel_target.to(DEVICE)
                stop_target = stop_target.to(DEVICE)

                mel_out, mel_postnet, stop_pred = model(text_emb, mel_target) # Still use teacher forcing for comparable loss

                # Calculate losses
                mel_loss = mel_loss_fn(mel_out, mel_target) + mel_loss_fn(mel_postnet, mel_target)
                stop_loss = stop_token_loss_fn(stop_pred, stop_target)

                # Accumulate losses
                val_epoch_mel_loss += mel_loss.item()
                val_epoch_stop_loss += stop_loss.item()

                 # Update progress bar
                progress_bar_val.set_postfix({
                    "Mel Loss": f"{mel_loss.item():.4f}",
                    "Stop Loss": f"{stop_loss.item():.4f}"
                })

        avg_val_mel_loss = val_epoch_mel_loss / num_val_batches
        avg_val_stop_loss = val_epoch_stop_loss / num_val_batches
        avg_val_total_loss = avg_val_mel_loss + avg_val_stop_loss

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Mel Loss: {avg_train_mel_loss:.4f}, Train Stop Loss: {avg_train_stop_loss:.4f}, Train Total Loss: {avg_train_total_loss:.4f}")
        print(f"  Val Mel Loss:   {avg_val_mel_loss:.4f}, Val Stop Loss:   {avg_val_stop_loss:.4f}, Val Total Loss:   {avg_val_total_loss:.4f}")

        # --- Checkpoint Saving ---
        if (epoch + 1) % SAVE_EVERY == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/tacotron2_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

        # Optional: Save best model based on validation loss
        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            os.makedirs("checkpoints", exist_ok=True)
            best_save_path = f"checkpoints/tacotron2_best_val.pt"
            torch.save(model.state_dict(), best_save_path)
            print(f"*** Best validation loss improved. Saved model to {best_save_path} ***")


if __name__ == "__main__":
    # --- Add this block ---
    # Set the start method for multiprocessing BEFORE any CUDA operations
    # or DataLoader instantiation if possible.
    if DEVICE.type == 'cuda': # Only set if using CUDA
        try:
            # Use 'spawn' instead of 'fork' to avoid CUDA issues with multiprocessing
            mp.set_start_method('spawn', force=True)
            print("Multiprocessing start method set to 'spawn'.")
        except RuntimeError as e:
            # Catch exception if context is already set (e.g., in interactive environments)
            if "context has already been set" in str(e):
                 print("Multiprocessing start method 'spawn' was already set.")
            else:
                 print(f"Warning: Could not set start method 'spawn': {e}")
    # --- End of added block ---

    # --- The rest of your main execution block ---
    AUDIO_DIR = "../data/VCTK-Corpus/VCTK-Corpus/wav48/p250"
    TRANSCRIPT_DIR = "../data/VCTK-Corpus/VCTK-Corpus/txt/p250"

    all_file_ids = sorted([
        f.split('.')[0] for f in os.listdir(AUDIO_DIR)
        if f.endswith('.wav')
    ])

    train_ids, val_ids = train_test_split(all_file_ids, test_size=0.1, random_state=42)

    train_dataset = TacotronDataset(AUDIO_DIR, TRANSCRIPT_DIR, file_ids=train_ids)
    val_dataset = TacotronDataset(AUDIO_DIR, TRANSCRIPT_DIR, file_ids=val_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=tacotron_collate_fn,
        num_workers=4, # Keep your desired number of workers
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=tacotron_collate_fn,
        num_workers=4, # Keep your desired number of workers
        pin_memory=True
    )

    model = Tacotron2(input_dim=768, mel_dim=80).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    mel_loss_fn = nn.MSELoss()
    stop_token_loss_fn = nn.BCEWithLogitsLoss()

    # Make sure the train function is defined before calling it
    # (Paste your full train function definition here if it wasn't above)

    # Optional: Load checkpoint if resuming training
    # ...

    train() # Call your training function
