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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from inference import tacotron_inference, plot_attention

BATCH_SIZE = 16
INITIAL_LEARNING_RATE = 1e-4  # Lower LR for resuming/fine-tuning
WEIGHT_DECAY = 1e-6  # Add weight decay
NUM_EPOCHS = 500  # Increased significantly for better training
SAVE_EVERY = 5

  # Save less frequently to avoid excessive checkpoints
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_TO_LOAD = "checkpoints/tacotron2_best_val.pt"  # Checkpoint to resume from

# Loss Functions
mel_loss_fn = nn.MSELoss()
stop_token_loss_fn = nn.BCEWithLogitsLoss()

def generate_and_save_attention_plot(model, sample_data, save_path, device, epoch_num):
    """Generates and saves an attention plot for a validation sample."""
    model.eval()  # Set to evaluation mode for inference
    try:
        with torch.no_grad():
            # Extract the first sample from the batch
            text_emb, _, _ = sample_data
            print(f"Batch shape: {text_emb.shape}")
            
            # Take just the first item from the batch
            text_emb_single = text_emb[0:1]  # Keep batch dimension but use only first example
            print(f"Single example shape: {text_emb_single.shape}")
            
            text_emb_single = text_emb_single.to(device)
            
            # Run inference on the single example
            print("Running inference for attention plot...")
            mel_output, alignment = tacotron_inference(model, text_emb_single, device, max_decoder_steps=600, stop_threshold=0.5)
            print(f"Generated mel shape: {mel_output.shape}, alignment shape: {alignment.shape}")
            
            # Plot the attention
            plot_attention(alignment, f"Validation Sample - Epoch {epoch_num}", save_path)
            print(f"Attention plot saved to {save_path}")
            
    except Exception as e:
        print(f"Could not generate attention plot: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace for better debugging
    finally:
        model.train()  # Set back to training mode

def train(model, optimizer, scheduler, train_loader, val_loader):
    best_val_loss = float('inf')
    
    # If checkpoint contains best_val_loss, load it
    if hasattr(scheduler, 'best_val_loss'):
        best_val_loss = scheduler.best_val_loss
    
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_epoch_mel_loss = 0
        train_epoch_stop_loss = 0
        num_train_batches = len(train_loader)
        
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        # Display current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.8f}")
        
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        
        for batch in progress_bar_train:
            text_emb, mel_target, stop_target = batch
            
            text_emb = text_emb.to(DEVICE)
            mel_target = mel_target.to(DEVICE)
            stop_target = stop_target.to(DEVICE)
            
            optimizer.zero_grad()
            mel_out, mel_postnet, stop_pred = model(text_emb, mel_target)  # Teacher forcing
            
            # Calculate losses
            mel_loss = mel_loss_fn(mel_out, mel_target) + mel_loss_fn(mel_postnet, mel_target)
            stop_loss = stop_token_loss_fn(stop_pred, stop_target)
            loss = mel_loss + stop_loss
            
            # Backpropagate and update weights
            loss.backward()
            # Gradient clipping
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
        model.eval()
        val_epoch_mel_loss = 0
        val_epoch_stop_loss = 0
        num_val_batches = len(val_loader)
        
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False)
        
        with torch.inference_mode():  # Disable gradient calculations
            for batch in progress_bar_val:
                text_emb, mel_target, stop_target = batch
                
                text_emb = text_emb.to(DEVICE)
                mel_target = mel_target.to(DEVICE)
                stop_target = stop_target.to(DEVICE)
                
                mel_out, mel_postnet, stop_pred = model(text_emb, mel_target)
                
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
        
        # Step the LR scheduler
        scheduler.step(avg_val_total_loss)
        
        # Save checkpoint dictionary with model state, optimizer state, etc.
        save_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'current_val_loss': avg_val_total_loss
        }
        
        # Regular checkpoint saving
        if (epoch + 1) % SAVE_EVERY == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/tacotron2_epoch_{epoch+1}.pt"
            torch.save(save_dict, save_path)
            print(f"Checkpoint saved to {save_path}")

            if fixed_val_batch:
                os.makedirs("visualizations", exist_ok=True)
                plot_save_path = f"visualizations/attention_epoch_{epoch+1}.png"
                generate_and_save_attention_plot(model, fixed_val_batch, plot_save_path, DEVICE, epoch + 1)
        
        # Best model saving
        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            os.makedirs("checkpoints", exist_ok=True)
            best_save_path = f"checkpoints/tacotron2_best_val.pt"
            save_dict['best_val_loss'] = best_val_loss
            torch.save(save_dict, best_save_path)
            print(f"*** Best validation loss improved ({best_val_loss:.4f}). Saved model to {best_save_path} ***")


if __name__ == "__main__":
    # Set multiprocessing start method
    if DEVICE.type == 'cuda':
        try:
            mp.set_start_method('spawn', force=True)
            print("Multiprocessing start method set to 'spawn'.")
        except RuntimeError as e:
            if "context has already been set" in str(e):
                print("Multiprocessing start method 'spawn' was already set.")
            else:
                print(f"Warning: Could not set start method 'spawn': {e}")
    
    # Define dataset paths
    AUDIO_DIR = "../data/VCTK-Corpus/VCTK-Corpus/wav48/p250"
    TRANSCRIPT_DIR = "../data/VCTK-Corpus/VCTK-Corpus/txt/p250"
    
    # Create datasets and dataloaders
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
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=tacotron_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # try:
    #     fixed_val_batch = next(iter(val_loader))
    #     print("Fixed validation batch obtained for plotting.")
    # except StopIteration:
    #     print("Validation loader is empty, cannot get fixed batch for plotting.")
    #     fixed_val_batch = None
    # except Exception as e:
    #     print(f"Error getting fixed validation batch: {e}")
    #     fixed_val_batch = None
    try:
    # Get a fixed validation batch for plotting attention
        val_iter = iter(val_loader)
        fixed_val_batch = next(val_iter)
        print(f"Fixed validation batch obtained with shapes: {fixed_val_batch[0].shape}, {fixed_val_batch[1].shape}")
    except Exception as e:
        print(f"Error getting fixed validation batch: {e}")
        fixed_val_batch = None
    
    # Initialize model
    model = Tacotron2(input_dim=768, mel_dim=80).to(DEVICE)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Initialize scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Load checkpoint if available
    start_epoch = 0
    if CHECKPOINT_TO_LOAD and os.path.exists(CHECKPOINT_TO_LOAD):
        try:
            print(f"Loading checkpoint: {CHECKPOINT_TO_LOAD}")
            checkpoint_data = torch.load(CHECKPOINT_TO_LOAD, map_location=DEVICE)
            
            # Check if it's a state_dict only checkpoint or a full checkpoint with optimizer, etc.
            if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
                model.load_state_dict(checkpoint_data['state_dict'])
                
                # Load optimizer state if available
                if 'optimizer' in checkpoint_data:
                    optimizer.load_state_dict(checkpoint_data['optimizer'])
                    print("Optimizer state loaded.")
                
                # Load scheduler state if available
                if 'scheduler' in checkpoint_data:
                    try:
                        scheduler.load_state_dict(checkpoint_data['scheduler'])
                        print("Scheduler state loaded.")
                    except Exception as e:
                        print(f"Warning: Could not load scheduler state: {e}. Reinitializing scheduler.")
                
                # Store best_val_loss for reference in training
                scheduler.best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
                
                # Get epoch for logging
                start_epoch = checkpoint_data.get('epoch', 0)
                print(f"Resuming from epoch {start_epoch}.")
            else:
                # Assuming it's just the model state_dict
                model.load_state_dict(checkpoint_data)
            
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Error loading checkpoint {CHECKPOINT_TO_LOAD}: {e}")
            print("Starting training from scratch.")
    else:
        print("No checkpoint specified or found. Starting training from scratch.")
    
    # Start training
    train(model, optimizer, scheduler, train_loader, val_loader)
