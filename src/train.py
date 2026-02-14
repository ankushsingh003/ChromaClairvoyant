import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .dataset import get_loader
from .model import CNNtoRNN

# Robust Logging Helper
def log(message):
    print(message, flush=True)
    with open("train_log.txt", "a") as f:
        f.write(message + "\n")

# Clear log at startup
if os.path.exists("train_log.txt"):
    os.remove("train_log.txt")

log(">>> src/train.py module loaded successfully.")

def train(dry_run=False):
    log(">>> Initializing Training Process...")
    # Dynamic path resolution
    base_dataset_path = "d:/7GB_IMAGE_TO_TEXT_DATASET"
    root_dir = os.path.join(base_dataset_path, "train_val_images")
    img_csv = os.path.join(base_dataset_path, "img.csv")
    annot_csv = os.path.join(base_dataset_path, "annot.csv")

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = 10000 
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 1 if dry_run else 100
    save_model = True

    # Create models directory if not exists
    if not os.path.exists("models"):
        os.makedirs("models")
        log("Created 'models' directory.")

    # Tensorboard
    writer = SummaryWriter("runs/flickr_dry_run" if dry_run else "runs/flickr")
    step = 0

    # Initialize loader
    log(f"Loading dataset from {base_dataset_path}...")
    try:
        train_loader, dataset = get_loader(
            root_dir=root_dir,
            img_csv=img_csv,
            annot_csv=annot_csv,
            transform=transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            num_workers=0, 
            batch_size=4 if dry_run else 32,
            shuffle=True
        )
        vocab_size = len(dataset.vocab)
        log(f"Dataset loaded. Vocab size: {vocab_size}. Batches: {len(train_loader)}")
    except Exception as e:
        log(f"FATAL: Failed to load dataset: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    
    # Initialize model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    log(f"Starting {'Dry Run' if dry_run else 'Full Training'}...")
    try:
        for epoch in range(num_epochs):
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            batches_processed = 0
            for idx, (imgs, captions) in loop:
                imgs = imgs.to(device)
                captions = captions.to(device) # Shape: (batch_size, seq_len)

                # SOS token is at captions[:, 0].
                # We want to input SOS to SOS+words and predict words+EOS
                outputs = model(imgs, captions[:, :-1])
                
                # IMPORTANT: In DecoderRNN, we cat 'features' to embeddings.
                # If captions[:, :-1] has length L, cat(features, embeddings) has length L+1.
                # outputs has shape (batch_size, L+1, vocab_size).
                # targets (captions[:, 1:]) has shape (batch_size, L).
                # So we take outputs[:, :-1, :] to match targets.
                
                targets = captions[:, 1:]
                
                # Check for same length
                if outputs.shape[1] != targets.shape[1]:
                    # Adjust outputs to match targets sequence length
                    outputs = outputs[:, :targets.shape[1], :]

                loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))

                writer.add_scalar("Training loss", loss.item(), global_step=step)
                step += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batches_processed += 1
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss=loss.item())

                if batches_processed % 1 == 0:
                    log(f"Batch {batches_processed} processed. Loss: {loss.item():.4f}")

                # Stop early for dry run
                if dry_run and batches_processed >= 5:
                    log(f"\n[DRY RUN] Processed {batches_processed} batches. Proceeding to save.")
                    break
            
            if save_model:
                os.makedirs("models", exist_ok=True)
                checkpoint_path = f"models/my_checkpoint_epoch_{epoch}.pth.tar"
                main_checkpoint_path = "models/my_checkpoint.pth.tar"
                
                log(f"Saving checkpoint to {checkpoint_path}...")
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                torch.save(checkpoint, checkpoint_path)
                torch.save(checkpoint, main_checkpoint_path)
                log("Successfully saved checkpoints.")

            if dry_run:
                break
    except Exception as e:
        log(f"ERROR during training loop: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a short training session to verify code")
    args = parser.parse_args()
    
    try:
        train(dry_run=args.dry_run)
    except Exception as e:
        log(f"Entry point error: {e}")
