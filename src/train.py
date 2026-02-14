
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import get_loader
from model import CNNtoRNN

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    root_dir = "d:/7GB_IMAGE_TO_TEXT_DATASET/train_val_images"
    img_csv = "d:/7GB_IMAGE_TO_TEXT_DATASET/img.csv"
    annot_csv = "d:/7GB_IMAGE_TO_TEXT_DATASET/annot.csv"

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = 10000 # Will strictly depend on dataset builder
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100
    load_model = False
    save_model = True
    train_CNN = False

    # Tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # Initialize loader first to get vocab size
    train_loader, dataset = get_loader(
        root_dir=root_dir,
        img_csv=img_csv,
        annot_csv=annot_csv,
        transform=transform,
        num_workers=2,
    )
    vocab_size = len(dataset.vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        # Load checkpoint logic here
        pass

    model.train()

    print("Starting Training...")
    for epoch in range(num_epochs):
        # Save model
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            torch.save(checkpoint, f"../models/my_checkpoint_epoch_{epoch}.pth.tar")
            print(f"Model saved at epoch {epoch}")

        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch}/{num_epochs}] Loss: {loss.item():.4f}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"An error occurred: {e}")
