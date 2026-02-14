
import torch
import torchvision.transforms as transforms
from PIL import Image
from .model import CNNtoRNN
from .dataset import get_loader

def load_checkpoint(checkpoint, model, optimizer=None):
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

def generate_caption(image_path, model, vocabulary, transform, device):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    caption_words = model.caption_image(img_tensor, vocabulary)
    return ' '.join(caption_words)

def get_inference_model(vocab_size, checkpoint_path, embed_size=256, hidden_size=256, num_layers=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Re-initialize model architecture
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    if checkpoint_path and torch.cuda.is_available(): # logic to check file existence
        print(f"Loading checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        load_checkpoint(checkpoint, model)
    elif checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        load_checkpoint(checkpoint, model)
        
    return model, device





if __name__ == "__main__":
    import argparse
    import os
    from dataset import TextOCRDataset

    # Default paths
    ROOT_DIR = "d:/7GB_IMAGE_TO_TEXT_DATASET/train_val_images"
    IMG_CSV = "d:/7GB_IMAGE_TO_TEXT_DATASET/img.csv"
    ANNOT_CSV = "d:/7GB_IMAGE_TO_TEXT_DATASET/annot.csv"

    parser = argparse.ArgumentParser(description='Image Captioning Prediction')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to caption')
    parser.add_argument('--checkpoint_path', type=str, default='models/my_checkpoint.pth.tar', help='Path to model checkpoint')
    
    args = parser.parse_args()

    # check if image exists
    if not os.path.exists(args.image_path):
        print(f"Image not found at {args.image_path}")
        exit(1)

    # Define transform (same as training)
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    print("Loading vocabulary...")
    # We need the dataset to get the vocabulary
    # This might take a moment as it reads the CSVs
    try:
        dataset = TextOCRDataset(ROOT_DIR, IMG_CSV, ANNOT_CSV, transform=None)
        vocab = dataset.vocab
        vocab_size = len(vocab)
        print(f"Vocabulary loaded with {vocab_size} tokens.")
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        print("Using dummy vocabulary for testing...")
        class MockVocab:
            def __len__(self): return 100
            itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
            stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        vocab = MockVocab()
        vocab_size = 100

    print("Loading model...")
    model, device = get_inference_model(vocab_size, args.checkpoint_path if os.path.exists(args.checkpoint_path) else None)
    
    print(f"Generating caption for {args.image_path}...")
    try:
        caption = generate_caption(args.image_path, model, vocab, transform, device)
        print("-" * 30)
        print(f"Caption: {caption}")
        print("-" * 30)
    except Exception as e:
        print(f"Error during generation: {e}")
