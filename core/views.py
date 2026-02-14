
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os
import torch
import torchvision.transforms as transforms
from src.predict import get_inference_model, generate_caption
from src.dataset import TextOCRDataset  # To get vocab

# Global variables to cache model/vocab
MODEL = None
VOCAB = None
DEVICE = None
TRANSFORM = None

def load_model_resources():
    global MODEL, VOCAB, DEVICE, TRANSFORM
    if MODEL is None:
        print("Loading model and vocabulary...")
        
        # Define transform
        TRANSFORM = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        
        # Load dataset to get vocab
        # Ideally vocab should be pickled, but rebuilding is okay for this demo.
        # Updated path to be absolute or relative to project root if needed
        # Updated paths to be relative to BASE_DIR
        dataset_dir = os.environ.get('DATASET_DIR', os.path.join(BASE_DIR, "7GB_IMAGE_TO_TEXT_DATASET"))
        root_dir = os.path.join(dataset_dir, "train_val_images")
        img_csv = os.path.join(dataset_dir, "img.csv")
        annot_csv = os.path.join(dataset_dir, "annot.csv")
        
        try:
            if not os.path.exists(img_csv):
                raise FileNotFoundError(f"Dataset CSV not found at {img_csv}")
            dataset = TextOCRDataset(root_dir, img_csv, annot_csv)
            VOCAB = dataset.vocab
            
            # Load model
            # Check if checkpoint exists, else init random model
            # Checkpoint path relative to manage.py execution (root)
            checkpoint_path = "models/my_checkpoint.pth.tar" 
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found at {checkpoint_path}, using random model.")
                checkpoint_path = None # Will init random model
                
            MODEL, DEVICE = get_inference_model(
                vocab_size=len(VOCAB),
                checkpoint_path=checkpoint_path
            )
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading resources: {e}")
            # Fallback for empty vocab if dataset not found, to allow UI testing
            class MockVocab:
                def __len__(self): return 100
                itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
            VOCAB = MockVocab()
            MODEL, DEVICE = get_inference_model(vocab_size=100, checkpoint_path=None)

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        # Ensure model is loaded
        load_model_resources()
        
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        # Save temp file
        path = default_storage.save(f"tmp/{image_file.name}", ContentFile(image_file.read()))
        full_path = os.path.join(settings.MEDIA_ROOT, path)
        
        try:
            caption = generate_caption(full_path, MODEL, VOCAB, TRANSFORM, DEVICE)
            # Cleanup
            if os.path.exists(full_path):
                os.remove(full_path)
                
            return JsonResponse({'caption': caption})
        except Exception as e:
            print(f"Prediction error: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid method'}, status=405)
