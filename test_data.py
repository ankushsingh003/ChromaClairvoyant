import torch
from src.dataset import get_loader
import torchvision.transforms as transforms
import os

def test():
    print(">>> Starting Dataset Test...")
    base_dataset_path = "d:/7GB_IMAGE_TO_TEXT_DATASET"
    root_dir = os.path.join(base_dataset_path, "train_val_images")
    img_csv = os.path.join(base_dataset_path, "img.csv")
    annot_csv = os.path.join(base_dataset_path, "annot.csv")

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    try:
        loader, dataset = get_loader(
            root_dir=root_dir,
            img_csv=img_csv,
            annot_csv=annot_csv,
            transform=transform,
            batch_size=4,
            num_workers=0
        )
        print(f"Dataset size: {len(dataset)}")
        
        # Pull one batch
        it = iter(loader)
        imgs, targets = next(it)
        
        print(f"Batch loaded successfully!")
        print(f"Images shape: {imgs.shape}")
        print(f"Targets shape: {targets.shape}")
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
