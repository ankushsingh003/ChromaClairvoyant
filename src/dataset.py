
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import collections

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return str(text).lower().split()

    def build_vocabulary(self, sentence_list):
        frequencies = collections.Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class TextOCRDataset(Dataset):
    def __init__(self, root_dir, img_csv, annot_csv, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load and process DataFrames
        self.imgs = pd.read_csv(img_csv)
        self.annots = pd.read_csv(annot_csv)
        
        print("Grouping annotations...")
        self.captions = self.annots.groupby('image_id')['utf8_string'].apply(lambda x: ' '.join(str(v) for v in x)).reset_index()
        
        # Merge with images df to ensure we have paths
        self.df = pd.merge(self.imgs, self.captions, left_on='id', right_on='image_id', how='inner')
        
        print(f"Found {len(self.df)} images with annotations.")
        
        # Initialize vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.df["utf8_string"].tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row['image_id']
        caption = row['utf8_string']
        img_path = os.path.join(self.root_dir, row['file_name'])

        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Image not found: {img_path}")
            img = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)

class CollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        try:
            # Diagnostic: Print shapes if there's an issue
            for i, item in enumerate(batch):
                if not isinstance(item[0], torch.Tensor):
                    print(f"ERROR: Item {i} image is not a tensor! Type: {type(item[0])}")
            
            imgs = [item[0].unsqueeze(0) for item in batch]
            
            # Check for shape consistency
            first_shape = imgs[0].shape
            for i, img in enumerate(imgs):
                if img.shape != first_shape:
                    print(f"SHAPE MISMATCH: Item {i} has shape {img.shape}, expected {first_shape}")

            imgs = torch.cat(imgs, dim=0)
            
            targets = [item[1] for item in batch]
            targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

            return imgs, targets
        except Exception as e:
            print(f"CollateFn Error: {e}")
            raise e

def get_loader(
    root_dir,
    img_csv,
    annot_csv,
    transform,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
):
    dataset = TextOCRDataset(root_dir, img_csv, annot_csv, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CollateFn(pad_idx=pad_idx),
    )

    return loader, dataset
