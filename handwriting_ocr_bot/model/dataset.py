"""
dataset.py — загрузка и аугментация датасета рукописного текста
Формат TSV: <имя_файла>\t<текст>
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


#алфавит + спец сиволы
ALPHABET = " !\"#%&'()*+,-./0123456789:;?АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ" \
           "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

#токен для CTC
BLANK_IDX = 0
NUM_CLASSES = len(ALPHABET) + 1

char2idx = {ch: idx + 1 for idx, ch in enumerate(ALPHABET)}
idx2char = {idx + 1: ch for idx, ch in enumerate(ALPHABET)}
idx2char[BLANK_IDX] = ""


def encode_text(text: str) -> list[int]:
    return [char2idx[ch] for ch in text if ch in char2idx]


def decode_ctc(indices: list[int]) -> str:
    result = []
    prev = BLANK_IDX
    for idx in indices:
        if idx != BLANK_IDX and idx != prev:
            result.append(idx2char.get(idx, ""))
        prev = idx
    return "".join(result)


class HandwritingDataset(Dataset):
    def __init__(
        self,
        tsv_path: str,
        data_dir: str,
        img_height: int = 64,
        img_width: int = 512,
        augment: bool = False,
    ):
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.samples = self._load_tsv(tsv_path)

        base_transforms = [
            T.Grayscale(num_output_channels=1),
            T.Resize((img_height, img_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]

        aug_transforms = [
            T.Grayscale(num_output_channels=1),
            T.Resize((img_height, img_width)),
            T.RandomAffine(
                degrees=3,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
                shear=3,
            ),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]

        self.transform = T.Compose(aug_transforms if augment else base_transforms)

    def _load_tsv(self, tsv_path: str) -> list[tuple[str, str]]:
        samples = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    filename, text = parts
                    filtered = "".join(ch for ch in text if ch in char2idx)
                    if filtered:
                        samples.append((filename, filtered))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filename, text = self.samples[idx]
        img_path = os.path.join(self.data_dir, filename)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.img_width, self.img_height), color=255)

        image = self.transform(image)
        label = torch.tensor(encode_text(text), dtype=torch.long)
        return image, label, text


def collate_fn(batch):
    images, labels, texts = zip(*batch)
    images = torch.stack(images, 0)

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)

    return images, labels_concat, label_lengths, texts