
import models.utils as utils

# Libraries
import math
import time
import torch

import torch.optim as optim
import matplotlib.pyplot as plt

from data.collators import AudioQACollator
from data.datasets import AudioQADataset
from data.processors import get_audio_processor
from data.processors import get_tokenizer
from models.audio_language_model import AudioLanguageModel

from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from models.config import TrainConfig, ALMConfig
#Otherwise, the tokenizer will through a warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.autograd.set_detect_anomaly(True)

device = "cpu"
print(f"Using device: {device}")


def get_dataloaders(train_cfg, alm_cfg, tokenizer):
    # Create datasets
    audio_processor = get_audio_processor(alm_cfg)

    combined_train_data = []
    for dataset_name in train_cfg.train_dataset_name:
        train_ds = load_dataset(
        path = train_cfg.train_dataset_path,
        name = dataset_name,
    )
        combined_train_data.append(train_ds['train'])
    train_ds = concatenate_datasets(combined_train_data)
    train_ds = train_ds.shuffle(seed=0) # Shuffle the training dataset, so train and val get equal contributions from all concatinated datasets

    # Apply cutoff if specified
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)  # Use the entire dataset
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)

    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    train_dataset = AudioQADataset(train_ds.select(range(train_size)), tokenizer, audio_processor)
    aqa_collator = AudioQACollator(tokenizer, alm_cfg.lm_max_length)
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=aqa_collator,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader

if __name__ == "__main__":
    train_cfg = TrainConfig()
    alm_cfg = ALMConfig()
    tokenizer = get_tokenizer(alm_cfg.lm_tokenizer)

    train_loader = get_dataloaders
    for batch in train_loader:
        audios = batch["audio"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        print(f"\ninput_ids.shape = {input_ids.shape}")
        print(f"labels.shape = {labels.shape}")