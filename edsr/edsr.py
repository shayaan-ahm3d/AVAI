from dataset import Div2kDataset, Mode
from edsr_model import Edsr
from edsr_utils import get_random_patch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import math
import random
from torchvision.transforms import ToTensor, Compose, Normalize

SCALE = 4 # Super-resolution factor
N_RESBLOCKS = 16 # Number of residual blocks
N_FEATS = 64 # Number of filters
PATCH_SIZE = 48 # LR patch size (HR patch size will be PATCH_SIZE * SCALE)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs/EDSR")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
low_path = Path("dataset/DIV2K_train_LR_x8")
high_path = Path("dataset/DIV2K_train_HR")

def train():
    print(f"EDSR Scale: x{SCALE})")
    model = Edsr(scale=SCALE, n_resblocks=N_RESBLOCKS, n_feats=N_FEATS).to(DEVICE)
    
    criterion = nn.L1Loss() # paper uses L1
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Note: We load raw images and crop patches in the loop to save memory
    transform = Compose([
        ToTensor(),
    ])
    dataset = Div2kDataset(low_path, high_path, transform, mode=Mode.TRAIN)
    print(f"Loaded {len(dataset)} images")

    print("Starting training: ")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Custom batching loop since we need to crop patches on the fly
        # Standard DataLoader might fail if images have different sizes
        # So we shuffle indices and process manually or use a custom collate
        
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for i in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[i:i + BATCH_SIZE]
            low_batch = []
            high_batch = []
            
            for idx in batch_indices:
                low, high = dataset[idx]
                
                # Ensure image is large enough for patch
                if low.shape[0] < PATCH_SIZE or low.shape[1] < PATCH_SIZE:
                    continue
                    
                low_patch, high_patch = get_random_patch(low, high, PATCH_SIZE, SCALE)
                
                low_batch.append(transform(low_patch))
                high_batch.append(transform(high_patch))
            
            if not low_batch:
                continue
                
            low_tensor = torch.stack(low_batch).to(DEVICE)
            high_tensor = torch.stack(high_batch).to(DEVICE)
            
            optimizer.zero_grad()
            output = model(low_tensor)
            loss = criterion(output, high_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / (len(dataset) // BATCH_SIZE)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), OUTPUT_DIR / f"edsr_x{SCALE}_epoch{epoch+1}.pth")