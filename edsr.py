from dataset import Div2kDataset, Mode
from edsr_model import Edsr
from edsr_utils import get_random_patch, log_metrics
from inr_utils import ssim

from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import Optimizer, Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose
from skimage.metrics import peak_signal_noise_ratio
from lpips import LPIPS

SCALE = 4 # Super-resolution factor
N_RESBLOCKS = 16 # Number of residual blocks
N_FEATS = 64 # Number of filters
PATCH_SIZE = 24 # Low patch size (High patch size will be PATCH_SIZE * SCALE)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
VAL_EVERY = 100
CHECKPOINT_EVERY = 10

low_path = Path("dataset/DIV2K_train_LR_x8")
high_path = Path("dataset/DIV2K_train_HR")
test_low_path = Path("dataset/DIV2K_valid_LR_x8")
test_high_path = Path("dataset/DIV2K_valid_HR")

def collate_fn(batch):
    """
    To be able to use DataLoader with random patches
    """
    low_tensors = []
    high_tensors = []
    for low, high in batch:
        if low.shape[1] < PATCH_SIZE or low.shape[2] < PATCH_SIZE:
            continue
        
        low_patch, high_patch = get_random_patch(low, high, PATCH_SIZE, SCALE)
        low_tensors.append(low_patch)
        high_tensors.append(high_patch)
    
    if not low_tensors:
        return None
        
    return torch.stack(low_tensors), torch.stack(high_tensors)

@torch.no_grad()
def validate(model: Module, dataloader: DataLoader) -> tuple[float, float]:
    """Validation loop, returns mean PSNR and SSIM"""
    model.eval()

    acc_psnr = 0.0
    acc_ssim = 0.0
    
    for i, (low, high) in enumerate(dataloader):
        low_tensor: Tensor = low.to(DEVICE)
        
        output: Tensor = model(low_tensor)
        
        # Metrics need NumpPy array, (B, C, H, W) -> (H, W, C)
        output_np: np.ndarray = output.squeeze(0).cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        high_np: np.ndarray = high.squeeze(0).numpy().transpose(1, 2, 0)
        
        acc_psnr += peak_signal_noise_ratio(high_np, output_np, data_range=1.0)
        acc_ssim += ssim(high_np, output_np, data_range=1.0)

    mean_psnr: float = acc_psnr / len(dataloader)
    mean_ssim: float = acc_ssim / len(dataloader)

    return mean_psnr, mean_ssim

@torch.no_grad()
def test(model: Module, dataloader: DataLoader) -> tuple[float, float, float]:
    """Testing loop, returns mean PSNR, SSIM & LPIPS"""
    model.eval()
    lpips_model: LPIPS = LPIPS().to(DEVICE).eval()

    acc_psnr = 0.0
    acc_ssim = 0.0
    acc_lpips = 0.0

    for i, (low, high) in enumerate(dataloader):
        low_tensor: Tensor = low.to(DEVICE)
        high_tensor: Tensor = high.to(DEVICE)
        
        output: Tensor = model(low_tensor)
        
        # Metrics need NumpPy array, (B, C, H, W) -> (H, W, C)
        output_np: np.ndarray = output.squeeze(0).cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        high_np: np.ndarray = high.squeeze(0).numpy().transpose(1, 2, 0)
        
        acc_psnr += peak_signal_noise_ratio(high_np, output_np, data_range=1.0)
        acc_ssim += ssim(high_np, output_np, data_range=1.0)

        # LPIPS needs a tensor in (B, C, H, W) format within range [-1, 1]
        output_scaled = output.clamp(0, 1) * 2.0 - 1.0
        high_scaled = high_tensor * 2.0 - 1.0
        
        # .item() to get float value and detach from graph
        acc_lpips += lpips_model(high_scaled, output_scaled).item()

    mean_psnr: float = acc_psnr / len(dataloader)
    mean_ssim: float = acc_ssim / len(dataloader)
    mean_lpips: float = acc_lpips / len(dataloader)

    return mean_psnr, mean_ssim, mean_lpips

def train(model: Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          criterion: Module,
          optimiser: Optimizer,
          logger: SummaryWriter) -> None:
    max_psnr = 0.0
    global_step = 0

    print("Starting training")
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        epoch_loss = 0
        
        for batch in train_dataloader:
            if batch is None:
                continue
            low_tensor, high_tensor = batch
            
            low_tensor = low_tensor.to(DEVICE)
            high_tensor = high_tensor.to(DEVICE)
            
            optimiser.zero_grad()
            output = model(low_tensor)
            loss = criterion(output, high_tensor)
            loss.backward()
            optimiser.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")
        
        # Validation & Checkpointing
        if (epoch + 1) % VAL_EVERY == 0:
            val_psnr, val_ssim = validate(model, val_dataloader)
            print(f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")
            
            # Log metrics
            log_metrics(logger=logger, mode="val", loss=avg_loss, psnr=val_psnr, ssim=val_ssim, lpips=0.0, step=global_step)
        
            if val_psnr > max_psnr:
                max_psnr = val_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'loss': loss,
                    'psnr': val_psnr
                }, LOG_DIR / f"edsr_x{SCALE}_psnr={val_psnr}.pth")
                print(f"Saved best model (PSNR: {val_psnr:.2f})")

logger = SummaryWriter( "logs", flush_secs=5)

model = Edsr(scale=SCALE, n_resblocks=N_RESBLOCKS, n_feats=N_FEATS).to(DEVICE)

criterion = nn.L1Loss() # paper uses L1
optimiser = Adam(model.parameters(), lr=LEARNING_RATE)

transform = Compose([
    ToTensor(),
])

train_val_dataset = Div2kDataset(low_path, high_path, transform, mode=Mode.TRAIN)
test_dataset = Div2kDataset(test_low_path, test_high_path, transform, mode=Mode.TEST)

train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [0.8, 0.2])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=cpu_count(), collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=cpu_count())
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=cpu_count())

print(f"Loaded {len(train_dataset)} training images")
print(f"Loaded {len(val_dataset)} validation images")
print(f"Loaded {len(test_dataset)} test images")

train(model, train_dataloader, val_dataloader, criterion, optimiser, logger)

print("Running final test...")
test_psnr, test_ssim, test_lpips = test(model, test_dataloader)
print(f"Test Results - PSNR: {test_psnr:.2f} dB, SSIM: {test_ssim:.4f}, LPIPS: {test_lpips:.4f}")