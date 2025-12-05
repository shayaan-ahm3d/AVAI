from dataset import Div2kDataset, Mode
from edsr_model import Edsr
from edsr_utils import get_random_patch, log_metrics, get_unique_log_dir
from inr_utils import ssim

from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module, functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer, Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from PIL import ImageDraw
import argparse

from skimage.metrics import peak_signal_noise_ratio
from lpips import LPIPS

SCALE = 8 # Super-resolution factor
N_RESBLOCKS = 16 # Number of residual blocks
N_FEATS = 64 # Number of filters
PATCH_SIZE = 24 # Low patch size (High patch size will be PATCH_SIZE * SCALE)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 700
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("outputs/EDSR")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_EVERY = 100

low_path = Path("dataset/DIV2K_train_LR_x8")
high_path = Path("dataset/DIV2K_train_HR")
test_low_path = Path("dataset/DIV2K_valid_LR_x8")
test_high_path = Path("dataset/DIV2K_valid_HR")

class PatchedDataset(Dataset):
    def __init__(self, dataset, patch_size, scale):
        self.dataset = dataset
        self.patch_size = patch_size
        self.scale = scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        low, high = self.dataset[idx]
        # crop here to minimise memory usage
        low_patch, high_patch = get_random_patch(low, high, self.patch_size, self.scale)
        return low_patch, high_patch

@torch.no_grad()
def validate(model: Module, dataloader: DataLoader) -> tuple[float, float]:
    """Validation loop, returns mean PSNR and SSIM"""
    model.eval()

    acc_psnr = 0.0
    acc_ssim = 0.0
    num_images = 0
    
    for i, (low, high) in enumerate(dataloader):
        low_tensor: Tensor = low.to(DEVICE)
        
        output: Tensor = model(low_tensor)
        
        # Iterate over batch
        for j in range(low_tensor.size(0)):
            output_np: np.ndarray = output[j].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            high_np: np.ndarray = high[j].numpy().transpose(1, 2, 0)
            
            acc_psnr += peak_signal_noise_ratio(high_np, output_np, data_range=1.0)
            acc_ssim += ssim(high_np, output_np, data_range=1.0)
            num_images += 1

    mean_psnr: float = acc_psnr / num_images if num_images > 0 else 0.0
    mean_ssim: float = acc_ssim / num_images if num_images > 0 else 0.0

    return mean_psnr, mean_ssim

@torch.no_grad()
def test(model: Module, dataloader: DataLoader) -> tuple[float, float, float]:
    """Testing loop, returns mean PSNR, SSIM & LPIPS"""
    model.eval()
    lpips_model: LPIPS = LPIPS().to(DEVICE).eval()

    acc_psnr = 0.0
    acc_ssim = 0.0
    acc_lpips = 0.0
    num_images = 0

    print(f"Saving test images to {OUTPUT_DIR}")

    for i, (low, high) in enumerate(dataloader):
        low_tensor: Tensor = low.to(DEVICE)
        high_tensor: Tensor = high.to(DEVICE)
        
        output: Tensor = model(low_tensor)
        
        # Bicubic Upsampling for comparison
        bicubic = F.interpolate(low_tensor, size=high_tensor.shape[2:], mode='bicubic', align_corners=False).clamp(0, 1)
        
        # LPIPS needs a tensor in (B, C, H, W) format within range [-1, 1]
        output_scaled = output.clamp(0, 1) * 2.0 - 1.0
        high_scaled = high_tensor * 2.0 - 1.0
        
        # lpips_model returns (B, 1, 1, 1)
        # We calculate batch LPIPS for the mean, but we also need individual for annotation
        batch_lpips = lpips_model(high_scaled, output_scaled)
        acc_lpips += batch_lpips.sum().item()

        # Metrics need NumpPy array, (B, C, H, W) -> (H, W, C)
        for j in range(low_tensor.size(0)):
            output_np: np.ndarray = output[j].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            high_np: np.ndarray = high[j].numpy().transpose(1, 2, 0)
            bicubic_np: np.ndarray = bicubic[j].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            
            # Model Metrics
            p_val = peak_signal_noise_ratio(high_np, output_np, data_range=1.0)
            s_val = ssim(high_np, output_np, data_range=1.0)
            l_val = batch_lpips[j].item()
            
            acc_psnr += p_val
            acc_ssim += s_val
            
            # Bicubic metrics
            b_p_val = peak_signal_noise_ratio(high_np, bicubic_np, data_range=1.0)
            b_s_val = ssim(high_np, bicubic_np, data_range=1.0)
            
            b_l_val = lpips_model(high_tensor[j:j+1] * 2 - 1, bicubic[j:j+1] * 2 - 1).item()
            
            # Create annotated images
            bicubic_labeled = add_labels(bicubic[j], b_p_val, b_s_val, b_l_val, "Bicubic")
            pred_labeled = add_labels(output[j], p_val, s_val, l_val, "EDSR")
            gt_labeled = add_labels(high[j], float('inf'), 1.0, 0.0, "Ground Truth")
            
            # Concatenate images side-by-side (dim=2 is width)
            combined = torch.cat((bicubic_labeled, pred_labeled, gt_labeled), dim=2)
            save_image(combined, OUTPUT_DIR / f"EDSR_test_{num_images + 1}.png")
            num_images += 1

    mean_psnr: float = acc_psnr / num_images if num_images > 0 else 0.0
    mean_ssim: float = acc_ssim / num_images if num_images > 0 else 0.0
    mean_lpips: float = acc_lpips / num_images if num_images > 0 else 0.0

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    log_dir: str = get_unique_log_dir(log_dir=Path("logs"), scale=SCALE, learning_rate=LEARNING_RATE, log_name="edsr")
    logger = SummaryWriter(log_dir=log_dir, flush_secs=5)

    model = Edsr(scale=SCALE, n_resblocks=N_RESBLOCKS, n_feats=N_FEATS).to(DEVICE)

    criterion = nn.L1Loss() # paper uses L1
    optimiser = Adam(model.parameters(), lr=LEARNING_RATE)
    
    if args.checkpoint:
        if Path(args.checkpoint).exists():
            checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            print(f"Checkpoint {args.checkpoint} not found!")

    transform = Compose([
        ToTensor(),
    ])

    train_val_dataset = Div2kDataset(low_path, high_path, transform, mode=Mode.TRAIN)
    test_dataset = Div2kDataset(test_low_path, test_high_path, transform, mode=Mode.TEST)

    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [0.8, 0.2])

    train_dataset_patched = PatchedDataset(train_dataset, PATCH_SIZE, SCALE)
    val_dataset_patched = PatchedDataset(val_dataset, PATCH_SIZE, SCALE)
    test_dataset_patched = PatchedDataset(test_dataset, PATCH_SIZE, SCALE)

    train_dataloader = DataLoader(train_dataset_patched, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=cpu_count())
    val_dataloader = DataLoader(val_dataset_patched, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=cpu_count())
    test_dataloader = DataLoader(test_dataset_patched, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=cpu_count())

    print(f"Loaded {len(train_dataset)} training images")
    print(f"Loaded {len(val_dataset)} validation images")
    print(f"Loaded {len(test_dataset)} test images")

    if args.train:
        train(model, train_dataloader, val_dataloader, criterion, optimiser, logger)

    print("Running inference...")
    test_psnr, test_ssim, test_lpips = test(model, test_dataloader)
    print(f"Test - PSNR: {test_psnr:.2f} dB | SSIM: {test_ssim:.4f} | LPIPS: {test_lpips:.4f}")

    logger.close()

def add_labels(img_tensor: Tensor, psnr: float, ssim: float, lpips: float, name: str) -> Tensor:
    img_pil = TF.to_pil_image(img_tensor.cpu().clamp(0, 1))
    draw = ImageDraw.Draw(img_pil)
    text = f"{name}\nPSNR: {psnr:.2f}\nSSIM: {ssim:.3f}\nLPIPS: {lpips:.3f}"
    draw.text((5, 5), text, fill=(255, 255, 255))
    return TF.to_tensor(img_pil).to(DEVICE)