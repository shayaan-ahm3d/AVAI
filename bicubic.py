import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pathlib import Path
import csv
from skimage.metrics import peak_signal_noise_ratio
from lpips import LPIPS

from dataset import Div2kDataset, Mode
from inr_utils import ssim

SCALE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("outputs/bicubic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

low_path = Path("dataset/DIV2K_valid_LR_x8")
high_path = Path("dataset/DIV2K_valid_HR")

def main():
    dataset = Div2kDataset(low_path, high_path, transform=ToTensor(), mode=Mode.TEST)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    lpips_model = LPIPS(net='alex').to(DEVICE)
    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Create a CSV writer for each noise level
    csv_files = {}
    csv_writers = {}
    
    for std in noise_levels:
        f = open(OUTPUT_DIR / f"bicubic_noise_std={std}.csv", mode='w', newline='')
        w = csv.writer(f)
        w.writerow(["image_idx", "psnr", "ssim", "lpips"])
        csv_files[std] = f
        csv_writers[std] = w

    totals = {std: {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0} for std in noise_levels}

    print(f"Starting Bicubic Upscaling (x{SCALE}) with Noise on {len(dataset)} images...")

    try:
        for i, (low, high) in enumerate(dataloader):
            low = low.to(DEVICE)
            high = high.to(DEVICE)
            
            for std in noise_levels:
                # Add AWGN
                noise = torch.randn_like(low) * std
                low_noisy = (low + noise).clamp(0, 1)

                # Bicubic Upscaling
                upscaled = F.interpolate(low_noisy, size=high.shape[2:], mode='bicubic', align_corners=False)
                upscaled = upscaled.clamp(0, 1)

                # Calculate Metrics
                upscaled_np = upscaled.squeeze().cpu().numpy().transpose(1, 2, 0)
                high_np = high.squeeze().cpu().numpy().transpose(1, 2, 0)
                
                psnr_val = peak_signal_noise_ratio(high_np, upscaled_np, data_range=1.0)
                ssim_val = ssim(high_np, upscaled_np, data_range=1.0)
                
                upscaled_scaled = upscaled * 2.0 - 1.0
                high_scaled = high * 2.0 - 1.0
                lpips_val = lpips_model(high_scaled, upscaled_scaled).item()

                # Write to specific CSV
                csv_writers[std].writerow([i, psnr_val, ssim_val, lpips_val])
                
                totals[std]['psnr'] += psnr_val
                totals[std]['ssim'] += ssim_val
                totals[std]['lpips'] += lpips_val
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} images")

        print(f"\nAverage Results:")
        for std in noise_levels:
            avg_psnr = totals[std]['psnr'] / len(dataset)
            avg_ssim = totals[std]['ssim'] / len(dataset)
            avg_lpips = totals[std]['lpips'] / len(dataset)
            
            print(f"Noise {std}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
            csv_writers[std].writerow(["Average", avg_psnr, avg_ssim, avg_lpips])
            
    finally:
        # Close all files
        for f in csv_files.values():
            f.close()
        print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()