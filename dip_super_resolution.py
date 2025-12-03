from dataset import Div2kDataset, Mode
from utils.common_utils import get_noise, np_to_torch, pil_to_np, torch_to_np, np_to_pil
from utils.denoising_utils import get_params, optimize
from utils.sr_utils import get_baselines
from models import get_net

import torch
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PLOT = False

low_res_path = Path("dataset/DIV2K_valid_LR_x8")
high_res_path = Path("dataset/DIV2K_valid_HR")
dataset = Div2kDataset(low_res_path, high_res_path, transform=lambda x: x, mode=Mode.TRAIN)

# Optimization and network hyperparameters
pad = 'reflection'
OPT_OVER = 'net'
INPUT = 'noise'
reg_noise_std = 1.0 / 30.0
LR = 0.01
OPTIMIZER = 'adam'
show_every = 100
exp_weight = 0.99
num_iter = 500
input_depth = 3
ENABLE_NOISE = True
NOISE_STD = 0.01

# Patch-specific settings
PATCH_SIZE = 256
PATCH_OVERLAP = 0

OUTPUT_DIR = Path("outputs/DIP")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

lpips_model = lpips.LPIPS().to(device)
lpips_model.eval()

mse = torch.nn.MSELoss().to(device=device)


def build_sr_net():
    net = get_net(
        input_depth,
        'skip',
        pad,
        skip_n33d=128,
        skip_n33u=128,
        skip_n11=4,
        num_scales=5,
        upsample_mode='bilinear',
    )
    return net.to(device=device)


def _sliding_window_indices(length, window, stride):
    if length <= window:
        return [0]

    positions = list(range(0, length - window + 1, stride))
    if positions[-1] != length - window:
        positions.append(length - window)
    return positions


def generate_patch_coords(height, width, patch_size, overlap):
    """Yield top/left/bottom/right tuples that tile the HR image."""
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    patch_h = min(patch_size[0], height)
    patch_w = min(patch_size[1], width)
    stride_h = max(patch_h - overlap, 1)
    stride_w = max(patch_w - overlap, 1)

    for top in _sliding_window_indices(height, patch_h, stride_h):
        for left in _sliding_window_indices(width, patch_w, stride_w):
            bottom = min(top + patch_h, height)
            right = min(left + patch_w, width)
            yield top, left, bottom, right


def run_dip_on_patch(hr_shape, low_patch_np, patch_idx, log_progress=False):
    net = build_sr_net()
    net_input = get_noise(
        input_depth,
        INPUT,
        hr_shape
    ).to(device=device).detach()
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    iteration = 0
    
    # Create low-res target for unsupervised loss
    low_patch_torch = np_to_torch(low_patch_np).to(device=device)

    def closure():
        nonlocal net_input, out_avg, iteration

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1.0 - exp_weight)

        # Downsample the network output to match the low-res target
        out_downsampled = torch.nn.functional.interpolate(out, size=low_patch_torch.shape[-2:], mode='bicubic', align_corners=False)
        
        loss = mse(out_downsampled, low_patch_torch)
        loss.backward()

        if log_progress and patch_idx == 0 and iteration % show_every == 0:
            print(f'Patch {patch_idx} | iter {iteration} | loss {loss.item():.6f}')

        iteration += 1
        return loss

    params = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, params, closure, LR, num_iter)

    with torch.no_grad():
        final_patch = out_avg if out_avg is not None else net(net_input)

    return torch_to_np(final_patch)


def ssim(chw_a, chw_b):
    a = np.clip(chw_a, 0, 1).transpose(1, 2, 0)
    b = np.clip(chw_b, 0, 1).transpose(1, 2, 0)
    return structural_similarity(a, b, data_range=1.0, channel_axis=2)


def _lpips(chw_a, chw_b):
    def _to_tensor(arr):
        tens = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        tens = tens.unsqueeze(0)
        return tens * 2.0 - 1.0  # scale [0,1] -> [-1,1]

    with torch.no_grad():
        score = lpips_model(_to_tensor(chw_a), _to_tensor(chw_b))
    return float(score.squeeze().cpu().numpy())


def crop_lr_hr_pair(low_img_raw, high_img_raw, factor=8, divisible_by=32):
    w_lr, h_lr = low_img_raw.size
    w_lr_new = w_lr - w_lr % divisible_by
    h_lr_new = h_lr - h_lr % divisible_by
    
    left_lr = (w_lr - w_lr_new) // 2
    top_lr = (h_lr - h_lr_new) // 2
    
    low_img = low_img_raw.crop((left_lr, top_lr, left_lr + w_lr_new, top_lr + h_lr_new))
    
    left_hr = left_lr * factor
    top_hr = top_lr * factor
    w_hr_new = w_lr_new * factor
    h_hr_new = h_lr_new * factor
    
    high_img = high_img_raw.crop((left_hr, top_hr, left_hr + w_hr_new, top_hr + h_hr_new))
    
    return low_img, high_img


def super_resolve_image(low_img, high_img, log_progress=False):
    low_np = pil_to_np(low_img)
    if ENABLE_NOISE:
        low_np = add_noise(low_np, std=NOISE_STD)
    
    low_img_for_baselines = np_to_pil(low_np)
    high_np = pil_to_np(high_img)

    height, width = high_np.shape[1:]
    patch_coords = list(generate_patch_coords(height, width, PATCH_SIZE, PATCH_OVERLAP))
    if log_progress:
        print(f'Training {len(patch_coords)} patches with {PATCH_OVERLAP}px overlap...')

    reconstruction = np.zeros_like(high_np)
    weight_map = np.zeros((1, height, width), dtype=np.float32)

    for idx, (top, left, bottom, right) in enumerate(patch_coords):
        # high_patch_np = high_np[:, top:bottom, left:right]
        
        top_lr, left_lr = top // 8, left // 8
        bottom_lr, right_lr = bottom // 8, right // 8
        low_patch_np = low_np[:, top_lr:bottom_lr, left_lr:right_lr]
        
        hr_shape = (bottom - top, right - left)

        patch_out = run_dip_on_patch(hr_shape, low_patch_np, idx, log_progress=log_progress)

        ph = min(patch_out.shape[1], bottom - top)
        pw = min(patch_out.shape[2], right - left)

        reconstruction[:, top:top + ph, left:left + pw] += patch_out[:, :ph, :pw]
        weight_map[:, top:top + ph, left:left + pw] += 1.0

        if log_progress:
            print(f'Finished patch {idx + 1}/{len(patch_coords)}')

    final_output = reconstruction / np.clip(weight_map, 1e-8, None)

    lr_bicubic_np, lr_bic_sharp_np, lr_nearest_np = get_baselines(low_img_for_baselines, high_img)

    baseline_psnr = peak_signal_noise_ratio(high_np, lr_bicubic_np)
    final_psnr = peak_signal_noise_ratio(high_np, final_output)
    baseline_ssim = ssim(high_np, lr_bicubic_np)
    final_ssim = ssim(high_np, final_output)
    baseline_lpips = _lpips(high_np, lr_bicubic_np)
    final_lpips = _lpips(high_np, final_output)

    return (
        final_output,
        low_np,
        high_np,
        lr_bicubic_np,
        baseline_psnr,
        final_psnr,
        baseline_ssim,
        final_ssim,
        baseline_lpips,
        final_lpips,
        len(patch_coords),
    )

all_psnr = []
all_ssim = []
all_lpips = []
metric_filename = f"dip_noisy_std={NOISE_STD}.csv" if ENABLE_NOISE else "dip.csv"
metrics_log_path = OUTPUT_DIR / metric_filename
metrics_log_path.write_text(
    "sample,baseline_psnr,dip_psnr,baseline_ssim,dip_ssim,baseline_lpips,dip_lpips,num_patches\n"
)

def add_noise(image: np.ndarray, std: float) -> np.ndarray:
    awgn = np.random.normal(loc=0.0, scale=std, size=image.shape)
    return np.clip(image + awgn, 0.0, 1.0)

for i in range(len(dataset)):
    low_img_raw, high_img_raw = dataset[i]
    
    low_img, high_img = crop_lr_hr_pair(low_img_raw, high_img_raw)

    sample_name = dataset.low_paths[i].stem if hasattr(dataset, 'low_paths') else f"sample_{i:05d}"

    log_progress = PLOT and i == 0

    (
        final_output,
        low_np,
        high_np,
        lr_bicubic_np,
        baseline_psnr,
        final_psnr,
        baseline_ssim,
        final_ssim,
        baseline_lpips,
        final_lpips,
        num_patches,
    ) = super_resolve_image(low_img, high_img, log_progress=log_progress)

    print(
        f"[{i + 1}/{len(dataset)}] {sample_name}: "
        f"baseline {baseline_psnr:.2f} dB / {baseline_ssim:.3f} SSIM / {baseline_lpips:.3f} LPIPS -> "
        f"DIP {final_psnr:.2f} dB / {final_ssim:.3f} SSIM / {final_lpips:.3f} LPIPS ({num_patches} patches)."
    )

    if i % 10 == 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Bicubic
        axes[0].imshow(np.clip(lr_bicubic_np.transpose(1, 2, 0), 0, 1))
        axes[0].set_title(f"Bicubic\nPSNR: {baseline_psnr:.2f} dB")
        axes[0].axis('off')
        
        # DIP
        axes[1].imshow(np.clip(final_output.transpose(1, 2, 0), 0, 1))
        axes[1].set_title(f"DIP\nPSNR: {final_psnr:.2f} dB")
        axes[1].axis('off')
        
        # Ground Truth
        axes[2].imshow(np.clip(high_np.transpose(1, 2, 0), 0, 1))
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')
        
        comparative_figure_filename = f"{sample_name}_comparison.png" if not ENABLE_NOISE else f"{sample_name}_noise_std={NOISE_STD}_comparison.png"
        comparative_figure_path = OUTPUT_DIR / comparative_figure_filename
        plt.tight_layout()
        plt.savefig(comparative_figure_path)
        plt.close(fig)
        print(f"Saved output to {comparative_figure_path}")

    with metrics_log_path.open("a") as log_file:
        log_file.write(
            f"{sample_name},{baseline_psnr:.4f},{final_psnr:.4f},{baseline_ssim:.4f},{final_ssim:.4f},{baseline_lpips:.4f},{final_lpips:.4f},{num_patches}\n"
        )

    all_psnr.append(final_psnr)
    all_ssim.append(final_ssim)
    all_lpips.append(final_lpips)

if all_psnr:
    avg_psnr = sum(all_psnr) / len(all_psnr)
    avg_ssim = sum(all_ssim) / len(all_ssim)
    avg_lpips = sum(all_lpips) / len(all_lpips)
    summary = (
        f'Average DIP metrics over {len(all_psnr)} images: '
        f'{avg_psnr:.2f} dB PSNR, {avg_ssim:.3f} SSIM, {avg_lpips:.3f} LPIPS'
    )
    print(summary)
    with metrics_log_path.open("a") as log_file:
        log_file.write(summary + "\n")
