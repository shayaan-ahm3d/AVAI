import numpy as np
import torch
from pathlib import Path

from PIL import Image as PILImage
from skimage.metrics import peak_signal_noise_ratio

from dataset import Div2kDataset, Mode
from models import get_net
from utils.common_utils import get_noise, np_to_pil, np_to_torch, pil_to_np, torch_to_np
from utils.denoising_utils import get_params, optimize, plot_image_grid
from utils.sr_utils import crop_image

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
dtype = torch.float32

PLOT = True

low_res_path = Path("dataset/DIV2K_train_LR_x8")
high_res_path = Path("dataset/DIV2K_train_HR")
dataset = Div2kDataset(low_res_path, high_res_path, Mode.TRAIN)

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

# Patch-specific settings
PATCH_SIZE = 256
PATCH_OVERLAP = 0
OUTPUT_DIR = Path("data/sr_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mse = torch.nn.MSELoss().to(device=device)


def build_sr_net():
    """Create a fresh DIP super-resolution network for a patch."""
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
    return net.to(device=device, dtype=dtype)


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


def run_dip_on_patch(high_patch_np, patch_idx, log_progress=False):
    net = build_sr_net()
    net_input = get_noise(
        input_depth,
        INPUT,
        (high_patch_np.shape[1], high_patch_np.shape[2])
    ).to(device=device, dtype=dtype).detach()
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    iteration = 0
    high_patch_torch = np_to_torch(high_patch_np).to(device=device, dtype=dtype)

    def closure():
        nonlocal net_input, out_avg, iteration

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1.0 - exp_weight)

        loss = mse(out, high_patch_torch)
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


def super_resolve_image(low_img, high_img, log_progress=False):
    low_np = pil_to_np(low_img)
    high_np = pil_to_np(high_img)

    height, width = high_np.shape[1:]
    patch_coords = list(generate_patch_coords(height, width, PATCH_SIZE, PATCH_OVERLAP))
    if log_progress:
        print(f'Training {len(patch_coords)} patches with {PATCH_OVERLAP}px overlap...')

    reconstruction = np.zeros_like(high_np)
    weight_map = np.zeros((1, height, width), dtype=np.float32)

    for idx, (top, left, bottom, right) in enumerate(patch_coords):
        high_patch_np = high_np[:, top:bottom, left:right]
        patch_out = run_dip_on_patch(high_patch_np, idx, log_progress=log_progress)

        ph = min(patch_out.shape[1], bottom - top)
        pw = min(patch_out.shape[2], right - left)

        reconstruction[:, top:top + ph, left:left + pw] += patch_out[:, :ph, :pw]
        weight_map[:, top:top + ph, left:left + pw] += 1.0

        if log_progress:
            print(f'Finished patch {idx + 1}/{len(patch_coords)}')

    final_output = reconstruction / np.clip(weight_map, 1e-8, None)

    lr_bicubic = low_img.resize(high_img.size, PILImage.Resampling.BICUBIC)
    lr_bicubic_np = pil_to_np(lr_bicubic)

    baseline_psnr = peak_signal_noise_ratio(high_np, lr_bicubic_np)
    final_psnr = peak_signal_noise_ratio(high_np, final_output)

    return final_output, low_np, high_np, lr_bicubic_np, baseline_psnr, final_psnr, len(patch_coords)

all_metrics = []

for sample_idx in range(len(dataset)):
    low_img_raw, high_img_raw = dataset[sample_idx]
    low_img = crop_image(low_img_raw)
    high_img = crop_image(high_img_raw)
    sample_name = dataset.low_paths[sample_idx].stem if hasattr(dataset, 'low_paths') else f"sample_{sample_idx:05d}"

    log_progress = PLOT and sample_idx == 0

    (final_output,
     low_np,
     high_np,
     lr_bicubic_np,
     baseline_psnr,
     final_psnr,
     num_patches) = super_resolve_image(low_img, high_img, log_progress=log_progress)

    output_img = np_to_pil(np.clip(final_output, 0, 1))
    output_path = OUTPUT_DIR / f"{sample_name}_sr.png"
    output_img.save(output_path)

    print(
        f"[{sample_idx + 1}/{len(dataset)}] {sample_name}: "
        f"baseline {baseline_psnr:.2f} dB -> DIP {final_psnr:.2f} dB ({num_patches} patches)."
    )
    print(f"Saved output to {output_path}")

    all_metrics.append(final_psnr)

    if log_progress:
        plot_image_grid([
            np.clip(lr_bicubic_np, 0, 1),
            np.clip(final_output, 0, 1),
            np.clip(high_np, 0, 1)
        ], factor=13, nrow=3)

if all_metrics:
    avg_psnr = sum(all_metrics) / len(all_metrics)
    print(f'Average DIP PSNR over {len(all_metrics)} images: {avg_psnr:.2f} dB')
