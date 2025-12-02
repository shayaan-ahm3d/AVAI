from dataset import Div2kDataset, Mode
from inr_utils import convert_pixel_value_range, laplace, divergence, gradient
from siren import Siren

from pathlib import Path
from multiprocessing import cpu_count

import torch
from torch.nn import Module, MSELoss
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from lpips import LPIPS
import matplotlib.pyplot as plt
from PIL.Image import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model: Module, low_res_image: torch.Tensor, steps: int) -> None:
    input_coords, ground_truth_pixel_values = Div2kDataset.get_coordinate_to_pixel_value_mapping(low_res_image)
    input_coords = input_coords.to(DEVICE)
    ground_truth_pixel_values = ground_truth_pixel_values.to(DEVICE)

    for step in range(steps):
        model_output_pixel_values, _ = model(input_coords)

        out = model_output_pixel_values.reshape([low_res_image.shape[1], low_res_image.shape[2], 3]).permute([2, 0, 1])
        gt = ground_truth_pixel_values.reshape([low_res_image.shape[1], low_res_image.shape[2], 3]).permute([2, 0, 1])

        loss = mse(out, gt)
        
        if not step % steps_til_summary:
            print("Step %d: Total Loss %0.9f" % (step, loss))

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

def evaluate_model(model: Module, high_res_image: torch.Tensor, index: int):
    input_coords, ground_truth_pixel_values = Div2kDataset.get_coordinate_to_pixel_value_mapping(high_res_image)
    input_coords.to(DEVICE)
    ground_truth_pixel_values.to(DEVICE)

    with torch.no_grad():
        model_output_pixel_values, _ = model(input_coords)
        loss = mse(model_output_pixel_values, ground_truth_pixel_values)

        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        axes[0].imshow(convert_pixel_value_range(ground_truth_pixel_values).cpu().view(high_res_image.shape[1], high_res_image.shape[2], 3).detach().numpy())
        axes[0].set_title("Ground Truth", fontsize=20)
        axes[1].imshow(convert_pixel_value_range(model_output_pixel_values).cpu().view(high_res_image.shape[1], high_res_image.shape[2], 3).detach().numpy())
        axes[1].set_title("Model Output", fontsize=20)
        
        plt.savefig(f"outputs/INR/{index}-SISR")

        print("Total loss: %0.6f" % (loss))

# SIREN super resolution model: (x, y) -> (R, G, B)
super_resolve = Siren(in_features=2, out_features=3, hidden_features=256,
                  hidden_layers=3, outermost_linear=True).to(DEVICE)
mse = MSELoss()
optimiser = torch.optim.Adam(super_resolve.parameters() , lr=1e-5)
# LPIPS model to calculate metrics
lpips_model = LPIPS().to(DEVICE).eval()
# Since the whole image is our dataset, this is just the number of gradient descent steps.
total_steps = 10_000
steps_til_summary = total_steps // 10

low_res_path = Path("dataset/DIV2K_train_LR_x8")
high_res_path = Path("dataset/DIV2K_train_HR")
transform = Compose([
    ToTensor(),
    Normalize(mean=torch.Tensor([0.5, 0.5, 0.5]), std=torch.Tensor([0.5, 0.5, 0.5]))
    ])
dataset = Div2kDataset(low_root=low_res_path, high_root=high_res_path, transform=transform, mode=Mode.TEST)

for i, (low, high) in enumerate(iter(dataset)):
    low = low.to(DEVICE)
    high = high.to(DEVICE)

    train_model(super_resolve, low, total_steps)
    evaluate_model(super_resolve, high, i)