from dataset import Div2kInr
from inr_utils import convert_pixel_value_range, laplace, divergence, gradient
from siren import Siren

from pathlib import Path
from multiprocessing import cpu_count

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

super_resolve = Siren(in_features=2, out_features=3, hidden_features=256,
                  hidden_layers=3, outermost_linear=True).to(device)

lpips_model = lpips.LPIPS().to(device).eval()

total_steps = 10_000 # Since the whole image is our dataset, this just means 5000 gradient descent steps.
steps_til_summary = total_steps // 10

channels = 3
optimiser = torch.optim.Adam(super_resolve.parameters() , lr=1e-5)
low_res_image = Div2kInr(Path("0001x8.png"))
dataloader = DataLoader(low_res_image, batch_size=1, pin_memory=True, num_workers=cpu_count())
mse = MSELoss()

model_input_coords, ground_truth_pixel_values = next(iter(dataloader))
model_input_coords, ground_truth_pixel_values = model_input_coords.to(device), ground_truth_pixel_values.to(device)

for step in range(total_steps):
    model_output_pixel_values, coords = super_resolve(model_input_coords)

    out = model_output_pixel_values.reshape([1, low_res_image.image.height, low_res_image.image.width, 3]).permute([0, 3, 1, 2])
    gt = ground_truth_pixel_values.reshape([1, low_res_image.image.height, low_res_image.image.width, 3]).permute([0, 3, 1, 2])

    loss = mse(out, gt)
    
    if not step % steps_til_summary:
        print("Step %d, Total Loss %0.9f" % (step, loss))

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

with torch.no_grad():
    high_res_image = Div2kInr(Path("0001.png"))
    dataloader = DataLoader(high_res_image, batch_size=1, pin_memory=True, num_workers=cpu_count())

    model_input_coords, ground_truth_pixel_values = next(iter(dataloader))
    model_input_coords, ground_truth_pixel_values = model_input_coords.to(device), ground_truth_pixel_values.to(device)

    model_output_pixel_values, coords = super_resolve(model_input_coords)
    loss = mse(model_output_pixel_values, ground_truth_pixel_values)

    fig, axes = plt.subplots(1, 2, figsize=(16,8))
    axes[0].imshow(convert_pixel_value_range(ground_truth_pixel_values).cpu().view(high_res_image.image.height, high_res_image.image.width, 3).detach().numpy())
    axes[0].set_title("Ground Truth", fontsize=20)
    axes[1].imshow(convert_pixel_value_range(model_output_pixel_values).cpu().view(high_res_image.image.height, high_res_image.image.width, 3).detach().numpy())
    axes[1].set_title("Model Output", fontsize=20)
    
    plt.savefig("comparison")

    print("Total loss: %0.6f" % (loss))