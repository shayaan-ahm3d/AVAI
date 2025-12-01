from dataset import ImagePreparation_color
from inr_utils import map_to_01, laplace, divergence, gradient
from siren import Siren

from multiprocessing import cpu_count

import torch
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_siren = Siren(in_features=2, out_features=3, hidden_features=256,
                  hidden_layers=3, outermost_linear=True).to(device)

lpips_model = lpips.LPIPS().to(device).eval()

total_steps = 5000 # Since the whole image is our dataset, this just means 5000 gradient descent steps.
steps_til_summary = total_steps // 10
channels = 3
optimizer = torch.optim.Adam(img_siren.parameters() ,lr=1e-5)
low_size = (175, 255)
image = ImagePreparation_color('0001x8.png', low_size)
print(f"{image.height=}")
print(f"{image.width=}")
dataloader = DataLoader(image, batch_size=1, pin_memory=True, num_workers=cpu_count())
mse = MSELoss()
torch.cuda.empty_cache()

model_input_coords, ground_truth_pixel_values = next(iter(dataloader))
model_input_coords, ground_truth_pixel_values = model_input_coords.cuda(), ground_truth_pixel_values.cuda()

for step in range(total_steps):
    model_output_pixel_values, coords = img_siren(model_input_coords)

    out = model_output_pixel_values.reshape([1, low_size[0], low_size[1], 3]).permute([0,3,1,2])
    gt = ground_truth_pixel_values.reshape([1, low_size[0], low_size[1], 3]).permute([0,3,1,2])

    loss = mse(out, gt)
    
    if not step % steps_til_summary:
        print("Step %d, Total Loss %0.9f" % (step, loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

high_size = (1404, 2040)
with torch.no_grad():
    high = ImagePreparation_color('0001.png', high_size)
    print(f"{high.height=}")
    print(f"{high.width=}")
    dataloader = DataLoader(high, batch_size=1, pin_memory=True, num_workers=cpu_count())

    model_input_coords, ground_truth_pixel_values = next(iter(dataloader))
    model_input_coords, ground_truth_pixel_values = model_input_coords.to(device), ground_truth_pixel_values.to(device)

    model_output_pixel_values, coords = img_siren(model_input_coords)
    loss = mse(model_output_pixel_values, ground_truth_pixel_values)

    fig, axes = plt.subplots(1, 2, figsize=(16,8))
    axes[0].imshow(map_to_01(ground_truth_pixel_values).cpu().view(high_size[0], high_size[1], 3).detach().numpy())
    axes[0].set_title('Ground Truth', fontsize=20)
    axes[1].imshow(map_to_01(model_output_pixel_values).cpu().view(high_size[0], high_size[1], 3).detach().numpy())
    axes[1].set_title('Model Output', fontsize=20)
    
    plt.savefig("comparison")

    print("Total loss: %0.6f" % (loss))