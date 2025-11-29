from dataset import Div2kDataset, Mode

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import matplotlib.pyplot as plt


# ### Step 1 Initialize SIREN Layers
# The sine layer is the basic building block of SIREN. SIREN is essentially a MLP (multi-layer perceptron) with sine activation. Note that SIREN is highly dependent on weights initialization to preserve the distribution of activations through the network. The authors propose the following initializations (check section 3.2 in the paper if you are interested):
# 
# * Weight is uniformly distributed such that
# $$
# w_i \sim U\left(-\frac{c}{\sqrt{n}}, \frac{c}{\sqrt{n}}\right)
# $$
# where c = 6.
#     
# * Initialize the first layer of the sine network with weights so that the sine function can be:
# $$
# \sin(\omega_0 \cdot W\mathbf{x} + b)
# $$
# where `omega_0` = 30.
#   
# To simplify, the key point here is that these initializations ensure the input to each sine activation is normally distributed with a standard deviation of 1, while the output of a SIREN is always arcsine distributed within the range of [-1, 1].
# 
# **Task 1:** Now complete the forward function below. Follow the equation provided below to complete the forward function. Try using [torch.sin()](https://pytorch.org/docs/stable/generated/torch.sin.html) as your sine activation function.
# $$
# \sin(\omega_0 \cdot W\mathbf{x} + b)
# $$
# 
# 


class SineLayer(nn.Module):
    """ Linear layer followed by the sine activation

    If `is_first == True`, then it represents the first layer of the network.
    In this case, omega_0 is a frequency factor, which simply multiplies the activations before the nonlinearity.
    Note that it influences the initialization scheme.

    If `is_first == False`, then the weights will be divided by omega_0 so as to keep the magnitude of activations constant,
    but boost gradients to the weight matrix.
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        # Initialize a linear layer with specified input and output features
        # 'bias' indicates whether to include a bias term
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    # initialize weights uniformly
    def init_weights(self):
        # diasble gradient calculation in initialization
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        # Task 1 TODO
        # 1. pass input through linear layer (self.linear layer performs the linear transformation on the input)
        x = self.linear(input)

        # 2. scale the output of the linear transformation by the frequency factor
        x = x * self.omega_0

        # 3. apply sine activation
        x = torch.sin(x)

        return x


class Siren(nn.Module):
    """ SIREN architecture """

    def __init__(self, in_features, out_features, hidden_features=256, hidden_layers=3, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        # add the first layer
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        # append hidden layers
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))


        if outermost_linear:
            # add a final Linear layer
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad(): # weights intialization
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            # otherwise, add a SineLayer
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net) # sequential wrapper of SineLayer and Linear

    def forward(self, coords):
        # coords represents the 2D pixel coordinates
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords


# ### Step 2 Data Preparation

# Let's generate a grid of coordinates over a 2D space and reshape the output into a flattened format.
def get_mgrid(sidelen1,sidelen2, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''

    if sidelen1 >= sidelen2:
      # use sidelen1 steps to generate the grid
      tensors = tuple(dim * [torch.linspace(-1, 1, steps = sidelen1)])
      mgrid = torch.stack(torch.meshgrid(*tensors), dim = -1)
      # crop it along one axis to fit sidelen2
      minor = int((sidelen1 - sidelen2)/2)
      mgrid = mgrid[:,minor:sidelen2 + minor]

    if sidelen1 < sidelen2:
      tensors = tuple(dim * [torch.linspace(-1, 1, steps = sidelen2)])
      mgrid = torch.stack(torch.meshgrid(*tensors), dim = -1)

      minor = int((sidelen2 - sidelen1)/2)
      mgrid = mgrid[minor:sidelen1 + minor,:]

    # flatten the gird
    mgrid = mgrid.reshape(-1, dim)

    return mgrid


# Make sure we convert the input to tensor and do normalization using transformations to the range [-1, 1], which corresponds to a mean of 0.5 and a standard deviation of 0.5.
def image_to_tensor(img):
    transform = Compose([
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


low = Image.open("0001x8.png").convert("RGB")
high = Image.open("0001.png").convert("RGB")

low_width, low_height = low.size
high_width, high_height = high.size

print(f"Low-res  size: {low_width}x{low_height}")
print(f"High-res size: {high_width}x{high_height}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create an ImageData instance
#input_image = 
#dataloader = DataLoader(input_image, batch_size=1, pin_memory=True, num_workers=0)

img_siren = Siren(in_features=2, out_features=3, hidden_features=256,
                  hidden_layers=3, outermost_linear=True).to(device)

lpips_model = lpips.LPIPS().to(device)
lpips_model.eval()

# prepare tensors for supervised super-resolution
low_np = np.array(low).astype(np.float32) / 255.0
high_np = np.array(high).astype(np.float32) / 255.0

low_tensor = torch.from_numpy(low_np).permute(2, 0, 1).contiguous().to(device)
high_tensor = torch.from_numpy(high_np).permute(2, 0, 1).contiguous().to(device)

low_target = low_tensor.unsqueeze(0)  # (1, 3, H_lr, W_lr)
high_target_flat = high_tensor.view(3, -1).permute(1, 0)  # (H_hr*W_hr, 3)

high_coords = get_mgrid(high_height, high_width).to(device).float()

# define differential operators that allow us to leverage autograd to compute gradients, the laplacian, etc.
def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = torch.zeros_like(y[..., :1])
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


# **Task 3:**
# 
# * Compute the gradient and the Laplacian of the output.
# 
# * Visualize the model output, gradient output and laplacian output.
# 
# *  What patterns do you notice in the gradient output compared to the model output? How do the edges detected by the Laplacian correlate with features in the model output?
# 
# Answer: able to explain where the model is sensitive to changes in input, and describe Laplacian (second derivative) can highlight areas of rapid intensity change - edges.

total_steps = 500 # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 10
channels = 3

optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

print(low_target.shape, high_target_flat.shape)

## Training Loop
for step in range(total_steps):
    model_output, autograd_coords = img_siren(high_coords)

    high_pred = model_output.view(high_height, high_width, channels).permute(2, 0, 1).unsqueeze(0)
    loss = F.mse_loss(model_output, high_target_flat)

    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f" % (step, loss))

        lap = laplace(model_output, autograd_coords)
        grad = gradient(model_output, autograd_coords)

        output_img = high_pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        grad_magnitude_img = grad.norm(dim=-1).detach().cpu().view(high_height, high_width).numpy()
        laplacian_img = lap.detach().cpu().view(high_height, high_width).numpy()

        plots = [output_img, grad_magnitude_img, laplacian_img]
        titles = ['Model Output', 'Gradient Magnitude', 'Laplacian']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, plot, title in zip(axes, plots, titles):
            cmap = 'gray' if plot.ndim == 2 else None
            ax.imshow(plot.squeeze(), cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
        plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()


## Super Resolution
with torch.no_grad():
    sr_output, _ = img_siren(high_coords)
    sr_image = sr_output.view(high_height, high_width, channels).clamp(0.0, 1.0).cpu().numpy()
    sr_tensor = sr_output.view(high_height, high_width, channels).permute(2, 0, 1).unsqueeze(0)

    bicubic_up = F.interpolate(low_target, size=(high_height, high_width), mode='bicubic', align_corners=False)
    hr_tensor = high_tensor.unsqueeze(0)

    def tensor_to_np(img_tensor):
        return img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0)

    bicubic_np = tensor_to_np(bicubic_up)
    sr_np = sr_image
    hr_img = high_np

    psnr_bicubic = peak_signal_noise_ratio(hr_img, bicubic_np, data_range=1.0)
    psnr_inr = peak_signal_noise_ratio(hr_img, sr_np, data_range=1.0)

    ssim_bicubic = structural_similarity(hr_img, bicubic_np, data_range=1.0, channel_axis=2)
    ssim_inr = structural_similarity(hr_img, sr_np, data_range=1.0, channel_axis=2)

    def to_lpips_range(img_tensor):
        return img_tensor * 2.0 - 1.0

    lpips_bicubic = lpips_model(to_lpips_range(bicubic_up), to_lpips_range(hr_tensor)).item()
    lpips_inr = lpips_model(to_lpips_range(sr_tensor), to_lpips_range(hr_tensor)).item()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Input Low-res')
plt.imshow(low_np)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Super-res (INR)')
plt.imshow(sr_image)
plt.axis('off')
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Bicubic ↑')
plt.imshow(bicubic_np)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('INR SR')
plt.imshow(sr_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Ground Truth')
plt.imshow(high_np)
plt.axis('off')
plt.show()

print('\nMetric comparison (higher PSNR/SSIM, lower LPIPS):')
print(f"PSNR  - Bicubic: {psnr_bicubic:.2f} dB | INR: {psnr_inr:.2f} dB")
print(f"SSIM  - Bicubic: {ssim_bicubic:.4f} | INR: {ssim_inr:.4f}")
print(f"LPIPS - Bicubic: {lpips_bicubic:.4f} | INR: {lpips_inr:.4f}")