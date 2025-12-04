import random

from torch import Tensor
from torchvision.transforms.functional import crop

def get_random_patch(low_res: Tensor, high_res: Tensor, patch_size: int, scale: int = 8) -> tuple[Tensor, Tensor]:
    low_res_width, low_res_height = low_res.shape
    
    # Calculate random crop position
    tx: int = random.randrange(0, low_res_width - patch_size + 1)
    ty: int = random.randrange(0, low_res_height - patch_size + 1)
    
    low_patch: Tensor = crop(low_res, ty, tx, patch_size, patch_size)
    
    # Crop HR (coordinates scaled)
    tx_hr, ty_hr = tx * scale, ty * scale
    high_patch_size: int = patch_size * scale
    high_patch: Tensor = crop(high_res, ty_hr, tx_hr, high_patch_size, high_patch_size)
    
    return low_patch, high_patch