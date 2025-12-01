from enum import Enum
from pathlib import Path

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class Mode(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


class Div2kDataset(Dataset):
    def __init__(
        self,
        low_root: Path,
        high_root: Path,
        mode: Mode = Mode.TRAIN,
        transform = None,
    ) -> None:
        self.low_root = Path(low_root)
        self.high_root = Path(high_root)
        self.mode = mode
        self.transform = transform

        self.low_paths = self._collect_files(self.low_root)
        self.high_paths = self._collect_files(self.high_root)

        if len(self.low_paths) != len(self.high_paths):
            raise ValueError(
                f"Mismatch: {len(self.low_paths)} LR vs {len(self.high_paths)} HR images."
            )

    def __len__(self) -> int:
        return len(self.low_paths)

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
        low_img = Image.open(self.low_paths[idx]).convert("RGB")
        high_img = Image.open(self.high_paths[idx]).convert("RGB")

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img

    def _collect_files(self, root: Path) -> list[Path]:
        return sorted([
            p for p in Path(root).rglob("*")
            if p.is_file()
        ])
    

class Div2kInr(Dataset):
    def __init__(self, path: Path) -> None:
            super().__init__()
            self.image = Image.open(path)
            transform = Compose([
                Resize((self.image.height, self.image.width)),
                ToTensor(),
                Normalize(mean=torch.Tensor([0.5, 0.5, 0.5]), std=torch.Tensor([0.5, 0.5, 0.5]))
            ])

            self.coords: Tensor = get_mgrid(self.image.height, self.image.width)
            self.pixels: Tensor = transform(self.image).permute(1, 2, 0).contiguous().view(self.image.height * self.image.width, 3)

    def __len__(self) -> int:
      return 1

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
      if idx > 0: raise IndexError("Single image super resolution method, therefore only 1 image!")
      return self.coords, self.pixels
  
# ### Step 2 Data Preparation

# Let's generate a grid of coordinates over a 2D space and reshape the output into a flattened format.
def get_mgrid(side_len1, side_len2, dim=2) -> Tensor:
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''

    if side_len1 >= side_len2:
      # use sidelen1 steps to generate the grid
      tensors = tuple(dim * [torch.linspace(-1, 1, steps = side_len1)])
      mgrid = torch.stack(torch.meshgrid(*tensors), dim = -1)
      # crop it along one axis to fit sidelen2
      minor = int((side_len1 - side_len2)/2)
      mgrid = mgrid[:,minor:side_len2 + minor]

    if side_len1 < side_len2:
      tensors = tuple(dim * [torch.linspace(-1, 1, steps = side_len2)])
      mgrid = torch.stack(torch.meshgrid(*tensors), dim = -1)

      minor = int((side_len2 - side_len1)/2)
      mgrid = mgrid[minor:side_len1 + minor,:]

    # flatten the gird
    mgrid = mgrid.reshape(-1, dim)

    return mgrid