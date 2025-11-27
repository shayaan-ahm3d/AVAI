from enum import Enum
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


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