"""Dataset for Geometry Dash difficulty prediction.

Expects pre-extracted frames (run extract_frames.py first).

Frame directory structure:
    frames_dir/{n}stars/{video_stem}/0001.jpg
                                     0002.jpg
                                     ...

Frames are stored at 256x256. Pass image_size < 256 to resize down at load time.
The difficulty label (0-indexed, 0..9) is derived from the {n}stars folder name.
"""

import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


AUGMENT_TRANSFORM = T.Compose([
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
])


class GeometryDashDataset(Dataset):
    def __init__(
        self,
        frames_dir,
        image_size: int = 64,
        max_frames: int = 20 * 3,
        augment: bool = False,
    ):
        """
        Args:
            frames_dir: Root directory containing {n}stars/{video_stem}/ subdirs of JPEGs.
            image_size: Resize frames to (image_size, image_size) at load time.
                        Set to 256 to use the stored resolution without resizing.
            max_frames: Use at most this many frames per video (taken from the start).
            augment: Apply random flip + color jitter during loading (use for training set).
        """
        self.image_size = image_size
        self.max_frames = max_frames
        self.augment = augment

        self.samples: list[tuple[Path, int]] = []  # (frame_dir, label 0-indexed)

        frames_dir = Path(frames_dir)
        for stars_dir in sorted(frames_dir.iterdir()):
            if not stars_dir.is_dir():
                continue
            name = stars_dir.name
            if not name.endswith("stars"):
                continue
            try:
                stars = int(name[: -len("stars")])
            except ValueError:
                continue
            if not (1 <= stars <= 10):
                continue

            for video_dir in sorted(stars_dir.iterdir()):
                if video_dir.is_dir() and any(video_dir.glob("*.jpg")):
                    self.samples.append((video_dir, stars - 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_dir, label = self.samples[idx]

        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if self.max_frames:
            frame_paths = frame_paths[: self.max_frames]

        frames = []
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.image_size != img.shape[0]:  # stored as square, so h == w
                img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            if self.augment:
                tensor = AUGMENT_TRANSFORM(tensor)
            tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            frames.append(tensor)

        if not frames:
            frames = [torch.zeros(3, self.image_size, self.image_size)]

        return torch.stack(frames), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Custom collate: keeps variable-length frame tensors as a list."""
    frames_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return frames_list, labels
