"""Dataset for Geometry Dash difficulty prediction.

Videos live under: videos_dir/{n}stars/{level_id}_{k}.{ext}
The difficulty label (0-indexed, 0..9) is derived from the {n}stars folder name.
"""

import cv2
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


class GeometryDashDataset(Dataset):
    def __init__(
        self,
        videos_dir,
        frame_interval_sec: float = 4.0,
        crop_fraction: float = 1.0,
        image_size: int = 224,
    ):
        """
        Args:
            videos_dir: Root directory containing {n}stars subdirectories of videos.
            frame_interval_sec: Sample one frame every this many seconds.
            crop_fraction: Fraction of min(H,W) to use for the center square crop (0 < f <= 1).
            image_size: Resize cropped frame to (image_size, image_size) for ResNet.
        """
        self.frame_interval_sec = frame_interval_sec
        self.crop_fraction = crop_fraction
        self.image_size = image_size

        self.samples: list[tuple[str, int]] = []  # (video_path, label 0-indexed)

        videos_dir = Path(videos_dir)
        for stars_dir in sorted(videos_dir.iterdir()):
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

            for video_file in sorted(stars_dir.iterdir()):
                if video_file.suffix.lower() in VIDEO_EXTENSIONS:
                    self.samples.append((str(video_file), stars - 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_path, label = self.samples[idx]
        frames = self._extract_frames(video_path)
        return frames, torch.tensor(label, dtype=torch.long)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _center_crop_square(self, frame):
        """Return a center-cropped square from a HxWx3 numpy frame."""
        h, w = frame.shape[:2]
        size = int(min(h, w) * self.crop_fraction)
        cy, cx = h // 2, w // 2
        half = size // 2
        return frame[cy - half : cy + half, cx - half : cx + half]

    def _extract_frames(self, video_path: str) -> torch.Tensor:
        """Return (T, C, H, W) float tensor of normalized frames."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frame_step = max(1, int(round(fps * self.frame_interval_sec)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self._center_crop_square(frame)
            frame = cv2.resize(
                frame, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )

            tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            frames.append(tensor)

            frame_idx += frame_step

        cap.release()

        if not frames:
            frames = [torch.zeros(3, self.image_size, self.image_size)]

        return torch.stack(frames)  # (T, C, H, W)


def collate_fn(batch):
    """Custom collate: keeps variable-length frame tensors as a list."""
    frames_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return frames_list, labels
