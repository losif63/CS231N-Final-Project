"""Model layer: backbones, factory, DifficultyModel wrapper.

We deliberately do NOT eagerly import individual backbone classes here — they
depend on env-specific deps (pytorchvideo for the CNN backbones; transformers
for the Qwen-VL backbone) and not every env has every dep. Use
`build_backbone(name)` from `factory`; it lazy-imports just the class you
need.
"""

from .difficulty_model import DifficultyModel
from .factory import BACKBONES, build_backbone

__all__ = [
    "DifficultyModel",
    "BACKBONES",
    "build_backbone",
]
