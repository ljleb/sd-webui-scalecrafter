import scipy
import torch
from pathlib import Path


current_sampling_step = 0
dispersion_transform_discovery_paths = [Path(__file__).parent.parent / "dispersion_transforms"]
dispersion_transforms = {}


class DispersionTransform:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model_cache = None

    def load_model(self, device, dtype):
        if self.model_cache is None:
            dispersion_transform = scipy.io.loadmat(str(self.model_path))["R"]
            self.model_cache = torch.tensor(dispersion_transform, device=device, dtype=dtype)

        return self.model_cache.to(device=device, dtype=dtype)


def discover_dispersion_transforms():
    dispersion_transforms.clear()
    dispersion_transforms["None"] = None

    for discovery_path in dispersion_transform_discovery_paths:
        for file in discovery_path.iterdir():
            if file.is_file() and file.suffix == ".mat":
                dispersion_transforms[file.stem] = DispersionTransform(file)

    return dispersion_transforms
