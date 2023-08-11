from abc import ABC
import torch
from torch import Tensor

class AssetDataloader(ABC):
    def reset(self, batch_size: int, steps: int) -> Tensor:
        raise NotImplementedError

    def seed(self, seed):
        pass

class MockAssetDataloader(ABC):
    """ Testing
    """
    def reset(self, batch_size: int, steps: int) -> Tensor:
        return (1+torch.arange(batch_size * steps)).reshape((batch_size, steps))

    def seed(self, seed):
        torch.manual_seed(seed)
