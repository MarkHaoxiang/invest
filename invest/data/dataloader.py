from abc import ABC
from typing import Sequence, Tuple

class AssetDataloader(ABC):
    def __init__(self, window: int = 50):
        self._window = window

    def reset(self) -> Tuple[float, Sequence[float]]:
        raise NotImplementedError

    def step(self) -> Tuple[float, Sequence[float]]:
        """
        Returns
            Price of asset
        """
        raise NotImplementedError

    @property
    def done(self) -> bool:
        raise NotImplementedError
    
    @property
    def window(self) -> float:
        return self._window

class SingleAssetDataloader(AssetDataloader):
    def __init__(self, prices: Sequence[int], window: int = 50):
        super().__init__(window=window)
        assert window < len(prices), "Window size is larger than data amount"
        self._prices = prices
        self._counter = window

    def reset(self) -> Tuple[float, Sequence[float]]:
        self._counter = self._window
        return self._prices[self._counter], self._prices[self._counter-self._window:self._counter]

    def step(self) -> Tuple[float, Sequence[float]]:
        self._counter += 1
        return self._prices[self._counter], self._prices[self._counter-self._window:self._counter]

    def __len__(self):
        return len(self._prices)
    
    @property    
    def done(self) -> bool:
        return self._counter >= len(self) - 1