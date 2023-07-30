import unittest

import numpy as np

from invest.env import SingleAssetEnv
from invest.data.dataloader import SingleAssetDataloader

class TestRollout(unittest.TestCase):
    def test_SingleAssetEnv(self):
        dataloader = SingleAssetDataloader(
            prices = np.random.random(1024),
            window=128
        )
        env = SingleAssetEnv(dataloader)
        env.rollout(20)

