import unittest

from reinforce.data.dataloader import PolygonAssetDataloader

class TestDataloader(unittest.TestCase):
    def test_PolygonDataloader(self):
        with PolygonAssetDataloader(local_cache='tests/data/polygon_candle') as dataloader:
            assert dataloader.reset(2,10).shape == (2,10)
        #with PolygonAssetDataloader() as dataloader:
        #    assert dataloader.reset(2,10).shape == (2,10)