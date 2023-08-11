from typing import Optional, Sequence, Tuple, Union, List
import os
import random

import yaml
import pandas as pd
from tiingo import TiingoClient

from invest.env import AssetDataloader

class TingoAssetDataloader(AssetDataloader):
    def __init__(self,
                 tickers: Optional[Union[str, List[str]]] = None,
                 filter = None,
                 frequency = 'daily',
                 window: int = 28):
        super().__init__(window)

        with open('data/setup.yaml', 'r') as f:
            settings = yaml.safe_load(f)
            api_key = settings['tiingo']['key']

        self.client = TiingoClient({
            'session': True,
            'api_key': api_key
        })

        path = "data/tiingo/supported_tickers.csv"
        assert os.path.isfile(path)
        df = pd.read_csv(path)
        if filter != None:
            df = filter(df)
        if isinstance(tickers, str):
            tickers = [tickers]
        all_tickers = df['tickers'].unique()
        if tickers is None:    
            tickers = all_tickers
        else:
            assert all((t in all_tickers for t in tickers)), "Cannot find ticker"

        self._tickers = tickers

    def reset(self) -> Tuple[float, Sequence[float]]:
        self.ticker = random.choice(self.tickers)
 