import os
import logging
from typing import Optional, Sequence
from abc import ABC
from contextlib import AbstractContextManager
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import polygon
from polygon.enums import TickerType
import tiingo

# TODO: The failcounter fallback should really be replaced with a proper database system

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

class PolygonAssetDataloader(AssetDataloader, AbstractContextManager):
    """ polygon.io

    Params:
        local_cache: use CSV files downloaded from the API stored locally
    """
    def __init__(self,
                 api_key = None,
                 local_cache: Optional[str] = None,
                 use_only_cached: bool = True,
                 tickers: Optional[Sequence[str]] = None,
                 price_key = 'o',
                 seed: Optional[int] = None):
        self.api_key = api_key
        if api_key is None:
            self.api_key = os.getenv("POLYGON_API_KEY")
        self.seed(seed)

        assert price_key in ['o', 'vw', 'c', 'h', 'l']
        self._price_key = price_key

        self._local_cache = local_cache
        self._use_only_cached = use_only_cached   
        if not local_cache is None:
            assert os.path.isdir(local_cache), "Cache directory given is invalid."
        self._stock_client = polygon.StocksClient(self.api_key)
        reference_client = polygon.ReferenceClient(self.api_key)

        if tickers is None:
            if self._use_only_cached and not self._local_cache is None:
                tickers = os.listdir(self._local_cache)
                tickers = filter(lambda x : x.endswith(".csv"), tickers)
                tickers = [t[:-4] for t in tickers]
                self._all_tickers = tickers
            else:
                logging.warn("Recommend using cached data")
                all_stock_tickers = reference_client.get_tickers(symbol_type=TickerType.COMMON_STOCKS, all_pages=True)
                all_etf_tickers = reference_client.get_tickers(symbol_type=TickerType.ETF, all_pages=True)
                self._all_tickers = all_etf_tickers + all_stock_tickers
                self._all_tickers = [t['ticker'] for t in self._all_tickers]
        else:
            self._all_tickers = [str.upper(t) for t in tickers]
        reference_client.close()

    def __exit__(self, *args): 
        self._stock_client.close()

    def reset(self, batch_size: int, steps: int) -> Tensor:
        tickers = self.rng.choice(self._all_tickers, size=batch_size, replace=True)        
        data = []
        previous_success = "SPY" if "SPY" in tickers else None
        for ticker in tickers:
            fail_counter = 0
            while True:
                price = None
                if self._local_cache != None:
                    df = pd.read_csv(f"{self._local_cache}/{ticker}.csv")
                    price = df[self._price_key].to_numpy(dtype=np.float32)
                    try:
                        df = pd.read_csv(f"{self._local_cache}/{ticker}.csv")
                        price = df[self._price_key].to_numpy(dtype=np.float32)
                    except:
                        pass
                if price is None:
                    price = self._stock_client.get_aggregate_bars(
                                    ticker,
                                    timespan='hour',
                                    from_date=date.today()-timedelta(days=3650),
                                    to_date=date.today(),
                                    full_range=True
                                )
                    try:
                        price = np.array([p[self._price_key] for p in price], dtype=np.float32)
                    except:
                        ticker = self.rng.choice(tickers)
                        fail_counter += 1
                        if fail_counter > 10:
                            if not previous_success is None:
                                price = previous_success
                                break
                            raise ValueError("Unable to find ticker with sufficient steps")
                        continue
                if len(price) < steps:
                    ticker = self.rng.choice(tickers)
                    fail_counter += 1
                    if fail_counter > 10:
                        if not previous_success is None:
                            price = previous_success
                            break
                        raise ValueError("Unable to find ticker with sufficient steps")
                else:
                    break
            previous_success = price
            start = self.rng.integers(low=0, high=len(price)-steps+1)
            data.append(torch.from_numpy(price[start:start+steps]))
        return torch.stack(data)

    def seed(self, seed):
        if not seed is None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)