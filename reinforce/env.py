from typing import Optional
import logging

import torch
from torch import Tensor
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from torchrl.envs import EnvBase

from reinforce.data.dataloader import AssetDataloader

class AssetEnv(EnvBase):
    """ Represents trading assets with an option to buy/sell at the current market price

    Assumes that trades are instantly filled, there exists no spread and that they do not impact
    the market price
    """
    def __init__(self,
                 dataloader: AssetDataloader,
                 slippage: float = 0.005,
                 allow_short: bool = True,
                 batch_size: int = 1,
                 window = 128,
                 max_steps = 1000,
                 seed: Optional[int] = None,
                 **kwargs):
        assert batch_size > 0, "Batch size must be a postive integer"
        super().__init__(**kwargs, batch_size=[batch_size])

        self._dataloader = dataloader
        self._allow_short = allow_short
        self._slippage = slippage
        self._window = window
        self._max_steps = max_steps

        if not seed is None:
            self._set_seed(seed)

        self.observation_spec = CompositeSpec(
            observation = UnboundedContinuousTensorSpec((*self.batch_size, self._window), device=self.device),
            shape=self.batch_size,
            device=self.device
        )

        self.action_spec = BoundedTensorSpec(
            minimum=-1.0 if self._allow_short else 0.0,
            maximum=1.0,
            shape=(*self.batch_size, 1),
            device=self.device
        )
        self.action_key = "action"
        self.reward_key = "reward"

        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*self.batch_size,1), device=self.device)
        self.done_spec = DiscreteTensorSpec(n=2, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device)
    
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        # (batch_size, price history)
        self._history: Tensor = self._dataloader.reset(self.batch_size[0], self._max_steps+self._window-1).to(self.device)
        self._history = self._history / self._history[:, 0].unsqueeze(1) # Normalization for reward - do we want to do this dynamically with current price instead?
        # Money + Value of held assets
        self._value = torch.ones(self.batch_size).to(self.device)
        # Proportion of funds in the asset
        self._position = torch.zeros(self.batch_size).to(self.device)

        self._episode = self._history.unfold(1, self._window, 1)
        self._episode_step = 0 

        tensordict_out = TensorDict(
            {
                "observation": AssetEnv._normalize_by_current_price(self._episode[:, self._episode_step]),
                "done": Tensor([self.done]).to(torch.bool).to(self.device).expand(self.batch_size).unsqueeze(1).clone()
            },
            batch_size=self.batch_size,
            device=self.device
        )
        return tensordict_out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.done:
            raise Exception("Step has been called when environment is done")
        
        self._episode_step += 1
        price = self._episode[:, self._episode_step][:, -1]
        if torch.any(torch.isnan(price)):
            logging.warn("NaN found in price. Clean the dataloader.")
            price = torch.where(torch.isnan,self._episode[:, self._episode_step-1][:, -1], price)
        self._previous_value = self._value
        self._value = self._value * (1-self._position) + self._position * price
        desired_position = tensordict['action'].squeeze()
        if not self._allow_short:
            torch.clip(desired_position, min=0, out=desired_position)

        delta_position = desired_position - self._position
        self._position = desired_position
        self._value -= torch.abs(delta_position) * price * self._slippage

        tensordict_out = TensorDict(
            {
                "observation": AssetEnv._normalize_by_current_price(self._episode[:, self._episode_step]),
                "reward": (self._value-self._previous_value).unsqueeze(1),
                "done": Tensor([self.done]).to(torch.bool).to(self.device).expand(self.batch_size).unsqueeze(1).clone()
            },
            batch_size=self.batch_size,
            device=self.device
        )
        result = tensordict_out.select().set("next", tensordict_out)
        return result

    def _set_seed(self, seed):
        self._dataloader.seed(seed)

    @property
    def done(self):
        return self._episode_step >= self._max_steps - 1

    @staticmethod
    def _normalize_by_current_price(tensor: Tensor) -> Tensor:
        # Returns input normlized by the last slice
        return tensor / tensor[:, -1].unsqueeze(1)
