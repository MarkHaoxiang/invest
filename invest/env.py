import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec
from torchrl.envs import EnvBase

from invest.data.dataloader import AssetDataloader

class SingleAssetEnv(EnvBase):
    """ Represents trading an asset with an option to buy sell at current market price.

    Assumes that our trades are instantly filled and do not impact the market price.
    """
    def __init__(self,
                 dataloader: AssetDataloader,
                 slippage: float = 0.0005,
                 allow_short: bool = True,
                 **kwargs):
        super().__init__(**kwargs, batch_size=[])
        self._dataloader = dataloader
        self._allow_short = allow_short
        self._slippage = slippage

        self.observation_spec = CompositeSpec(
            history = UnboundedContinuousTensorSpec((*self.batch_size, self._dataloader.window)),
            price = UnboundedContinuousTensorSpec((*self._batch_size, 1)),
            position = UnboundedContinuousTensorSpec((*self.batch_size, 1)),
            shape=self.batch_size
        )
        self.action_spec = BoundedTensorSpec(
            minimum=-1.0 if allow_short else 0.0,
            maximum=1.0,
            shape=(*self.batch_size, 1)
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*self.batch_size, 1))

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self._baseline_price, history = self._dataloader.reset()   # Normalization
        self._price = 1.0
        self._value    = 1.0                                       # Money + Value of asset
        self._position = 0.0                                       # Proportion of funds in the asset
        history = history / self._baseline_price
        out = TensorDict(
            {
                "next": {
                    "history":  torch.Tensor(history,          device=self.device),
                    "price":    torch.Tensor([self._price],    device=self.device),
                    "position": torch.Tensor([self._position], device=self.device)
                }
            },
            self.batch_size
        )
        return out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Obtain updates to the asset
        if self._dataloader.done:
            raise Exception("Step has been called when asset dataloader is complete")
        self._price, history = self._dataloader.step()
        self._previous_value = self._value
        self._value = self._value * (1-self._position) + self._position * self._price
        history = history / self._baseline_price
        # Process actions
        desired_position = tensordict['action']
        if abs(desired_position) > 1.0 or not self._allow_short and desired_position < 0:
            raise ValueError(f"The desired asset position {desired_position} is out of bounds.")
        delta_position = desired_position - self._position
        self._position = desired_position
        self._value -= abs(delta_position) * self._price * self._slippage
        # Package into output
        out = TensorDict(
            {
                "next": {
                    "history": torch.Tensor(history, device=self.device),
                    "price": torch.Tensor([self._price], device=self.device),
                    "position": torch.Tensor([self._position], device=self.device),
                    "reward": torch.Tensor([self._value-self._previous_value], device=self.device),
                    "done": torch.Tensor([self._dataloader.done], device=self.device).bool()
                }
            },
            tensordict.shape
        )

        return out
    
    def _set_seed(self, seed):
        pass
