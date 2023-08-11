import unittest

from torch import nn
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.collectors.collectors import SyncDataCollector

from reinforce.env import AssetEnv
from reinforce.data.dataloader import MockAssetDataloader

class TestRollout(unittest.TestCase):
    def test_AssetEnv(self):
        # Simple rollout
        dataloader = MockAssetDataloader()
        env = AssetEnv(dataloader, seed=1)
        env.rollout(20)

        # TorchRL Collector
        policy = SafeModule(
            nn.Linear(
                env.observation_spec['observation'].shape[-1],
                env.action_spec.shape[-1]
            ),
            in_keys=["observation"],
            out_keys=[env.action_key]
        )

        collector = SyncDataCollector(
            create_env_fn=env,
            policy=policy,
            frames_per_batch=100,
            total_frames=1000,
            device='cpu'
        )

        for i, _td in enumerate(collector):
            pass

        collector.shutdown()