from collections import defaultdict

from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from torch import nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.envs.transforms import TransformedEnv, Compose, RewardSum
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from reinforce.env import AssetEnv
from reinforce.data.dataloader import PolygonAssetDataloader


def main():
    DEVICE = 'cpu' if not torch.has_cuda else 'cuda:0'

    # Create Environments
        # Replace with your own Polygon cache
    CACHE_PATH = "/home/markhaoxiang/Projects/quantify/quantify-research/data/scripts/polygon_io/data"
    base_dataloader = PolygonAssetDataloader(local_cache=CACHE_PATH)
    env = AssetEnv(base_dataloader, batch_size=10, device=DEVICE)
    env.reset()
    eval_dataloader = PolygonAssetDataloader(local_cache=CACHE_PATH, tickers=['SPY'])
    eval_env = AssetEnv(eval_dataloader, max_steps=40000, device=DEVICE)
    eval_env.reset()

    env = TransformedEnv(
        env,
        Compose(
            RewardSum(in_keys=[env.reward_key])
        )
    )
    """
    eval_env = TransformedEnv(
        env,
        Compose(
            RewardSum(in_keys=[eval_env.reward_key])
        )
    )
    """
    check_env_specs(env)

    # Create actor / critic networks
    NUM_CELLS = 128
    actor_net = nn.Sequential(
        nn.LazyLinear(NUM_CELLS, device=DEVICE),
        nn.Tanh(),
        nn.LazyLinear(NUM_CELLS, device=DEVICE),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=DEVICE),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.minimum,
            "max": env.action_spec.space.maximum,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )

    value_net = nn.Sequential(
        nn.LazyLinear(NUM_CELLS, device=DEVICE),
        nn.Tanh(),
        nn.LazyLinear(NUM_CELLS, device=DEVICE),
        nn.Tanh(),
        nn.LazyLinear(NUM_CELLS, device=DEVICE),
        nn.Tanh(),
        nn.LazyLinear(1, device=DEVICE),
    )
    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    policy_module(env.reset())
    value_module(env.reset())

    # Create collectors and replay buffers
    TOTAL_FRAMES = 1000000
    FRAMES_PER_BATCH = 10000
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=FRAMES_PER_BATCH,
        total_frames=TOTAL_FRAMES,
        split_trajs=False,
        device=DEVICE,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(FRAMES_PER_BATCH),
        sampler=SamplerWithoutReplacement(),
    )


    # Define loss
    CLIP_EPSILON = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    GAMMA = 0.99
    LMBDA = 0.95
    ENTROPY_EPS = 1e-4
    LR = 3e-4

    advantage_module = GAE(
        gamma=GAMMA, lmbda=LMBDA, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
        clip_epsilon=CLIP_EPSILON,
        entropy_bonus=bool(ENTROPY_EPS),
        entropy_coef=ENTROPY_EPS,
        # these keys match by default but we set this for completeness
        value_target_key=advantage_module.value_target_key,
        critic_coef=1.0,
        gamma=0.99,
        loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, TOTAL_FRAMES // FRAMES_PER_BATCH, 0.0
    )

    logs = defaultdict(list)
    pbar = tqdm(total=TOTAL_FRAMES)
    eval_str = ""

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    SUB_BATCH_SIZE = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    NUM_EPOCHS = 10  # optimisation steps per batch of data collected
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(NUM_EPOCHS):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            with torch.no_grad():
                advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(FRAMES_PER_BATCH // SUB_BATCH_SIZE):
                subdata = replay_buffer.sample(SUB_BATCH_SIZE)
                loss_vals = loss_module(subdata.to(DEVICE))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optim step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data[("next", "reward")].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 8.8f} (init={logs['reward'][0]: 8.8f})"
        )
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 8.8f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our env horizon).
            # The ``rollout`` method of the env can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = eval_env.rollout(10000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout[("next", "reward")].sum().item()
                )
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 8.8f} "
                    f"(init: {logs['eval reward (sum)'][0]: 8.8f})"
                )
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.savefig("result.png")


    base_dataloader.__exit__()
    eval_dataloader.__exit__()

if __name__ == "__main__":
    main()