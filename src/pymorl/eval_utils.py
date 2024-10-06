from typing import Dict, Optional
import logging

from omegaconf import DictConfig
import numpy as np
import torch
import tqdm

from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor, OneHotCategorical
from torchrl.envs.transforms import TransformedEnv
from tensordict.nn.probabilistic import InteractionType, interaction_type, set_interaction_type

from env import RewardImportanceTransform
from advantage import compute_advantage

logger = logging.getLogger(__name__)


def gamma_returns(rewards: torch.Tensor | np.ndarray, gamma: float) -> torch.Tensor | np.ndarray:
    is_torch = isinstance(rewards, torch.Tensor)
    if is_torch:
        dev = rewards.device
        rewards = rewards.cpu().numpy()

    ret = np.zeros(tuple(rewards.shape))
    ret[-1] = rewards[-1]
    for i in range(len(ret) - 2, -1, -1):
        ret[i] = rewards[i] + gamma * ret[i + 1]

    if is_torch:
        ret = torch.from_numpy(ret).to(dev).float()
    return ret


def eval_pareto(actor: ProbabilisticActor, env: EnvBase, config: DictConfig, mixes: np.ndarray, use_mode: bool = False):
    inter_type = InteractionType.MODE if use_mode else InteractionType.RANDOM
    num_unrolls = config.eval.num_unrolls if (not use_mode) or config.env.is_stochastic else 1

    with set_interaction_type(inter_type):
        num_rewards = env.reward_spec.shape[0]
        rew = []
        for mix in tqdm.tqdm(mixes, desc='eval', position=0, leave=True):
            transformed_env = TransformedEnv(env, RewardImportanceTransform(num_rewards, weight=torch.from_numpy(mix)))
            sum_rew = np.zeros(num_rewards)
            for unroll_ind in range(num_unrolls):
                data = transformed_env.rollout(max_steps=config.env.max_episode_steps, policy=actor, auto_reset=True,
                                               break_when_any_done=True)
                eval_reward = data["next", "reward"].cpu().numpy()
                ret = gamma_returns(eval_reward, config.training.advantage.gamma)[0]
                sum_rew += ret
            rew.append(sum_rew / num_unrolls)
            # print(f'avg rev:{rew[-1]}')

    return np.array(rew)


def eval_fixed_head_rewards(actor: ProbabilisticActor, env: EnvBase, config: DictConfig, name: str):
    num_rewards = env.reward_spec.shape[0]
    mixes = np.eye(num_rewards, dtype=np.float32)
    rewards = eval_pareto(actor, env, config, mixes)
    return {f"eval_{name}_head{i}_ret": rewards[i, i] for i in range(num_rewards)}


def average_num_steps(next_done: torch.Tensor) -> float:
    sum_len = 0.0
    num_len = 0
    for i in range(next_done.shape[0]):
        cur_len = 1
        for j in range(next_done.shape[1]):
            if next_done[i, j, 0]:
                sum_len += cur_len
                cur_len = 1
                num_len += 1
            else:
                cur_len += 1
        if cur_len > 1:
            sum_len += cur_len
            num_len += 1

    if num_len == 0:
        logger.warning(f'Got empty next_done!')
        return 0.0
    else:
        return sum_len / num_len


def average_full_return(rewards: torch.Tensor, next_done: torch.Tensor, gamma: float) -> torch.Tensor:
    assert rewards.shape[:2] == next_done.shape[:2], f"Expected rewards and next_done to have the same shape[:2], " \
                                                     f"got {rewards.shape} and {next_done.shape}"
    assert rewards.ndim == 3, f"Expected rewards to be 3D, got {rewards.ndim}"
    if rewards.numel() == 0:
        logger.warning(f'Got empty rewards!')
        return 0.0

    returns_cfg = DictConfig({"estimator": "td1_no_bootstrapping", "gamma": gamma})
    full_val_shape = (rewards.shape[0], rewards.shape[1] + 1, rewards.shape[2])
    _, returns = compute_advantage(reward=rewards,
                                   full_predicted_value=torch.zeros(full_val_shape, device=rewards.device,
                                                                    dtype=rewards.dtype),
                                   next_terminated=next_done, next_done=next_done, advantage_cfg=returns_cfg)
    num_returns = returns.shape[0]
    sum_returns = returns[:, 0, :].sum(axis=0)
    done_inds = torch.where(next_done[:, :, 0])
    for i, j in zip(*done_inds):
        if j + 1 < returns.shape[1]:
            sum_returns += returns[i, j + 1]
            num_returns += 1

    return sum_returns / num_returns
