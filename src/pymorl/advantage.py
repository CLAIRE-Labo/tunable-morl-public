from typing import Optional

from omegaconf import DictConfig
import torch

from torchrl.objectives.value.functional import td0_return_estimate, td1_return_estimate, generalized_advantage_estimate
from torchrl.objectives.value.advantages import _call_value_nets
from torchrl.objectives.value import GAE, TD1Estimator
from torchrl.objectives.utils import hold_out_net
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictModuleBase

from policy import MultiRewardPolicy


# This module computes scalarized advantage from multi-objective vector advantage.
# It assumes that the vector advantage is already in the tensordict.
# If PopArt is present and training.advantage.popart_normalized is True, the vector advantage is
# normalized before scalarization
class ScalarizedAdvantage(TensorDictModuleBase):
    def __init__(self, policy: MultiRewardPolicy, advantage_module: TensorDictModuleBase,
                 popart_normalized_target_value_key: str = "popart_normalized_value_target",
                 popart_normalized_advantage_key="popart_normalized_advantage",
                 scalarized_advantage_key: str = "scalarized_advantage",
                 popart_normalized_advantage: bool = False):
        in_keys = advantage_module.in_keys
        out_keys = advantage_module.out_keys
        if policy.does_use_popart:
            out_keys.append(popart_normalized_target_value_key)
        if popart_normalized_advantage:
            out_keys.append(popart_normalized_advantage_key)
        out_keys.append(scalarized_advantage_key)
        super().__init__()

        self.policy = policy
        self.popart_normalized_target_value_key = popart_normalized_target_value_key
        self.popart_normalized_advantage_key = popart_normalized_advantage_key
        self.scalarized_advantage_key = scalarized_advantage_key
        self.popart_normalized = popart_normalized_advantage
        if self.popart_normalized:
            assert policy.does_use_popart, "PopArt normalization is requested but policy does not use PopArt"

        self.in_keys = in_keys
        self.out_keys = out_keys

    def forward(self, tensordict):
        td_out = tensordict

        assert "value_target" in td_out.keys(), \
            f"Advantage module should return 'value_target' key, got {td_out.keys()}"
        if self.policy.does_use_popart:
            td_out[self.popart_normalized_target_value_key] \
                = self.policy.popart_normalize_value(td_out["value_target"], recenter=True)

        assert "advantage" in td_out.keys(), f"Advantage module should return 'advantage' key, got {td_out.keys()}"
        advantage = td_out["advantage"]
        if self.popart_normalized:
            advantage = self.policy.popart_normalize_value(advantage, recenter=False)
            td_out[self.popart_normalized_advantage_key] = advantage

        assert "reward_importance" in td_out.keys(), \
            f"Advantage module should return 'reward_importance' key, got {td_out.keys()}"
        reward_importance = td_out["reward_importance"]
        assert advantage.shape == reward_importance.shape, \
            f"Advantage shape {advantage.shape} does not match reward_importance shape {reward_importance.shape}"
        td_out[self.scalarized_advantage_key] = (advantage * tensordict.get("reward_importance")).sum(-1, keepdim=True)

        return td_out


def create_advantage_estimator(advantage_cfg: DictConfig, policy: MultiRewardPolicy, value_network: TensorDictModule,
                               value_key: str) -> (TensorDictModuleBase, ScalarizedAdvantage):
    if advantage_cfg.estimator == "td1":
        # estimator = NonVectorizedTD1(advantage_cfg, value_network)
        estimator = TD1Estimator(gamma=advantage_cfg.gamma, value_network=value_network, average_rewards=False,
                                 differentiable=False, skip_existing=False)
    elif advantage_cfg.estimator == "gae":
        estimator = GAE(gamma=advantage_cfg.gamma, lmbda=advantage_cfg.lmbda, value_network=value_network,
                        average_gae=False, differentiable=False, vectorized=False, skip_existing=False)
    else:
        raise ValueError(f"Non-implemented advantage estimator: {advantage_cfg.estimator}")
    estimator.set_keys(value=value_key)

    scalarized = ScalarizedAdvantage(policy, estimator, scalarized_advantage_key="scalarized_advantage",
                                     popart_normalized_advantage=advantage_cfg.popart_normalized)

    return estimator, scalarized


# PopArt normalization of targets/advantages is not performed here because PopArt statistics are not yet updated
def compute_advantage(reward: torch.Tensor, full_predicted_value: torch.Tensor, next_terminated: torch.Tensor,
                      next_done: torch.Tensor, advantage_cfg: DictConfig):
    expected_shape = list(reward.shape)
    expected_shape[-2] += 1
    assert list(full_predicted_value.shape) == expected_shape, \
        f"Predicted value shape {full_predicted_value.shape} does not match rewards shape {reward.shape}" \
        f" (we expect an extra prediction at the end)"

    # A hack: we use multiobjective (vector) rewards and values, but computation of the target for them is the same
    # as a computation for vectorized single-objective envs
    next_done_tiling = (1,) * (len(reward.shape) - 1) + (full_predicted_value.shape[-1],)
    next_done = torch.tile(next_done, next_done_tiling)
    next_terminated = torch.tile(next_terminated, next_done_tiling)

    cur_predicted_value = full_predicted_value[..., :-1, :]
    next_predicted_value = full_predicted_value[..., 1:, :]
    if advantage_cfg.estimator == "td1_no_bootstrapping":
        next_predicted_value = torch.zeros_like(next_predicted_value)

    if advantage_cfg.estimator == "td0":
        # TODO one day I might fix the incorrect return type annotation in td0_return_estimate
        value_target = td0_return_estimate(advantage_cfg.gamma, next_predicted_value, reward,
                                           terminated=next_terminated, done=next_done)
        advantage = value_target - cur_predicted_value
    elif advantage_cfg.estimator in ["td1", "td1_no_bootstrapping"]:
        value_target = td1_return_estimate(advantage_cfg.gamma, next_predicted_value, reward,
                                           terminated=next_terminated, done=next_done)
        advantage = value_target - cur_predicted_value
    elif advantage_cfg.estimator == "gae":
        advantage, value_target = generalized_advantage_estimate(advantage_cfg.gamma, advantage_cfg.lmbda,
                                                                 cur_predicted_value, next_predicted_value, reward,
                                                                 terminated=next_terminated, done=next_done)
    else:
        raise ValueError(f"Non-implemented advantage estimator: {advantage_cfg.estimator}")

    return advantage, value_target
