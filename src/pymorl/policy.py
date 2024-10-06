from __future__ import annotations
import sys
from copy import deepcopy, copy
from abc import ABC, abstractmethod
from typing import Optional
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential, ProbabilisticTensorDictSequential
# TODO why is InteractionType not explicitly exported?
from tensordict.nn.probabilistic import InteractionType
from torchrl.envs import EnvBase
from torchrl.data import TensorSpec, OneHotDiscreteTensorSpec
from torchrl.modules import ValueOperator, ProbabilisticActor, OneHotCategorical

logger = logging.getLogger(__name__)


class DiscreteInputConverter(nn.Module):
    def __init__(self, min_x: int, min_y: int, max_x: int, max_y: int):
        super().__init__()
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.y_diff = max_y - min_y + 1
        self.x_diff = max_x - min_x + 1
        self.num_actions = self.x_diff * self.y_diff

    def forward(self, x):
        flattened_index = (x[..., 0] - self.min_x) * self.y_diff + (x[..., 1] - self.min_y)
        # print(f'{torch.max(flattened_index)=}')
        return torch.nn.functional.one_hot(flattened_index.long(), num_classes=self.num_actions).float()


# If the input is 1D, then this assumes that the batch dimension is missing. It adds the first dimension to the input
# and removes it from the outputs
def handle_batch_dim(forward):
    def wrap(self, *args):
        assert len(args) > 0
        if args[0].ndim == 1:
            args = tuple([(arg[None, ...] if arg is not None else None) for arg in args])
            res = forward(self, *args)
            if isinstance(res, tuple):
                assert len(res) > 0
                for r in res:
                    assert r.shape[0] > 0
                return tuple([r[0, ...] for r in res])
            else:
                return res[0, ...]
        else:
            return forward(self, *args)

    return wrap


def check_prob(logits: torch.Tensor, logprob: torch.Tensor, prob: torch.Tensor, num_actions: int) \
        -> (torch.Tensor, torch.Tensor):
    if torch.any(~torch.isfinite(prob)) or torch.any(prob < 0):
        logger.warning(
            f'Non-finite or negative probabilities:\nprob:\n{prob}\nlogprob:\n{logprob}\nlogits:\n{logits}')
        logger.warning(f'{torch.any(torch.isnan(prob))=} {torch.any(torch.isinf(prob))=} {torch.any(prob < 0)=} '
                       f'{torch.any(torch.isnan(logprob))=} {torch.any(torch.isinf(logprob))=} '
                       f'{torch.any(torch.isnan(logits))=} {torch.any(torch.isinf(logits))=}')
        logger.warning('Using random actions for non-finite or negative probabilities')
        bad_rows = torch.any(~torch.isfinite(prob), dim=1) | torch.any(prob < 0, dim=1)
        if bad_rows.ndim == 2:
            bad_rows = torch.any(bad_rows, dim=1)
        prob[bad_rows, ...] = 1 / num_actions
        logprob = torch.log(prob)
    return logprob, prob


# Multi-task PopArt value normalization head as described in
# [1] https://arxiv.org/pdf/1809.04474.pdf
class PopartNormValueHead(nn.Module):
    def __init__(self, num_features: int, num_outputs: int, lr: float, check_numerics: bool = False):
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.lr = lr
        self.check_numerics = check_numerics

        self.linear = nn.Linear(num_features, num_outputs)
        self.fresh = True
        self.means = torch.zeros(num_outputs)
        self.sigmas = torch.ones(num_outputs)

    def normalize_targets(self, targets: torch.Tensor, recenter: bool = True) -> torch.Tensor:
        return ((targets - self.means[None, :]) / self.sigmas[None, :]) if recenter else (
                targets / self.sigmas[None, :])

    def update_normalization(self, targets: torch.Tensor):
        assert targets.ndim == 2, f"{targets.shape=} must be 2D"
        assert targets.shape[1] == self.num_outputs, f"{targets.shape=} must be (B, {self.num_outputs})"

        self.means = self.means.to(targets.device)
        self.sigmas = self.sigmas.to(targets.device)

        immediate_means = torch.mean(targets, dim=0)
        immediate_sigmas = torch.std(targets, dim=0)

        if self.check_numerics:
            with torch.no_grad():
                rand_obs = torch.rand(self.num_features, device=targets.device, dtype=targets.dtype)
                old_out = self(rand_obs)

        if self.fresh:
            if torch.any(immediate_sigmas <= 1e-8):
                logger.warning(f'{immediate_means=} {immediate_sigmas=}.'
                               ' Stds have near-zeros, not initializing PopArt!')
                return self.normalize_targets(targets)
            new_means = immediate_means
            new_sigmas = immediate_sigmas
            self.fresh = False
        else:
            new_means = (1 - self.lr) * self.means + self.lr * immediate_means
            new_sigmas = (1 - self.lr) * self.sigmas + self.lr * immediate_sigmas
            if torch.any(new_sigmas <= 1e-8):
                logger.warning(f'{immediate_means=} {new_means=} {immediate_sigmas=} {new_sigmas=}.'
                               f'Stds have near-zeros, not updating PopArt!')
                return self.normalize_targets(targets)

        with torch.no_grad():
            new_w = self.linear.weight.data * self.sigmas[:, None] / new_sigmas[:, None]
            new_b = (self.linear.bias.data * self.sigmas + self.means - new_means) / new_sigmas
        self.linear.weight.data = new_w
        self.linear.bias.data = new_b

        self.means = new_means
        self.sigmas = new_sigmas

        if self.check_numerics:
            with torch.no_grad():
                new_out = self(rand_obs)
                # diff = torch.abs(new_out - old_out).max()
                # print(f'{diff=}')
                assert torch.allclose(old_out, new_out), f"{old_out=} != {new_out}"

        return self.normalize_targets(targets)

    @handle_batch_dim
    def forward(self, x):
        self.means = self.means.to(x.device)
        self.sigmas = self.sigmas.to(x.device)
        linx = self.linear(x)
        assert linx.ndim == 2, f"{linx.shape=} must be 2D"
        unnorm_linx = linx * self.sigmas[None, :] + self.means[None, :]
        return linx, unnorm_linx


class MultiRewardCritic(nn.Module):
    def __init__(self, multi_reward_actor_critic: MultiRewardPolicy):
        super().__init__()
        self.multi_reward_actor_critic = multi_reward_actor_critic

    def forward(self, obs, reward_importance):
        res = self.multi_reward_actor_critic.predict_value(obs, reward_importance)
        return res


class MultiRewardPolicy(nn.Module, ABC):
    def __init__(self, num_obs: int, num_actions: int, num_rewards: int, **kwargs):
        super().__init__()
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.num_rewards = num_rewards

        self.init_config = {
            'cls': self.__class__.__name__,
            'num_obs': num_obs,
            'num_actions': num_actions,
            'num_rewards': num_rewards,
            **kwargs
        }
        logger.info(f'Created a MultiRewardPolicy with {self.init_config=}')

    @abstractmethod
    def forward(self, obs, reward_importance):
        pass

    def predict_value(self, obs, reward_importance):
        raise ValueError(f'{self.__class__.__name__} does not implement predict_value')

    @staticmethod
    def load_policy(init_config: DictConfig, state_dict: dict, device: str) -> MultiRewardPolicy:
        conf = deepcopy(init_config)
        logger.info(f'Loading policy with init_config={conf}')
        cls = getattr(sys.modules[__name__], conf['cls'], None)
        if cls is None:
            raise ValueError(f'Unknown policy class: {conf["cls"]}')
        conf.pop('cls')
        policy = cls(**conf)
        policy.to(device)
        policy.load_state_dict(state_dict)
        return policy

    @staticmethod
    def load_policy_from_checkpoint(checkpoint_file: Path, device: str) -> MultiRewardPolicy:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        assert 'policy_init_config' in checkpoint and 'policy_state_dict' in checkpoint, \
            f"{checkpoint_file} does not contain a policy: {checkpoint}"
        return MultiRewardPolicy.load_policy(checkpoint['policy_init_config'], checkpoint['policy_state_dict'], device)

    @property
    def does_use_popart(self) -> bool:
        return False

    def update_popart(self, value_target: torch.Tensor) -> torch.Tensor:
        if self.does_use_popart:
            return self.value_head.update_normalization(value_target)
        else:
            return value_target

    def popart_normalize_value(self, value_target: torch.Tensor, recenter: bool = True) -> torch.Tensor:
        if self.does_use_popart:
            return self.value_head.normalize_targets(value_target, recenter=recenter)
        else:
            return value_target

    def create_critic(self) -> ValueOperator:
        in_keys = ["observation", "reward_importance"]
        out_keys = ["popart_normalized_predicted_value", "predicted_value"] if self.does_use_popart \
            else ["predicted_value"]
        return ValueOperator(module=MultiRewardCritic(self), in_keys=in_keys, out_keys=out_keys)


class MultiBodyNetwork(nn.Module):
    def __init__(self, input_dim: int, num_hidden: int, num_bodies: int, add_extra: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_bodies = num_bodies
        self.add_extra = add_extra

        # TODO without head freezing, we do not need a module list, can use a 3D tensor for weights
        self.bodies = nn.ModuleList([nn.Linear(input_dim, num_hidden) for _ in range(num_bodies)])
        if self.add_extra:
            self.extra = nn.Linear(num_hidden, num_hidden)

    @handle_batch_dim
    def forward(self, x, reward_importance):
        x_stack = torch.stack([body(x) for body in self.bodies], dim=0)
        x_stack = torch.relu(x_stack)
        if len(reward_importance.shape) == 1:
            # heads are mixed using the same set of reward_importance
            # reward_importance = reward_importance.flatten()
            assert reward_importance.shape[
                       0] == self.num_bodies, f"{reward_importance.shape=} must have {self.num_bodies} elements"
            new_shape = (reward_importance.shape[0],) + (1,) * (x_stack.ndim - 1)
            x_stack = (reward_importance.reshape(new_shape) * x_stack).sum(dim=0)
        elif len(reward_importance.shape) == 2:
            # Each row of reward_importance is used to mix one row of heads
            assert reward_importance.shape[
                       1] == self.num_bodies, f"{reward_importance.shape=} must have {self.num_bodies} columns"
            if reward_importance.shape[0] != x_stack.shape[1]:
                assert reward_importance.shape[
                           0] == 1, f"{reward_importance.shape=} must have 1 row or {x_stack.shape[1]} rows"
                reward_importance = reward_importance.repeat((x_stack.shape[1], 1))
            x_stack = torch.einsum('ik,kij->ij', reward_importance, x_stack)
            assert tuple(x_stack.shape) == (reward_importance.shape[0], self.num_hidden)
        else:
            raise ValueError(f"{reward_importance.shape=} must be 1D or 2D")
        if self.add_extra:
            x_stack = self.extra(x_stack)
        return torch.relu(x_stack)


class MultiBodyPolicy(MultiRewardPolicy):
    def __init__(self, num_obs: int, num_hidden: int, num_actions: int, num_rewards: int,
                 add_intermediate: bool = False, separate_value_network: bool = False,
                 popart_norm: bool = False, popart_lr: float = 1e-3):
        args = copy(locals())
        args.pop('self')
        if '__class__' in args:
            args.pop('__class__')
        super().__init__(**args)

        self.num_hidden = num_hidden
        self.add_intermediate = add_intermediate

        self.policy_body = MultiBodyNetwork(num_obs, num_hidden, num_rewards, add_extra=add_intermediate)
        self.head = nn.Linear(num_hidden, num_actions)
        self.separate_value_network = separate_value_network
        self.popart_norm = popart_norm

        # Visualizations of networks for different configurations:
        #   separate_value_network=False:
        #      alpha ->
        #      obs   -> features -> action
        #                        -> values
        #
        #   separate_value_network=True:
        #      alpha ->
        #      obs   -> features  -> action
        #
        #      alpha ->
        #      obs   -> features' -> values
        #   If add_intermediate=True, then obs -> features and obs -> features' have one hidden layer in between
        if self.separate_value_network:
            self.value_body = MultiBodyNetwork(num_obs, num_hidden, num_rewards, add_extra=add_intermediate)
        self.value_head = PopartNormValueHead(num_hidden, num_rewards, popart_lr) if self.popart_norm \
            else nn.Linear(num_hidden, num_rewards)

    @property
    def does_use_popart(self) -> bool:
        return self.popart_norm

    @handle_batch_dim
    def forward(self, obs, reward_importance):
        x_stack = self.policy_body(obs, reward_importance)
        logits = self.head(x_stack)
        logprob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(logprob)
        logprob, prob = check_prob(logits, logprob, prob, self.num_actions)

        predicted_value = self.predict_value(obs, reward_importance) if self.separate_value_network \
            else self.value_head(x_stack)
        if self.does_use_popart:  # popart returns both normalized and unnormalized values
            return logprob, prob, *predicted_value
        else:
            return logprob, prob, predicted_value

    # If the critic uses PopArt normalization, we return both normalized and unnormalized values
    @handle_batch_dim
    def predict_value(self, obs, reward_importance):
        valx = self.value_body(obs, reward_importance) if self.separate_value_network \
            else self.policy_body(obs, reward_importance)
        return self.value_head(valx)


class MLP(nn.Module):
    def __init__(self, num_in: int, num_hidden: int, add_extra: bool):
        super().__init__()
        self.add_extra = add_extra
        self.fc1 = nn.Linear(num_in, num_hidden)
        if add_extra:
            self.extra = nn.Linear(num_hidden, num_hidden)

    @handle_batch_dim
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        if self.add_extra:
            x = torch.relu(self.extra(x))
        return x


class HyperHeadPolicy(MultiRewardPolicy):
    def __init__(self, num_obs: int, num_hidden: int, num_actions: int, num_rewards: int,
                 num_hyper_hidden: Optional[int] = None, add_intermediate: bool = False,
                 feed_obs_into_hyper: bool = False, separate_value_network: bool = False, popart_norm: bool = False,
                 popart_lr: float = 1e-3):
        assert not popart_norm, f"PopArt normalization not implemented for HyperHeadPolicy"

        args = copy(locals())
        args.pop('self')
        if '__class__' in args:
            args.pop('__class__')
        super().__init__(**args)

        self.add_intermediate = add_intermediate
        self.feed_obs_into_hyper = feed_obs_into_hyper
        self.separate_value_network = separate_value_network
        self.popart_norm = popart_norm
        self.popart_lr = popart_lr

        # main arch
        self.num_hidden = num_hidden
        self.fc1 = nn.Linear(num_obs, num_hidden)
        self.add_intermediate = add_intermediate
        logger.info(f'hyperhead add intermediate? {self.add_intermediate}')
        if self.add_intermediate:
            self.intermediate = nn.Linear(num_hidden, num_hidden)

        if self.separate_value_network:
            self.value_body = nn.Linear(num_obs, num_hidden)
            if self.add_intermediate:
                self.value_intermediate = nn.Linear(num_hidden, num_hidden)

        # hyper arch
        hyper_input_dim = num_rewards + num_obs if self.feed_obs_into_hyper else num_rewards
        # The hypernet returns the weights of the head of the last layer (kernel and bias)
        policy_hyper_out_dim = num_hidden * num_actions + num_actions
        if num_hyper_hidden is None:
            num_hyper_hidden = policy_hyper_out_dim
        self.hyper_fc1 = nn.Linear(hyper_input_dim, num_hyper_hidden)
        self.policy_hyper_head = nn.Linear(num_hyper_hidden, policy_hyper_out_dim)

        value_hyper_out_dim = num_hidden * num_rewards + num_rewards
        if self.separate_value_network:
            self.value_hyper_fc1 = nn.Linear(hyper_input_dim, num_hyper_hidden)
        self.value_hyper_head = nn.Linear(num_hyper_hidden, value_hyper_out_dim)

        actor_hyper_output_dim = num_hidden * num_actions + num_actions

    @property
    def does_use_popart(self) -> bool:
        return False

    @handle_batch_dim
    def forward(self, obs, reward_importance):
        x = torch.relu(self.fc1(obs))
        if self.add_intermediate:
            x = torch.relu(self.intermediate(x))

        if self.separate_value_network:
            valx = torch.relu(self.value_body(obs))
            if self.add_intermediate:
                valx = torch.relu(self.value_intermediate(valx))
        else:
            valx = x

        # TODO add an option to feed previous observation
        inp = torch.cat([reward_importance, obs], dim=1) if self.feed_obs_into_hyper else reward_importance
        hyper_x = torch.relu(self.hyper_fc1(inp))
        hyper_policy = self.policy_hyper_head(hyper_x)
        # Batched matrices for the last layer: each input gets its own weights
        W_policy = hyper_policy[:, :-self.num_actions].reshape((obs.shape[0], self.num_hidden, self.num_actions))
        b_policy = hyper_policy[:, -self.num_actions:]

        logits = torch.matmul(x[:, None, :], W_policy)[:, 0, :] + b_policy
        logprob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(logprob)
        logprob, prob = check_prob(logits, logprob, prob, self.num_actions)

        if self.separate_value_network:
            hyper_value = torch.relu(self.value_hyper_fc1(inp))
        else:
            hyper_value = hyper_x
        hyper_value = self.value_hyper_head(hyper_value)
        W_value = hyper_value[:, :-self.num_rewards].reshape((obs.shape[0], self.num_hidden, self.num_rewards))
        b_value = hyper_value[:, -self.num_rewards:]
        predicted_value = torch.matmul(valx[:, None, :], W_value)[:, 0, :] + b_value
        return logprob, prob, predicted_value

    @handle_batch_dim
    def predict_value(self, obs, reward_importance):
        if self.separate_value_network:
            valx = self.value_body(obs)
            if self.add_intermediate:
                valx = torch.relu(self.value_intermediate(valx))
        else:
            valx = torch.relu(self.fc1(obs))
            if self.add_intermediate:
                valx = torch.relu(self.intermediate(valx))

        inp = torch.cat([reward_importance, obs], dim=1) if self.feed_obs_into_hyper else reward_importance.float()
        if self.separate_value_network:
            hyper_value = torch.relu(self.value_hyper_fc1(inp))
        else:
            hyper_value = torch.relu(self.hyper_fc1(inp))
        hyper_value = self.value_hyper_head(hyper_value)
        W_value = hyper_value[:, :-self.num_rewards].reshape((obs.shape[0], self.num_hidden, self.num_rewards))
        b_value = hyper_value[:, -self.num_rewards:]
        predicted_value = torch.matmul(valx[:, None, :], W_value)[:, 0, :] + b_value
        return predicted_value


class AlphaInputPolicy(MultiRewardPolicy):
    def __init__(self, num_obs: int, num_hidden: int, num_actions: int, num_rewards: int,
                 add_intermediate: bool = False, separate_value_network: bool = False,
                 popart_norm: bool = False, popart_lr: float = 1e-3):
        args = copy(locals())
        args.pop('self')
        if '__class__' in args:
            args.pop('__class__')
        super().__init__(**args)

        self.policy_body = MLP(num_obs + num_rewards, num_hidden, add_extra=add_intermediate)
        self.policy_head = nn.Linear(num_hidden, num_actions)

        self.separate_value_network = separate_value_network
        self.popart_norm = popart_norm

        if self.separate_value_network:
            self.value_body = MLP(num_obs + num_rewards, num_hidden, add_extra=add_intermediate)
        self.value_head = PopartNormValueHead(num_hidden, num_rewards, popart_lr) if self.popart_norm \
            else nn.Linear(num_hidden, num_rewards)

    @property
    def does_use_popart(self) -> bool:
        return self.popart_norm

    @handle_batch_dim
    def forward(self, obs, reward_importance):
        combined = torch.cat([obs, reward_importance], dim=1)
        x = self.policy_body(combined)
        logits = self.policy_head(x)
        logprob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(logprob)
        logprob, prob = check_prob(logits, logprob, prob, self.num_actions)

        predicted_value = self.predict_value(obs, reward_importance) if self.separate_value_network \
            else self.value_head(x)
        if self.does_use_popart:  # popart returns both normalized and unnormalized values
            return logprob, prob, *predicted_value
        else:
            return logprob, prob, predicted_value

    @handle_batch_dim
    def predict_value(self, obs, reward_importance):
        inp = torch.cat([obs, reward_importance], dim=1)
        valx = self.value_body(inp) if self.separate_value_network else self.policy_body(inp)
        return self.value_head(valx)


class ImportanceEmbedNetwork(nn.Module):
    def __init__(self, num_obs: int, num_rewards: int, num_hidden: int, add_extra: bool):
        super().__init__()
        self.num_obs = num_obs
        self.num_rewards = num_rewards
        self.num_hidden = num_hidden
        self.add_extra = add_extra

        self.obs_embed = nn.Linear(num_obs, num_hidden)
        self.alpha_embed = nn.Linear(num_rewards, num_hidden)
        if self.add_extra:
            self.extra = nn.Linear(num_hidden, num_hidden)

    def forward(self, obs, reward_importance):
        obs_embed = torch.sigmoid(self.obs_embed(obs))
        alpha_embed = torch.sigmoid(self.alpha_embed(reward_importance))
        x = obs_embed * alpha_embed
        if self.add_extra:
            x = torch.relu(self.extra(x))
        return x


class AlphaEmbedPolicy(MultiRewardPolicy):
    def __init__(self, num_obs: int, num_hidden: int, num_actions: int, num_rewards: int,
                 add_intermediate: bool = False,
                 separate_value_network: bool = False, popart_norm: bool = False, popart_lr: float = 1e-3):
        args = copy(locals())
        args.pop('self')
        if '__class__' in args:
            args.pop('__class__')
        super().__init__(**args)

        self.add_intermediate = add_intermediate
        self.separate_value_network = separate_value_network
        self.popart_norm = popart_norm

        self.policy_body = ImportanceEmbedNetwork(num_obs, num_rewards, num_hidden, add_extra=add_intermediate)
        self.policy_head = nn.Linear(num_hidden, num_actions)

        if self.separate_value_network:
            self.value_body = ImportanceEmbedNetwork(num_obs, num_rewards, num_hidden, add_extra=add_intermediate)
        self.value_head = PopartNormValueHead(num_hidden, num_rewards, popart_lr) if self.popart_norm \
            else nn.Linear(num_hidden, num_rewards)

    @property
    def does_use_popart(self) -> bool:
        return self.popart_norm

    @handle_batch_dim
    def forward(self, obs, reward_importance):
        x = self.policy_body(obs, reward_importance)
        logits = self.policy_head(x)
        logprob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(logprob)
        logprob, prob = check_prob(logits, logprob, prob, self.num_actions)

        predicted_value = self.predict_value(obs, reward_importance) if self.separate_value_network \
            else self.value_head(x)
        if self.does_use_popart:  # popart returns both normalized and unnormalized values
            return logprob, prob, *predicted_value
        else:
            return logprob, prob, predicted_value

    @handle_batch_dim
    def predict_value(self, obs, reward_importance):
        valx = self.value_body(obs, reward_importance) if self.separate_value_network \
            else self.policy_body(obs, reward_importance)
        return self.value_head(valx)


def wrap_policy_into_actor(policy: MultiRewardPolicy) -> ProbabilisticActor:
    out_keys = ["logprob", "prob"]
    if policy.does_use_popart:
        out_keys.append("popart_normalized_predicted_value")
    out_keys.append("predicted_value")
    print(f'{out_keys=}')
    in_keys = ["observation", "reward_importance"]
    module = TensorDictModule(policy, in_keys=in_keys, out_keys=out_keys)
    return ProbabilisticActor(module=module, distribution_class=OneHotCategorical, in_keys={"probs": "prob"},
                              default_interaction_type=InteractionType.RANDOM)


def create_policy(config: DictConfig, env: EnvBase) -> (MultiRewardPolicy, ProbabilisticActor):
    num_obs = env.observation_spec["observation"].shape[0]
    num_hidden = config.model.num_hidden
    num_actions = env.action_spec.shape[0]
    num_rewards = env.reward_spec.shape[0]
    add_intermediate = config.model.get("add_intermediate", False)
    separate_value_network = config.model.get("separate_value_network", False)
    popart_norm = config.model.get("popart_norm", False)
    popart_lr = config.model.get("popart_lr", 1e-3)

    if config.model.type == 'multibody_mlp':
        policy = MultiBodyPolicy(num_obs, num_hidden, num_actions, num_rewards, add_intermediate=add_intermediate,
                                 separate_value_network=separate_value_network, popart_norm=popart_norm,
                                 popart_lr=popart_lr)
    elif config.model.type == 'hyper_mlp':
        num_hyper_hidden = config.model.get("num_hyper_hidden")  # will return None if not specified
        policy = HyperHeadPolicy(num_obs=num_obs, num_hidden=num_hidden, num_actions=num_actions,
                                 num_rewards=num_rewards, num_hyper_hidden=num_hyper_hidden,
                                 add_intermediate=add_intermediate,
                                 feed_obs_into_hyper=config.model.feed_obs_into_hyper,
                                 separate_value_network=separate_value_network, popart_norm=popart_norm,
                                 popart_lr=popart_lr)
    elif config.model.type == 'alpha_input_mlp':
        policy = AlphaInputPolicy(num_obs=num_obs, num_hidden=num_hidden, num_actions=num_actions,
                                  num_rewards=num_rewards, add_intermediate=add_intermediate,
                                  separate_value_network=separate_value_network, popart_norm=popart_norm,
                                  popart_lr=popart_lr)
    elif config.model.type == 'alpha_embed_mlp':
        policy = AlphaEmbedPolicy(num_obs=num_obs, num_hidden=num_hidden, num_actions=num_actions,
                                  num_rewards=num_rewards, add_intermediate=add_intermediate,
                                  separate_value_network=separate_value_network, popart_norm=popart_norm,
                                  popart_lr=popart_lr)
    else:
        raise ValueError(f'Unknown model type: {config.model.type}')
    policy.to(config.device)
    return policy, wrap_policy_into_actor(policy)
