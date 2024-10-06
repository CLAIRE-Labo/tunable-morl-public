import os

# Fixing a random bug in pygame
# See https://stackoverflow.com/questions/15933493/pygame-error-no-available-video-device
os.environ["SDL_VIDEODRIVER"] = "dummy"

from typing import Literal, Optional, List

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import CompositeSpec, BoundedTensorSpec
from torchrl.envs.libs import MOGymEnv
from torchrl.envs import EnvBase, TransformedEnv, DTypeCastTransform, DeviceCastTransform, ObservationNorm, Compose, \
    RewardScaling, StepCounter, Transform

from sampling import uniform_simplex


# This is a TorchRL env transform that adds relative reward importance to the observation. It can either be random or
# provided by the user.
class RewardImportanceTransform(Transform):
    def __init__(self, num_rewards: int, weight_key: str = "reward_importance", dtype: torch.dtype = torch.float32,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight_key = weight_key
        self.num_rewards = num_rewards
        self.dtype = dtype
        if weight is not None:
            self.init_weight = weight.type(dtype)
            self.weight = torch.clone(self.init_weight)
        else:
            self.init_weight = None
            self.weight = self._random_weight()

    def _random_weight(self) -> torch.Tensor:
        return torch.from_numpy(uniform_simplex((self.num_rewards,))).type(self.dtype)

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        if self.init_weight is not None:
            self.weight = torch.clone(self.init_weight)
        else:
            self.weight = self._random_weight()
        self.weight = self.weight.to(tensordict.device)
        tensordict_reset[self.weight_key] = self.weight
        return tensordict_reset

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict[self.weight_key] = self.weight.to(tensordict.device)
        return tensordict

    def transform_observation_spec(self, observation_spec: CompositeSpec) -> CompositeSpec:
        if not isinstance(observation_spec, CompositeSpec):
            raise ValueError(
                f"observation_spec was expected to be of type CompositeSpec. Got {type(observation_spec)} instead."
            )
        observation_spec[self.weight_key] = BoundedTensorSpec(
            shape=torch.Size((self.num_rewards,)),
            dtype=self.dtype,
            device=observation_spec.device,
            low=0.0,
            high=1.0,
        )
        return observation_spec


def create_env(source: str, name: str, return_pixels_env: bool = False, discrete: bool = False, device: str = 'cuda',
               hypervolume_reference: Optional[np.ndarray | List] = None, is_stochastic: bool = True,
               reward_scale: Optional[np.ndarray] = None, **kwargs) -> EnvBase | tuple[EnvBase, EnvBase]:
    # hypervolume reference gets read from the config later, here we just prevent it from being passed to the env
    # same for is_stochastic
    if source == 'mo-gym':
        env = MOGymEnv(env_name=name, **kwargs)
        print(f'{env.observation_spec=}')
        print(f'{env.observation_spec["observation"]}')
        num_rewards = env.reward_spec.shape[0]
        transforms = []
        obs_norm_transform_ind = -1

        if discrete:
            assert name.startswith('deep-sea-treasure'), f'Only deep-sea-treasure is supported for now, got {name}'
        else:
            if device == 'cuda':
                transforms.append(DeviceCastTransform(device=device))
            if env.observation_spec["observation"].dtype != torch.float32:
                transforms.append(DTypeCastTransform(in_keys='observation',
                                                     dtype_in=env.observation_spec["observation"].dtype,
                                                     dtype_out=torch.float32))

            if name == 'mo-reacher-v4':
                # Reacher has observations [sin(theta), cos(theta), sin(phi), cos(phi), vx, vy],
                # where theta and phi are the angles of the two joints. We want to normalize them to [0, 1]
                # vx and vy are the velocities of the end effector. For them, we pre-computed normalization stats
                # using a random policy.
                loc = torch.tensor([0.5, 0.5, 0.5, 0.5, -0.0319, 0.2023], device=device, dtype=torch.float32)
                scale = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.8212, 1.2094], device=device, dtype=torch.float32)
            else:
                obs_space = env.observation_spec['observation'].space
                low = obs_space.low.to(device)
                high = obs_space.high.to(device)
                loc = -low / (high - low)
                scale = 1.0 / (high - low)
            # print(f'{low=}, {high=}')
            obs_norm_transform_ind = len(transforms)
            transforms.append(ObservationNorm(in_keys="observation", scale=scale, loc=loc))

            if reward_scale is not None:
                reward_scale = np.array(reward_scale)
                assert reward_scale.shape == (num_rewards,), f'{reward_scale.shape=}, {num_rewards=}'
                loc = torch.zeros(num_rewards, device=device)
                scale = torch.from_numpy(reward_scale).to(device).float()
                # RewardScaling doesn't work here
                transforms.append(ObservationNorm(in_keys="reward", loc=loc, scale=scale))

        transform = Compose(*transforms)
        env = TransformedEnv(env, transform)
        if device == 'cuda':
            env.cuda()

        # We pre-computed everything for now
        # if obs_norm_transform_ind != -1 and not env.transform[obs_norm_transform_ind].initialized:
        #     env.transform[obs_norm_transform_ind].init_stats(num_iter=10000)

        if return_pixels_env:
            pixels_env = MOGymEnv(env_name=name, from_pixels=True, pixels_only=False, **kwargs)
            pixels_env = TransformedEnv(pixels_env, transform)
            if device == 'cuda':
                pixels_env.cuda()
            return env, pixels_env
        else:
            return env
    else:
        raise ValueError(f'Unsupported (yet?) env source {source}')
