import sys
from pathlib import Path
import logging
from copy import deepcopy
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np

import torch
from tensordict import TensorDict
from torchrl.envs.libs import MOGymEnv
from torchrl.envs.transforms import TransformedEnv, RewardSum
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers import samplers
from torchrl.objectives import ReinforceLoss, PPOLoss

from pymorl.policy import create_policy
from pymorl.training import MyReinforce
from pymorl.env import create_env
from pymorl.advantage import create_advantage


@hydra.main(version_base=None, config_path="configs", config_name="run_multihead")
def main(config: DictConfig) -> None:
    # TODO put all wandb, algo params in config
    # TODO proper seeding
    print(config)
    wandb.init(
        # config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        project=config.wandb.project,
        # tags=config.wandb.tags,
        anonymous=config.wandb.anonymous,
        mode=config.wandb.mode,
        dir=Path(config.wandb.dir).absolute(),
    )

    env, pixels_env = create_env(device=config.device, return_pixels_env=True, **config.env)
    # env = TransformedEnv(env, RewardSum(in_keys=["reward"], out_keys=["cumulative_reward"]))
    logging.info(env)

    print(env.observation_spec)
    print(env.observation_spec["observation"].shape)
    print(type(env.observation_spec["observation"].shape))
    print(env.action_spec)
    print(env.reward_spec)

    # module = create_multihead_mlp_discrete_policy(num_obs=2, num_actions=4, **config.old_model)
    creature = create_policy(env, device=config.device, discrete=config.env.discrete, policy_config=config.model)
    # creature = MultiheadMlpDiscreteActorValue(num_obs=env.observation_spec["observation"].shape, num_actions=4, **config.old_model)
    print(creature.module)

    reward_ind = 0
    # TODO modify the spec to include logprob
    actor_i = creature.policy_operator(reward_ind, env.action_spec)
    value_critic_i = creature.value_operator(reward_ind)

    # print(actor_1)
    # print(value_critic_1)
    trc = config.training
    # TODO separate num_epochs for actor and critic
    data_collector = SyncDataCollector(env, actor_i, frames_per_batch=trc.frames_per_batch,
                                       total_frames=trc.frames_per_batch * trc.num_iter, device=config.device,
                                       reset_at_each_iter=True)

    if trc.name == 'ppo':
        loss_module = PPOLoss(actor=actor_i, critic=value_critic_i, entropy_coef=trc.entropy_coef)
        loss_module.set_keys(value=f"values_{reward_ind}")
        buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=trc.frames_per_batch, device=config.device),
                              sampler=samplers.SamplerWithoutReplacement(),
                              batch_size=trc.frames_per_subbatch, prefetch=trc.prefetch)
    elif trc.name == 'reinforce':
        loss_module = MyReinforce(actor_i, entropy_coef=trc.entropy_coef, normalize=trc.get("normalize", False))
    else:
        raise NotImplementedError(f"Unknown training name: {trc.name}")

    advantage_estimator = create_advantage(advantage_name=trc.advantage_name, gamma=trc.gamma,
                                           value_estimator=value_critic_i, value_key=f"values_{reward_ind}",
                                           **trc.get('advantage_kwargs', {}))

    # TODO write reinforce from scratch
    # loss_module = ReinforceLoss(actor=actor_i, critic=value_critic_i)
    # loss_module.set_keys(value_target=f"values_{reward_ind}")
    # loss_module.make_value_estimator(ValueEstimators.TD1)
    # loss_module.make_value_estimator(ValueEstimators.TD0)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(creature.module.parameters(), lr=trc.lr)

    x, y = np.meshgrid(np.arange(11, dtype=np.int32), np.arange(11, dtype=np.int32))
    # produces [[0 0], [0 1], ... [10 10]]
    all_obs = np.stack([y, x], axis=-1).reshape((-1, 2))
    all_obs = torch.from_numpy(all_obs).to(config.device)
    all_obs_td = TensorDict({"observation": all_obs}, [all_obs.shape[0]], device=config.device)
    # Training loop
    for i, data in tqdm(enumerate(data_collector), total=trc.num_iter):
        data["next", "reward"] = (data["next", "reward"][:, reward_ind, None]
                                  + 0.05 * data["next", "reward"][:, 1-reward_ind, None])
        data["unnorm_reward"] = data["next", "reward"][:, reward_ind]
        # TD1 updates the rewards in-place
        with torch.no_grad():
            advantage_estimator(data)
        if trc.name == 'ppo':
            buffer.empty()
            buffer.extend(data)
            for _ in range(trc.num_epochs):
                # Add data to the replay buffer
                # Sample a batch of data
                for ib in range(trc.batch_size // trc.frames_per_subbatch):
                    batch = buffer.sample(trc.frames_per_subbatch)
                    for ic in range(trc.num_iter_critic):
                        loss_critic = loss_module.loss_critic(batch).mean()
                        if ic == 0:
                            loss_critic_init = loss_critic.item()
                        # loss = loss_vals["loss_actor"] + loss_vals["loss_value"]
                        loss_critic.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    loss_vals = loss_module(batch)
                    loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                    loss_critic_fin = loss_vals["loss_critic"].item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # total_reward = batch["next", "reward"].sum()
                    wandb.log(
                        {"loss": loss.item(), "loss_critic_init": loss_critic_init, "loss_critic_fin": loss_critic_fin,
                         "entropy": loss_vals["entropy"].item(), "loss_entropy": loss_vals["loss_entropy"].item(),
                         "loss_objective": loss_vals["loss_objective"].item(), "iter": i})
        elif trc.name == 'reinforce':
            if not torch.any(data["next", "done"]):
                continue
            last_done = torch.max(torch.nonzero(data["next", "done"])).item()
            data = data[0:last_done + 1]

            loss_vals = loss_module(data)
            loss = loss_vals["loss_objective"] + loss_vals["loss_entropy"]
            loss.backward()
            grads = torch.cat([p.grad.flatten() for p in creature.module.parameters() if p.grad is not None])
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"loss": loss.item(), "iter": i, "entropy": loss_vals["entropy"].item(),
                       "loss_entropy": loss_vals["loss_entropy"].item(),
                       "loss_objective": loss_vals["loss_objective"].item(), "grad_norm": grads.norm().item()})

        with torch.no_grad():
            cum_reward = torch.cumsum(data["unnorm_reward"], dim=0)
            # Select the last cumulative reward for each episode
            cum_reward = cum_reward[torch.flatten(data["next", "done"]), ...]
            # Compute the cumulative reward between terminations
            cum_reward[1:] -= cum_reward[:-1].clone()
            wandb.log({"episode_reward": cum_reward.mean().item(), "episode_reward_std": cum_reward.std().item(),
                       "iter": i})

        if i % 30 == 0:
            render_rollout_data_ = pixels_env.reset()
            if config.env.name == 'deep-sea-treasure-v0':
                dist = actor_i.get_dist(all_obs_td)
                prob = dist.prob.detach().cpu().numpy()
                maxind = np.argmax(prob, axis=1)
                # Define a mapping from action indices to emojis
                emoji_map = {0: '🔼', 1: '🔽', 2: '◀️', 3: '▶️'}
                # Get the action with the maximum probability for each cell
                # Map the action indices to emojis and reshape to an 11x11 grid
                emoji_grid = np.array([emoji_map[i] for i in maxind]).reshape((11, 11))
                emoji_rows = [[''.join(row)] for row in emoji_grid]
                tab = wandb.Table(data=emoji_rows, columns=["best action"])
                wandb.log({"act": tab})
            # emoji_string = '\n'.join(emoji_rows)
            # tab = wandb.Table(data=[[emoji_string]], columns=["best action"])

            # print(cum_reward)
            images = []
            for action in data["action"]:
                # print(action)
                render_rollout_data_.set("action", action, inplace=True)
                # render_rollout_data = pixels_env.step(render_rollout_data)
                # pixels_env.step(render_rollout_data)
                out_data, render_rollout_data_ = pixels_env.step_and_maybe_reset(render_rollout_data_)
                images.append(out_data["pixels"].cpu().numpy())
                # print(f'{np.linalg.norm(images[0] - images[-1])=}')

            # create video from images
            wandb.log({"video": wandb.Video(np.stack(images, axis=0).transpose(0, 3, 1, 2), fps=8, format="mp4"),
                       "iter": i})


if __name__ == "__main__":
    main()
