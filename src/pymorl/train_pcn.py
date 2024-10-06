# Adapted from https://github.com/LucasAlegre/morl-baselines/blob/9afcb3a94a5b2611735aac370cd9230920264aed/examples/pcn_minecart.py

from typing import Optional, Union
from pathlib import Path

import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics
import torch as th
import torch.nn as nn

from morl_baselines.multi_policy.pcn.pcn import PCN
from morl_baselines.common.pareto import filter_pareto_dominated


# Add an option to have an extra depth layer to PCN
class AdaptedPCNModel(nn.Module):
    """Model for the PCN."""

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray,
                 hidden_dim: int = 64, use_extra_layer: bool = False):
        """Initialize the PCN model."""
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.scaling_factor = nn.Parameter(th.tensor(scaling_factor).float(), requires_grad=False)
        self.hidden_dim = hidden_dim
        self.use_extra_layer = use_extra_layer

        self.s_emb = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(self.reward_dim + 1, self.hidden_dim), nn.Sigmoid())
        if self.use_extra_layer:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim),
                nn.LogSoftmax(1),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim),
                nn.LogSoftmax(1),
            )

    def forward(self, state, desired_return, desired_horizon):
        """Return log-probabilities of actions."""
        c = th.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c * self.scaling_factor
        s = self.s_emb(state.float())
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        log_prob = self.fc(s * c)
        return log_prob


class AdaptedPCN(PCN):
    def __init__(
            self,
            env: Optional[gym.Env],
            scaling_factor: np.ndarray,
            learning_rate: float = 1e-3,
            gamma: float = 1.0,
            batch_size: int = 256,
            hidden_dim: int = 64,
            use_extra_layer: bool = False,
            save_pf_every_n: int = 10,
            project_name: str = "MORL-Baselines",
            experiment_name: str = "PCN",
            wandb_entity: Optional[str] = None,
            log: bool = True,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__(env, scaling_factor, learning_rate, gamma, batch_size, hidden_dim, project_name,
                         experiment_name, wandb_entity, log, seed, device)

        self.model = AdaptedPCNModel(
            self.observation_dim, self.action_dim, self.reward_dim, self.scaling_factor, hidden_dim=self.hidden_dim,
            use_extra_layer=use_extra_layer).to(self.device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.save_pf_every_n = save_pf_every_n
        self.eval_cnt = -1

    def evaluate(self, env, max_return, n=10):
        """Evaluate policy in the given environment."""
        self.eval_cnt += 1
        n = min(n, len(self.experience_replay))
        episodes = self._nlargest(n)
        returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
        returns = np.float32(returns)
        horizons = np.float32(horizons)
        e_returns = []
        for i in range(n):
            transitions = self._run_episode(env, returns[i], np.float32(horizons[i] - 2), max_return, eval_mode=True)
            # compute return
            for i in reversed(range(len(transitions) - 1)):
                transitions[i].reward += self.gamma * transitions[i + 1].reward
            e_returns.append(transitions[0].reward)
        distances = np.linalg.norm(np.array(returns) - np.array(e_returns), axis=-1)
        if self.eval_cnt % self.save_pf_every_n != 0:
            return e_returns, np.array(returns), distances

        col = [f"objective_{i}" for i in range(self.reward_dim)] \
              + [f"used_return_{i}" for i in range(self.reward_dim)] + ["used_horizon"]
        used_horizon = np.array(horizons, dtype=np.float32) - 2
        data = np.concatenate([np.array(e_returns), np.array(returns), used_horizon[:, None]], axis=-1)
        front = wandb.Table(
            columns=col,
            data=data.tolist(),
        )

        scatters = {
            f"PF step={self.global_step} obj_{ri} vs obj_{rj}": wandb.plot.scatter(front, f"objective_{ri}", f"objective_{rj}",
                                                                                   title=f"step {self.global_step} obj_{ri} vs obj_{rj}") \
            for ri in range(self.reward_dim) for rj in range(ri + 1, self.reward_dim)
        }
        suffix = f"{self.eval_cnt:04d}_{self.global_step}"
        wandb.log({f"eval/front_{suffix}": front, **scatters})

        out_dir = HydraConfig.get()['runtime']['output_dir']
        checkpoint_dir = Path(out_dir) / "checkpoints"
        self.save(f"checkpoint_{suffix}", str(checkpoint_dir))

        return e_returns, np.array(returns), distances


@hydra.main(version_base=None, config_path="configs", config_name="train_pcn")
def main(config: DictConfig):
    def make_env():
        env = mo_gym.make(config.env.name)
        env = MORecordEpisodeStatistics(env, gamma=config.gamma)
        return env

    env = make_env()

    # A patch to avoid passing these parameters through runai submit (which doesn't handle doublequotes well)
    if config.env.name == 'mo-reacher-v4':
        print('Hard-coding the reacher scaling factor and max return!', flush=True)
        config.scaling_factor = [0.1, 0.1, 0.1, 0.1, 0.1]
        config.max_return = [50, 50, 50, 50]
    elif config.env.name == 'mo-lunar-lander-v2':
        config.scaling_factor = [0.1, 0.1, 0.1, 0.1, 0.1]
        config.max_return = [100, 300, 0, 0]
    else:
        print(f"Using scaling from config. {config.env.name=}")


    agent = AdaptedPCN(
        env,
        scaling_factor=np.array(config.scaling_factor),
        learning_rate=config.lr,
        gamma=config.gamma,
        batch_size=config.batch_size,
        hidden_dim=config.hidden_dim,
        use_extra_layer=config.use_extra_layer,
        save_pf_every_n=config.save_pf_every_n,
        project_name=config.wandb.project,
        experiment_name=config.wandb.name,
        wandb_entity="claire-labo",
        log=True,
        seed=config.seed,
    )

    true_pf = None if not hasattr(env.unwrapped, "pareto_front") else env.unwrapped.pareto_front(gamma=config.gamma)
    agent.train(
        eval_env=make_env(),
        total_timesteps=int(config.num_samples),
        ref_point=np.array(config.env.hypervolume_reference),
        num_er_episodes=config.num_er_episodes,
        max_buffer_size=config.max_buffer_size,
        num_model_updates=config.num_model_updates,
        max_return=np.array(config.max_return),
        known_pareto_front=true_pf,
    )


if __name__ == "__main__":
    main()
