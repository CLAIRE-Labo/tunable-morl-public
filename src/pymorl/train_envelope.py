# Script adapted from
# https://github.com/LucasAlegre/morl-baselines/blob/main/examples/envelope_minecart.py

import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.envelope.envelope import Envelope


@hydra.main(version_base=None, config_path="configs", config_name="train_envelope")
def main(config: DictConfig):
    def make_env():
        env = mo_gym.make(config.env.name)
        env = MORecordEpisodeStatistics(env, gamma=config.gamma)
        # env = mo_gym.LinearReward(env)
        return env

    env = make_env()
    eval_env = make_env()
    # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    true_pf = None if not hasattr(env.unwrapped, "pareto_front") else env.unwrapped.pareto_front(gamma=config.gamma)

    epsilon_decay_steps = config.num_samples // 2
    homotopy_decay_steps = config.num_samples // 4

    agent = Envelope(
        env,
        max_grad_norm=0.1,
        learning_rate=config.lr,
        gamma=config.gamma,
        batch_size=64,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(2e6),
        initial_epsilon=1.0,
        final_epsilon=config.final_epsilon,
        epsilon_decay_steps=epsilon_decay_steps,
        initial_homotopy_lambda=0.0,
        final_homotopy_lambda=1.0,
        homotopy_decay_steps=homotopy_decay_steps,
        learning_starts=100,
        envelope=True,
        gradient_updates=1,
        target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
        tau=1,
        log=True,
        seed=config.seed,
        project_name=config.wandb.project,
        experiment_name=config.wandb.name,
        wandb_entity="claire-labo",
    )

    agent.train(
        total_timesteps=config.num_samples,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        ref_point=np.array(config.env.hypervolume_reference),
        known_pareto_front=true_pf,
        num_eval_weights_for_front=config.num_eval_points,
        eval_freq=config.eval_freq,
        reset_num_timesteps=False,
        reset_learning_starts=False,
    )


if __name__ == "__main__":
    main()
    