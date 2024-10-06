import os
import logging
from pathlib import Path
import itertools

import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
import torch
# import gymnasium as gym
# import mo_gymnasium as mo_gym

from torchrl.envs import EnvBase

from pareto_front import ParetoFront, pf_metrics
from policy import create_policy
from training import eval_pareto, generate_fixed_importance
from ppo import train_ppo
from chebyshev_ppo import train_chebyshev_ppo
from env import create_env
from utils.seeding import seed_everything
from sampling import stratified_uniform_simplex

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train_moe")
def main(config: DictConfig) -> None:
    logger.info(config)

    job_name = config.wandb.get("name", None)

    if config.debug:
        torch.autograd.set_detect_anomaly(True)
        torch._logging.set_logs(dynamo=logging.DEBUG)
        torch._dynamo.config.verbose = True
        logger.info("Enabled debug mode!")

    wandb.init(
        name=job_name,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        project=config.wandb.project,
        tags=["barebones", "reinforce"],
        anonymous=config.wandb.anonymous,
        mode=config.wandb.mode,
        dir=Path(config.wandb.dir).absolute(),
    )
    seed_everything(config)

    def create_env_fn() -> EnvBase:
        return create_env('mo-gym', device=config.device, return_pixels_env=False, **config.env)

    env, pixels_env = create_env('mo-gym', device=config.device, return_pixels_env=True, **config.env)
    num_rewards = env.reward_spec.shape[0]
    policy, actor = create_policy(config, env)

    final_eval_importance = generate_fixed_importance(config.eval.num_pareto_points, num_rewards)
    wandb.log({'Final/importances': wandb.Table(columns=[f"objective_{ri}" for ri in range(num_rewards)],
                                                data=final_eval_importance)})
    hv_ref = np.array(config.env.hypervolume_reference)
    if hasattr(env.unwrapped, "pareto_front"):
        true_pf = ParetoFront(env.unwrapped.pareto_front(gamma=config.training.advantage.gamma))
        true_pf_metrics = pf_metrics(true_pf.filtered_points, true_pf, hv_ref, final_eval_importance)
    else:
        true_pf = None
    metrics = {f"true_pf/{k}": v for k, v in true_pf_metrics.items()} if true_pf is not None else {}
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    if config.training.rl_method in ['ppo', 'a2c']:
        train_ppo(policy, create_env_fn, name='random_mix', config=config, alpha_selection='random_mix',
                  pixels_env=pixels_env)
    elif config.training.rl_method == 'chebyshev_ppo':
        train_chebyshev_ppo(policy, create_env_fn, name='random_mix', config=config, alpha_selection='random_mix',
                            pixels_env=pixels_env)
    else:
        raise NotImplementedError(f"Unknown RL method {config.training.rl_method}")

    eval_mixes = generate_fixed_importance(config.eval.num_final_pareto_points, num_rewards)
    logger.info(f"final eval_mixes:\n{eval_mixes}\n")
    rew = eval_pareto(actor, pixels_env, config, eval_mixes)
    tab = wandb.Table(data=rew, columns=[f"objective_{ri}" for ri in range(num_rewards)])
    scatters = {
        f"Final PF obj_{ri} vs obj_{rj}": wandb.plot.scatter(tab, f"objective_{ri}", f"objective_{rj}",
                                                             title=f"Final PF obj_{ri} vs obj_{rj}") \
        for ri in range(num_rewards) for rj in range(ri + 1, num_rewards)
    }
    wandb.log({f"Final/PF": tab, **scatters})

    final_metrics = pf_metrics(rew, true_pf, hv_ref, final_eval_importance)
    for k, v in final_metrics.items():
        metrics[f"final_pf/{k}"] = v
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
    wandb.log({"Final/metrics": wandb.Table(columns=list(metrics.keys()), data=[list(metrics.values())])})


if __name__ == "__main__":
    main()
