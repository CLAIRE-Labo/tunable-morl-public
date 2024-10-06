import sys
import logging
from pathlib import Path
import itertools

import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
# import gymnasium as gym
# import mo_gymnasium as mo_gym

import torch

from policy import MultiRewardPolicy, wrap_policy_into_actor
from training import gamma_returns
from eval_utils import eval_pareto
from env import create_env
from utils.seeding import seed_everything

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="load_moe")
def main(config: DictConfig) -> None:
    logger.info(config)
    wandb.init(
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        project=config.wandb.project,
        anonymous=config.wandb.anonymous,
        mode=config.wandb.mode,
        dir=Path(config.wandb.dir).absolute(),
    )
    seed_everything(config)

    load_config = OmegaConf.load(config.load_config)
    env = create_env('mo-gym', device=config.device, **load_config.env)
    policy = MultiRewardPolicy.load_policy_from_checkpoint(config.load_checkpoint, config.device)
    actor = wrap_policy_into_actor(policy)

    rew = eval_pareto_2d(actor, env, load_config, config.eval.num_final_pareto_points)
    tab = wandb.Table(data=rew, columns=["objective_1", "objective_2"])
    wandb.log({"pareto": wandb.plot.scatter(tab, "objective_1", "objective_2")})


if __name__ == "__main__":
    main()
