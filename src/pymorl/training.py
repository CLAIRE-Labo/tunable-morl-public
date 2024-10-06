from __future__ import annotations

from typing import Optional, Union, List, Callable
from pathlib import Path
import logging
from copy import deepcopy

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import tqdm
import wandb
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

from torchrl.envs import EnvBase, ParallelEnv, SerialEnv
from torchrl.envs.transforms import TransformedEnv, StepCounter
from torchrl.collectors import SyncDataCollector
from torchrl.modules import ProbabilisticActor

from env import RewardImportanceTransform
from utils.drawing import add_text
from entropy import EntropyController
from policy import MultiRewardPolicy, wrap_policy_into_actor
from advantage import compute_advantage
from eval_utils import eval_pareto, eval_fixed_head_rewards, gamma_returns, average_num_steps, average_full_return
from pareto_front import ParetoFront, pf_metrics
from sampling import uniform_simplex, generate_fixed_importance
from timing import TrainingTimer

logger = logging.getLogger(__name__)


# def create_loss(actor: ProbabilisticTensorDictSequential, critic: TensorDictModule, config: DictConfig):
#     if config.name == 'ppo':
#         loss_module = PPOLoss(actor=actor, critic=critic, entropy_coef=config.entropy_coef)
#         loss_module.set_keys(value=f"values_{reward_ind}")
#     else:
#         raise NotImplementedError(f"Unknown training name: {trc.name}")


# This class tries to adjust beta so that loss \approx reward_loss * beta
class BetaAdjuster:
    def __init__(self, init_beta: float, dynamic_beta: bool, dynamic_beta_lr: float,
                 target_actor_to_critic: float = 1.0):
        self.beta = init_beta
        self.dynamic_beta = dynamic_beta
        self.dynamic_beta_lr = dynamic_beta_lr
        self.target_actor_to_critic = target_actor_to_critic

    def adjust(self, grad_norm_policy: float, grad_norm_reward: float):
        if not self.dynamic_beta:
            return self.beta
        if np.abs(grad_norm_reward) < 1e-8:
            logger.warning(f"grad_norm_reward is too small: {grad_norm_reward}")
            return self.beta
        # We are striving that grad_norm_policy \approx grad_norm_reward * beta * target_actor_to_critic
        self.beta = self.dynamic_beta_lr * grad_norm_policy / (grad_norm_reward * self.target_actor_to_critic) + (
                1 - self.dynamic_beta_lr) * self.beta
        return self.beta


def _track_discarded(check_func):
    def wrap(self, *args, **kwargs):
        cur_is_discarded = check_func(self, *args, **kwargs)
        self.is_discarded.append(int(cur_is_discarded))
        return cur_is_discarded

    return wrap


class CheckpointSaver:
    def __init__(self, checkpoint_path: Path, checkpoint_format: str, checkpoint_every_n_steps: int,
                 policy: MultiRewardPolicy, optimizer: torch.optim.Optimizer, value_optimizer: Optional[torch.optim],
                 timer: TrainingTimer):
        Path(checkpoint_path).mkdir()
        self.checkpoint_format = str(checkpoint_path) + "/" + checkpoint_format
        logger.info(f"Saving checkpoints with format str {self.checkpoint_format}")

        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.checkpoint_iters = []
        self.policy = policy
        self.optimizer = optimizer
        self.value_optimizer = value_optimizer
        self.timer = timer
        self.cur_checkpoint = self.get_cur_checkpoint()
        self.last_checkpoint = self.get_cur_checkpoint()

    def maybe_save_checkpoint(self, i: int):
        if i == 0 or i % self.checkpoint_every_n_steps == self.checkpoint_every_n_steps - 1:
            with self.timer.eval_iter("checkpoint"):
                checkpoint = {
                    'iter': i,
                    'policy_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                if self.value_optimizer is not None:
                    checkpoint['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
                torch.save(checkpoint, self.checkpoint_format.format(i))
                logger.info(f"Saved checkpoint {i}")
                self.checkpoint_iters.append(i)

    def update_cur_checkpoint(self):
        self.last_checkpoint = self.cur_checkpoint
        self.cur_checkpoint = self.get_cur_checkpoint()

    def get_cur_checkpoint(self) -> dict:
        checkpoint = {
            'policy_state_dict': deepcopy(self.policy.state_dict()),
            'optimizer_state_dict': deepcopy(self.optimizer.state_dict()),
        }
        if self.value_optimizer is not None:
            checkpoint['value_optimizer_state_dict'] = deepcopy(self.value_optimizer.state_dict())
        return checkpoint

    def reload(self, checkpoint: dict):
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.value_optimizer is not None:
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

    def recover_from_checkpoint(self, recover_iter: int):
        assert recover_iter in self.checkpoint_iters, f"Checkpoint {recover_iter} not found in {self.checkpoint_iters}"
        checkpoint_file = self.checkpoint_format.format(recover_iter)
        logger.info(f"Reloading from checkpoint {checkpoint_file}")
        if not Path(checkpoint_file).exists():
            logger.warning("\n\n\nNo checkpoint to reload from!!\n\n\n")
        else:
            checkpoint = torch.load(checkpoint_file)
            self.reload(checkpoint)


class StepDiscarder:
    def __init__(self, policy: MultiRewardPolicy, optimizer: torch.optim.Optimizer,
                 value_optimizer: Optional[torch.optim.Optimizer], checkpoint_saver: CheckpointSaver,
                 discarder_config: DictConfig):
        self.policy = policy
        self.optimizer = optimizer
        self.value_optimizer = value_optimizer
        self.checkpoint_saver = checkpoint_saver
        self.config = discarder_config
        self.max_discarded_in_window = self.config.window_size * self.config.max_discarded_frac_in_window
        self.cur_iter = 0

        self.actor_grad_norms = []
        self.avg_sum_rewards = []
        self.entropies = []
        self.is_discarded = []

    @_track_discarded
    def check_step_and_maybe_reset(self, cur_entropy: float, cur_rewards: List[torch.Tensor],
                                   cur_actor_grad_norm) -> bool:
        self.cur_iter += 1
        new_avg_sum_rewards = np.mean([torch.sum(r).item() for r in cur_rewards])

        for param in self.policy.parameters():
            if (~torch.isfinite(param)).any().item():
                logger.warning("Non-finite numbers in the policy parameters, reverting last step")
                return True

        def update_stats():
            self.actor_grad_norms.append(cur_actor_grad_norm)
            self.avg_sum_rewards.append(new_avg_sum_rewards)
            self.entropies.append(cur_entropy)

        if len(self.avg_sum_rewards) < self.config.num_burnin_steps:
            update_stats()
            return False

        if np.all(self.is_discarded[-self.config.force_step_after_n_fails:]):
            logger.warning("Too many consecutive discarded steps, forcing a step")
            update_stats()
            return False

        if np.sum(self.is_discarded[-self.config.window_size:]) >= self.max_discarded_in_window:
            # logger.warning("Too many discarded steps in the window, forcing a step")
            update_stats()
            return False

        last_norms = self.actor_grad_norms[-self.config.num_small_actor_grads:]
        last_entropies = self.entropies[-self.config.num_small_actor_grads:]
        are_last_norms_small = len(self.actor_grad_norms) > self.config.num_small_actor_grads \
                               and np.all(np.array(last_norms) < self.config.small_actor_grad_norm)
        are_last_entropies_small = len(self.entropies) > self.config.num_small_actor_grads \
                                   and np.all(np.array(last_entropies) < self.config.small_entropy)
        # If the last N actor grad norms are small, the training has likely collapsed. Reset to the last checkpoint
        if are_last_norms_small or are_last_entropies_small:
            logger.warning(
                f"Resetting from a checkpoint because of small critic grad norm or small entropies: "
                f"{np.mean(last_norms)=}, {np.mean(last_entropies)=}, "
                f"{are_last_norms_small=}, {are_last_entropies_small=}")
            # Reload from the last checkpoint that is before the small-gradient iterations
            assert len(self.checkpoint_saver.checkpoint_iters) > 0, "No checkpoints saved"
            if self.checkpoint_saver.checkpoint_iters[0] >= self.cur_iter - self.config.num_small_actor_grads:
                recover_iter = self.checkpoint_saver.checkpoint_iters[0]
            else:
                good_iters = [i for i in self.checkpoint_saver.checkpoint_iters \
                              if i < self.cur_iter - self.config.num_small_actor_grads]
                assert len(good_iters) > 0
                recover_iter = good_iters[-1]
            logger.info(f"Recovering from checkpoint #{recover_iter}")
            self.checkpoint_saver.recover_from_checkpoint(recover_iter)
            return True

        # If the current reward is much smaller than the statistics predict, discard the step
        # Disable this for now. The agent must learn from strong negative rewards.
        # running_mean = np.mean(self.avg_sum_rewards)
        # running_std = np.std(self.avg_sum_rewards)
        # if new_avg_sum_rewards < running_mean - self.config.bad_sum_reward_factor * running_std:
        #     logger.warning(
        #         f"Suspiciously bad sum of rewards ({new_avg_sum_rewards:.2f} < "
        #         f"{running_mean:.2f} - {self.config.bad_sum_reward_factor:.2f}*{running_std:.2f}), reverting last step")
        #
        #     update_stats()  # We still save the stats. If the low rewards are the new standards, discarder has to adapt
        #     self.checkpoint_saver.reload(self.checkpoint_saver.last_checkpoint)
        #     return True

        # If the entropy decreased significantly but the reward also decreased, discard the step
        entropy_diffs = np.abs(np.diff(self.entropies))
        avg_entropy_diff = np.mean(entropy_diffs)
        std_entropy_diff = np.std(entropy_diffs)
        cur_diff = cur_entropy - self.entropies[-1]
        if new_avg_sum_rewards < self.avg_sum_rewards[-1] and cur_diff < 0 \
                and -cur_diff > avg_entropy_diff + self.config.bad_entropy_factor * std_entropy_diff:
            logger.warning(f"Suspiciously bad entropy (step={-cur_diff:.4e} < {avg_entropy_diff:.4e} + "
                           f"{self.config.bad_entropy_factor}*{std_entropy_diff:.4e}), reverting last step")
            self.checkpoint_saver.reload(self.checkpoint_saver.last_checkpoint)
            return True

        update_stats()
        return False


def compute_grad_norms_and_grads(optimizer: torch.optim.Optimizer, model: nn.Module, loss1: torch.Tensor,
                                 loss2: torch.Tensor, beta_adjuster: BetaAdjuster) \
        -> (float, float, float):
    """
    Compute the gradient norms for two losses and make a step with the combined gradients.
    This is used to dynamically adjust the relative weight of the two losses.
    """
    # Adapted from ChatGPT
    # Backward for l1, retain graph for l2 backward pass
    if torch.isnan(loss1).any():
        logger.warning("NaNs in the loss1")

    loss1.backward(retain_graph=True)
    grads_loss1 = {name: (param.grad.clone() if param.grad is not None else None) \
                   for name, param in model.named_parameters()}

    for name, grad in grads_loss1.items():
        if grad is not None and torch.isnan(grad).any():
            logger.warning(f"NaNs in the gradient of {name} after the first backward pass")

    optimizer.zero_grad()

    if torch.isnan(loss2).any():
        logger.warning("NaNs in the loss2")

    loss2.backward()
    grads_loss2 = {name: (param.grad.clone() if param.grad is not None else None) \
                   for name, param in model.named_parameters()}

    for name, grad in grads_loss2.items():
        if grad is not None and torch.isnan(grad).any():
            logger.warning(f"NaNs in the gradient of {name} after the second backward pass")

    norm_loss1 = torch.sqrt(sum(torch.sum(grad ** 2) for grad in grads_loss1.values() if grad is not None)).item()
    norm_loss2 = torch.sqrt(sum(torch.sum(grad ** 2) for grad in grads_loss2.values() if grad is not None)).item()
    if norm_loss1 < 1e-8 or norm_loss2 < 1e-8:
        logger.warning(f"norm_loss1 is too small: {norm_loss1=}, {norm_loss2=}")
    # else:
    #     logger.info(f'{norm_loss1=} {norm_loss2=}')

    new_beta = beta_adjuster.adjust(norm_loss1, norm_loss2)

    # Compute gradients of l1 + beta * l2
    for name, param in model.named_parameters():
        param.grad = torch.zeros_like(param.data) if grads_loss1[name] is None else grads_loss1[name]
        if grads_loss2[name] is not None:
            param.grad += new_beta * grads_loss2[name]

    return norm_loss1, norm_loss2, new_beta


def joint_actor_critic_step_and_log(loss: torch.Tensor, reward_loss: torch.Tensor, beta_adjuster: BetaAdjuster,
                                    policy: MultiRewardPolicy, optimizer: torch.optim.Optimizer,
                                    max_grad_norm: float) -> dict:
    policy_grad_norm, value_grad_norm, beta \
        = compute_grad_norms_and_grads(optimizer, policy, loss, reward_loss, beta_adjuster)
    grad_norm = np.hypot(policy_grad_norm, beta * value_grad_norm)
    grad_norm_clipped = min(grad_norm, max_grad_norm)
    relative_norm = 1000 if np.abs(beta * value_grad_norm) < 1e-8 \
        else policy_grad_norm / (beta * value_grad_norm)

    grad_norm_new = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm).item()
    # assert np.abs(grad_norm_new - grad_norm) < 1e-7, f"{grad_norm_new=}, {grad_norm=}"
    optimizer.step()
    optimizer.zero_grad()

    for param in policy.parameters():
        if torch.isnan(param).any():
            logger.warning("NaNs in the parameters post-step")

    return {"loss/reward_loss": reward_loss.item(),
            "normalization/value_grad_norm": value_grad_norm,
            "normalization/policy_grad_norm": policy_grad_norm,
            "normalization/relative_norm": relative_norm,
            "normalization/beta": beta}


def separate_critic_step_and_log(reward_loss: torch.Tensor, policy: MultiRewardPolicy,
                                 value_optimizer: torch.optim.Optimizer, max_grad_norm: float) -> dict:
    if reward_loss.isnan().any():
        logger.warning("NaNs in the reward loss")

    reward_loss.backward()

    for param in policy.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            logger.warning("NaNs in the gradient")

    value_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)

    for param in policy.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            logger.warning("NaNs in the gradient post-clip")

    value_grad_norm_clipped = min(value_grad_norm, max_grad_norm)
    value_optimizer.step()
    value_optimizer.zero_grad()

    for param in policy.parameters():
        if torch.isnan(param).any():
            logger.warning("NaNs in the parameters post-step")

    return {"loss/reward_loss": reward_loss.item(),
            "normalization/value_grad_norm": value_grad_norm,
            "normalization/value_grad_norm_clipped": value_grad_norm_clipped}


def actor_step_and_log(loss: torch.Tensor, policy: MultiRewardPolicy, optimizer: torch.optim.Optimizer,
                       max_grad_norm: float) -> dict:
    if torch.isnan(loss).any():
        logger.warning("NaNs in the loss")

    loss.backward()

    for param in policy.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            logger.warning("NaNs in the gradient")

    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm).item()

    for param in policy.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            logger.warning("NaNs in the gradient post-clip")

    grad_norm_clipped = min(grad_norm, max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    for param in policy.parameters():
        if torch.isnan(param).any():
            logger.warning("NaNs in the parameters post-step")

    return {"loss/actor_loss": loss.item(),
            "normalization/policy_grad_norm": grad_norm,
            "normalization/policy_grad_norm_clipped": grad_norm_clipped}


def update_popart_and_log(value_target: torch.Tensor, policy: MultiRewardPolicy) -> dict:
    norm_value_target = policy.update_popart(value_target)

    avg_norm_rewards = norm_value_target.mean(dim=0)
    avg_norm_rewards_dict = {f"normalization/avg_norm_value_target_{i}": avg_norm_rewards[i].item()
                             for i in range(avg_norm_rewards.shape[0])}
    sigmas = policy.value_head.sigmas
    means = policy.value_head.means
    sigma_dict = {f"popart/sigma_{i}": sigmas[i].item() for i in range(sigmas.shape[0])}
    mean_dict = {f"popart/mean_{i}": means[i].item() for i in range(means.shape[0])}
    metrics = {**avg_norm_rewards_dict, **sigma_dict, **mean_dict}
    return metrics


def wandb_pareto_items(pixels_env: EnvBase, actor: ProbabilisticActor, eval_alphas: np.ndarray, hv_ref: np.ndarray,
                       true_pf: ParetoFront, reward_importance: np.ndarray, i: int, config: DictConfig) -> dict:
    num_rewards = pixels_env.reward_spec.shape[0]

    def _get_items(use_mode: bool = False) -> dict:
        suf = "_mode" if use_mode else ""
        rew = eval_pareto(actor, pixels_env, config, eval_alphas, use_mode=use_mode)
        metrics = pf_metrics(rew, true_pf, hv_ref, reward_importance)
        pref_metrics = {f"pf_metrics{suf}/{k}": v for k, v in metrics.items()}
        tab = wandb.Table(data=rew, columns=[f"objective_{ri}" for ri in range(num_rewards)])
        scatters = {
            f"PF{suf} step={i} obj_{ri} vs obj_{rj}": \
                wandb.plot.scatter(tab, f"objective_{ri}", f"objective_{rj}", \
                                   title=f"PF{suf} step {i} obj_{ri} vs obj_{rj}") \
            for ri in range(num_rewards) for rj in range(ri + 1, num_rewards)
        }
        return {f"PF{suf}_step_{i}": tab, **scatters, **pref_metrics}

    items_mode = _get_items(True)
    items_rand = _get_items(False)
    return {**items_rand, **items_mode}


def set_lr(optimizer: torch.optim.Optimizer, value_optimizer: Optional[torch.optim.Optimizer], lr: float, vlr):
    for g in optimizer.param_groups:
        g['lr'] = lr
    if value_optimizer is not None:
        for g in value_optimizer.param_groups:
            g['lr'] = vlr


def generate_video(rollouts, text_color: str):
    vids = []
    for data in rollouts:
        vid = data["pixels"].cpu().numpy()
        text = f"importance: [{', '.join([f'{a.item():.2f}' for a in data['reward_importance'][0, :]])}]"
        vid = add_text(vid, text, color=text_color)
        # vid = np.concatenate([vid, np.zeros_like(vid)], axis=3)
        vid = vid.transpose(0, 3, 1, 2)
        vids.append(vid)
    video = np.concatenate(vids, axis=0)
    return video
