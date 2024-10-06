from __future__ import annotations

import sys
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

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential
from tensordict.nn.common import dispatch
from tensordict.nn.probabilistic import InteractionType
from torchrl.objectives import ClipPPOLoss, distance_loss
from torchrl.objectives.utils import ValueEstimators
from torchrl.envs import EnvBase, ParallelEnv, SerialEnv
from torchrl.envs.transforms import TransformedEnv, StepCounter
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.modules import ProbabilisticActor

from utils.drawing import add_text
from entropy import EntropyController
from policy import MultiRewardPolicy, wrap_policy_into_actor
from advantage import compute_advantage
from eval_utils import eval_pareto, eval_fixed_head_rewards, gamma_returns, average_num_steps, average_full_return
from pareto_front import ParetoFront
from sampling import uniform_simplex, generate_fixed_importance
from timing import TrainingTimer
from training import BetaAdjuster, CheckpointSaver, StepDiscarder, joint_actor_critic_step_and_log, \
    separate_critic_step_and_log, actor_step_and_log, update_popart_and_log, set_lr, generate_video, wandb_pareto_items
from advantage import create_advantage_estimator
from env import RewardImportanceTransform

logger = logging.getLogger(__name__)


def loss_critic_from_td(tensordict: TensorDictBase, with_popart: bool, loss_type: str, coef: float) -> torch.Tensor:
    if with_popart:
        target_return = tensordict["popart_normalized_value_target"]
        state_value = tensordict["popart_normalized_predicted_value"]
    else:
        target_return = tensordict["value_target"]
        state_value = tensordict["predicted_value"]

    loss_value = distance_loss(
        target_return,
        state_value,
        loss_function=loss_type,
    ).mean()
    return coef * loss_value


# Adapted from the original implementation in torchrl to handle multiple objectives & PopArt
class MultiRewardClipPPOLoss(ClipPPOLoss):
    def __init__(self, policy: MultiRewardPolicy, actor: ProbabilisticTensorDictSequential, critic: TensorDictModule,
                 training_cfg: DictConfig):
        super().__init__(actor=actor, critic=critic, clip_epsilon=training_cfg.ppo.clip_epsilon,
                         entropy_coef=training_cfg.entropy_coef, loss_critic_type="l2",
                         normalize_advantage=training_cfg.advantage.post_scalarization_normalized,
                         gamma=training_cfg.advantage.gamma, critic_coef=1.0)
        self.policy = policy
        self.my_critic = critic

    def loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        self.my_critic(tensordict)
        return loss_critic_from_td(tensordict, self.policy.does_use_popart, self.loss_critic_type, self.critic_coef)

    @dispatch
    def forward(self, tensordict: TensorDictBase, compute_critic_loss: bool = True) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get("scalarized_advantage")

        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean().item()
            scale = advantage.std().clamp_min(1e-6).item()
            advantage = (advantage - loc) / scale

        log_weight, dist = self._log_weight(tensordict)
        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same source. Here we sample according
            # to different, unrelated trajectories, which is not standard. Still it can give a idea of the dispersion
            # of the weights.
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        gain1 = log_weight.exp() * advantage

        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        gain2 = log_weight_clip.exp() * advantage

        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
        td_out = TensorDict({"loss_objective": -gain.mean()}, [])

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean())
            # Now done by the entropy controller
            # td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())
        if compute_critic_loss and self.critic_coef:
            loss_critic = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic.mean())
        td_out.set("ESS", ess.mean() / batch)
        return td_out


class MultiRewardA2CLoss(nn.Module):
    def __init__(self, policy: MultiRewardPolicy, actor: ProbabilisticTensorDictSequential, critic: TensorDictModule,
                 training_cfg: DictConfig):
        super().__init__()
        self.policy = policy
        self.my_actor = actor
        self.my_critic = critic
        self.cfg = training_cfg

    def loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        self.my_critic(tensordict)
        return loss_critic_from_td(tensordict, self.policy.does_use_popart, "l2", 1.0)

    def forward(self, tensordict: TensorDictBase, compute_critic_loss: bool = True) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get("scalarized_advantage")
        if self.cfg.advantage.post_scalarization_normalized and advantage.numel() > 1:
            loc = advantage.mean().item()
            scale = advantage.std().clamp_min(1e-6).item()
            advantage = (advantage - loc) / scale
        assert advantage.shape[1] == 1, f"Expected shape (len, 1), got {advantage.shape}"

        dist = self.my_actor.get_dist(tensordict)
        act_logprob = dist.log_prob(tensordict["action"])
        loss = -(act_logprob * advantage[:, 0]).mean()
        td_out = TensorDict({"loss_objective": loss}, [])

        td_out.set("entropy", dist.entropy().mean())
        if compute_critic_loss:
            td_out.set("loss_critic", self.loss_critic(tensordict))
        return td_out


def train_ppo(policy: MultiRewardPolicy, create_env_fn: Callable[[], EnvBase], name: str, config: DictConfig,
              alpha_selection: str = 'random_mix', pixels_env: Optional[EnvBase] = None):
    assert alpha_selection == 'random_mix', "Only random alpha selection is supported for now, got " \
                                            f"{alpha_selection} instead"
    timer = TrainingTimer(config)
    env = create_env_fn()

    num_recorded_videos = 0
    num_rewards = env.reward_spec.shape[0]
    num_actions = env.action_spec.shape[0]
    eval_importance = generate_fixed_importance(config.eval.num_pareto_points, num_rewards)
    eval_importance = eval_importance[np.argsort(eval_importance[:, 0])]
    eval_importance_th = torch.from_numpy(eval_importance).to(env.device)
    logger.info(f"eval_importance:\n{eval_importance}\n")
    logger.info(f"sorted_0:\n{np.sort(eval_importance[:, 0])}\n")
    tcfg = config.training
    value_cfg = config.training.value_loss
    advantage_cfg = config.training.advantage
    if advantage_cfg.popart_normalized:
        assert policy.does_use_popart, "Popart advantage normalization is enabled but the policy does have a popart head"
    assert tcfg.num_unrolls > 0
    assert advantage_cfg.estimator != "td1_no_boostrapping", "PPO needs a critic!"
    assert config.model.separate_value_network == (not config.training.value_loss.optimize_jointly), \
        "For now this confusing and unnecessary combination is disabled!"
    assert tcfg.timer.allocated_time is None, "For now I disabled explicit time limits, sorry :("

    def create_env_with_weights():
        return TransformedEnv(create_env_fn(), RewardImportanceTransform(num_rewards))

    # parallel_env = ParallelEnv(tcfg.num_unrolls, lambda: env)
    parallel_env = SerialEnv(tcfg.num_unrolls, create_env_with_weights)

    # TODO check -- could be related to small entropies?
    cur_lr = tcfg.optim.lr
    cur_value_lr = tcfg.optim.value_lr
    optimizer = torch.optim.Adam(policy.parameters(), lr=cur_lr)
    value_optimizer = torch.optim.Adam(policy.parameters(), lr=tcfg.optim.value_lr, weight_decay=value_cfg.weight_decay) \
        if not value_cfg.optimize_jointly else None
    # TODO undo this, just for debugging
    # value_optimizer = torch.optim.SGD(policy.parameters(), lr=tcfg.optim.value_lr) \
    #     if not value_cfg.optimize_jointly else None

    actor = wrap_policy_into_actor(policy)
    critic = policy.create_critic()
    if tcfg.rl_method == 'ppo':
        loss = MultiRewardClipPPOLoss(policy=policy, actor=actor, critic=critic, training_cfg=tcfg)
        loss.set_keys(value="predicted_value")
    else:
        loss = MultiRewardA2CLoss(policy=policy, actor=actor, critic=critic, training_cfg=tcfg)

    out_dir = HydraConfig.get()['runtime']['output_dir']

    hv_ref = np.array(config.env.hypervolume_reference)
    assert hv_ref.shape == (num_rewards,), f"Expected shape {(num_rewards,)}, got {hv_ref.shape}"
    true_pf = ParetoFront(env.unwrapped.pareto_front(gamma=advantage_cfg.gamma)) \
        if hasattr(env.unwrapped, "pareto_front") else None

    # this controls the weight of the reward prediction loss (only if it is optimized jointly with the policy loss)
    beta_adjuster = BetaAdjuster(init_beta=value_cfg.init_beta, dynamic_beta=value_cfg.dynamic_beta,
                                 dynamic_beta_lr=value_cfg.dynamic_beta_lr,
                                 target_actor_to_critic=value_cfg.target_actor_to_critic_ratio)
    checkpoint_saver = CheckpointSaver(Path(out_dir) / "checkpoints", "checkpoint_{}.pth",
                                       config.eval.checkpoint_every_n_steps, policy, optimizer, value_optimizer, timer)
    step_discarder = StepDiscarder(policy=policy, optimizer=optimizer, value_optimizer=value_optimizer,
                                   checkpoint_saver=checkpoint_saver, discarder_config=tcfg.step_discarder)
    num_discarded_steps = 0
    max_num_items = tcfg.num_samples if tcfg.num_samples is not None else tcfg.num_iter
    cur_num_items = 0
    entropy_controller = EntropyController(policy, tcfg.entropy_control, max_num_items)

    steps_per_sample = tcfg.num_unrolls * tcfg.unroll_length
    num_steps = np.ceil(tcfg.num_samples / steps_per_sample).astype(int) \
        if tcfg.num_samples is not None else tcfg.num_iter

    data_collector = SyncDataCollector(parallel_env, actor,
                                       frames_per_batch=tcfg.num_unrolls * tcfg.unroll_length,
                                       total_frames=tcfg.num_unrolls * tcfg.unroll_length * num_steps,
                                       max_frames_per_traj=config.env.max_episode_steps, device=actor.device)
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=tcfg.num_unrolls * tcfg.unroll_length, device=env.device),
        sampler=SamplerWithoutReplacement())

    advantage_module, scalarized_module \
        = create_advantage_estimator(advantage_cfg, policy, critic, "predicted_value")

    # TODO why doesn't critic generalize beteween epochs until very late in training?
    for i, data in enumerate(data_collector):
        cur_num_items += 1 if tcfg.num_samples is None else steps_per_sample
        with timer.train_iter():
            timer.add_samples(tcfg.unroll_length * tcfg.num_unrolls)

            # A hack to make the learner bootstrap on the final state if the episode did not terminate
            rewards = data["next", "reward"]
            avg_episode_len = average_num_steps(data["next", "done"])
            # PPO requires the original logprob
            # TODO add extra feature dim of 1 to sample_log_prob to coincide with the rest of the tensordict
            with torch.no_grad():
                sample_logprob = torch.masked_select(data["logprob"].reshape((-1, num_actions)),
                                                     data["action"].reshape((-1, num_actions)) == 1).reshape(
                    data.shape)
            data["sample_log_prob"] = sample_logprob

            data_flat = data
            del data  # to avoid bugs with using data in the future
            # A hack to make the adv estimator bootstrap on the last state if the episode did not terminate
            data_flat["next", "done"][:, -1, 0] = True

            # TODO REMOVE THIS. Only for debugging now!
            # This line makes the last state of a truncated trajectory to be considered as terminated
            # data_flat["next", "terminated"] = data_flat["next", "done"]

            fixed_done = torch.clone(data_flat["next", "done"])
            data_flat["next", "truncated"] = data_flat["next", "done"] & ~data_flat["next", "terminated"]

            data_flat = data_flat.reshape(-1)
            next_done_tiling = (1, num_rewards)
            # A hack to make advantage estimation work with multiple objectives
            data_flat["next", "done"] = torch.tile(data_flat["next", "done"], next_done_tiling)
            data_flat["next", "terminated"] = torch.tile(data_flat["next", "terminated"], next_done_tiling)
            data_flat["next", "truncated"] = torch.tile(data_flat["next", "truncated"], next_done_tiling)

            avg_return = average_full_return(rewards, fixed_done, advantage_cfg.gamma)
            train_rand_return_dict = {f"loss/rand_return_{i}": avg_return[i].item()
                                      for i in range(avg_return.shape[0])}

            if tcfg.rl_method == 'ppo':
                results = ppo_update_on_data(data_flat, loss, optimizer, value_optimizer, advantage_module,
                                             scalarized_module, beta_adjuster, entropy_controller, policy,
                                             replay_buffer, tcfg, cur_num_items)
            else:
                results = a2c_update_on_data(data_flat, loss, optimizer, value_optimizer, advantage_module,
                                             scalarized_module, beta_adjuster, entropy_controller, policy,
                                             tcfg, cur_num_items)

            cur_is_discarded = \
                step_discarder.check_step_and_maybe_reset(cur_entropy=results["avg_entropy"],
                                                          cur_rewards=[avg_return],
                                                          cur_actor_grad_norm=results["avg_policy_grad_norm"])

            if cur_is_discarded:
                # We have reset the policy. We will try to perform this step again with a smaller LR
                num_discarded_steps += 1
                cur_lr *= tcfg.optim.lr_mult_after_fail
                cur_value_lr *= tcfg.optim.lr_mult_after_fail
                set_lr(optimizer, value_optimizer, cur_lr, cur_value_lr)
            else:
                if cur_lr != tcfg.optim.lr:
                    cur_lr = tcfg.optim.lr
                    cur_value_lr = tcfg.optim.value_lr
                    set_lr(optimizer, value_optimizer, cur_lr, cur_value_lr)
                checkpoint_saver.update_cur_checkpoint()
            ws = config.training.step_discarder.window_size
            extra_metrics = {"discard/lr": cur_lr, "discard/num_discarded_steps": num_discarded_steps,
                             f"discard/discarded_among_last_{ws}": np.mean(step_discarder.is_discarded[-ws:])}
            logged_metrics = {**train_rand_return_dict, **extra_metrics}
        # stop timing the training iteration

        for logs in results["multistep_metrics"]:
            wandb.log(logs)
        wandb.log(logged_metrics)

        # TODO rewrite evals with the new method for handling reward_importance
        if i == 0 or (i + 1) % config.eval.every_n_steps == 0:
            with timer.eval_iter("eval_heads"):
                wandb.log(eval_fixed_head_rewards(actor, env, config, name=name))

        checkpoint_saver.maybe_save_checkpoint(i)

        # TODO remove code duplication in this eval
        total_expected_iter = i + 1 + timer.expected_remaining_iters()
        if num_recorded_videos < config.eval.max_num_vids \
                and i >= num_recorded_videos / config.eval.max_num_vids * total_expected_iter:
            with timer.eval_iter("generate_video"):
                num_recorded_videos += 1
                rollouts = []
                for iv in range(eval_importance.shape[0]):
                    tpenv = TransformedEnv(pixels_env, RewardImportanceTransform(num_rewards, weight=torch.from_numpy(
                        eval_importance[iv])))
                    vdata = tpenv.rollout(max_steps=tcfg.unroll_length, policy=actor, auto_reset=True,
                                          break_when_any_done=True)
                    rollouts.append(vdata)
                video = generate_video(rollouts, config.eval.text_color)
                # WanDB does not display high-fps well
                if config.eval.vid_fps >= 30:
                    video = video[::3]
                wandb.log({"video": wandb.Video(video, fps=config.eval.vid_fps, format="gif")})

        if i == 0 or (i + 1) % config.eval.pareto_every_n_steps == 0:
            with timer.eval_iter("compute_pareto"):
                logged_items = wandb_pareto_items(pixels_env, actor, eval_importance, hv_ref, true_pf, eval_importance,
                                                  i, config)
                wandb.log(logged_items)

        eval_times = {f"timer/time_{k}": v for k, v in timer.eval_time_for.items()}
        wandb.log({"timer/average_episode_length": avg_episode_len, "timer/train_iter_time": timer.train_iter_time,
                   "timer/expected_remaining_iter": timer.expected_remaining_iters(),
                   "timer/margin_for_final_eval": timer.margin_for_final_eval(), "timer/num_samples": timer.num_samples,
                   **eval_times})

        # Yeah please don't forget this
        data_collector.update_policy_weights_()


def ppo_update_on_data(data_flat, loss_module, optimizer, value_optimizer, advantage_module, scalarized_module,
                       beta_adjuster, entropy_controller, policy, replay_buffer, tcfg, cur_num_items) -> dict:
    batch_size = tcfg.num_unrolls * tcfg.unroll_length // tcfg.ppo.num_minibatches
    value_cfg = tcfg.value_loss

    avg_entropy = 0
    avg_policy_grad_norm = 0
    multistep_metrics = []
    for ei in range(tcfg.ppo.num_epochs):
        # Recompute advantages
        with torch.no_grad():
            advantage_module(data_flat)

            avg_metrics = {}
            if policy.does_use_popart:
                popart_logs = update_popart_and_log(data_flat["value_target"], policy)
                avg_metrics = {**avg_metrics, **popart_logs}
            scalarized_module(data_flat)
            avg_advantage = data_flat["advantage"].mean(dim=0)
            avg_advantage_dict = {f"normalization/avg_advantage_{i}": avg_advantage[i].item()
                                    for i in range(avg_advantage.shape[0])}
            avg_target_value = data_flat["value_target"].mean(dim=0)
            avg_target_value_dict = {f"normalization/avg_target_value_{i}": avg_target_value[i].item()
                                     for i in range(avg_target_value.shape[0])}
            avg_predicted_value = data_flat["predicted_value"].mean(dim=0)
            avg_predicted_value_dict = {f"normalization/avg_predicted_value_{i}": avg_predicted_value[i].item()
                                        for i in range(avg_predicted_value.shape[0])}
            avg_metrics = {**avg_metrics, **avg_advantage_dict, **avg_target_value_dict, **avg_predicted_value_dict}

            multistep_metrics.append(avg_metrics)

            replay_buffer.extend(data_flat)
        if value_cfg.optimize_jointly:
            for bi in range(tcfg.ppo.num_minibatches):
                batch_data = replay_buffer.sample(batch_size)
                losses = loss_module(batch_data)
                loss_entropy, logs_entropy = entropy_controller.entropy_loss(losses["entropy"], cur_num_items)
                actor_loss = losses["loss_objective"] + loss_entropy
                critic_loss = losses["loss_critic"]
                avg_entropy += losses["entropy"].item()
                if torch.isnan(actor_loss) or torch.isnan(critic_loss) or torch.isnan(loss_entropy):
                    print(f"Objective loss: {losses['loss_objective']} Actor loss: {actor_loss}, "
                          f" critic loss: {critic_loss}, entropy loss: {loss_entropy}")

                # Make a single step on both actor and critic params while controlling for the relative norm
                # of gradients using beta_adjuster
                logs = joint_actor_critic_step_and_log(actor_loss, critic_loss, beta_adjuster, policy,
                                                       optimizer, tcfg.optim.max_grad_norm)
                logs["epoch_index"] = ei
                logs["batch_index"] = bi
                logs = {**logs, **logs_entropy}
                avg_policy_grad_norm += logs["normalization/policy_grad_norm"]
                multistep_metrics.append(logs)
        else:
            # Run multiple inner epochs for the critic
            for ci in range(value_cfg.num_iter):
                for bi in range(tcfg.ppo.num_minibatches):
                    batch_data = replay_buffer.sample(batch_size)
                    critic_loss = loss_module.loss_critic(batch_data)
                    logs = separate_critic_step_and_log(critic_loss, policy, value_optimizer,
                                                        tcfg.optim.max_grad_norm)
                    multistep_metrics.append(logs)
                replay_buffer.empty()
                replay_buffer.extend(data_flat)
            for bi in range(tcfg.ppo.num_minibatches):
                batch_data = replay_buffer.sample(batch_size)
                losses = loss_module(batch_data, compute_critic_loss=False)
                loss_entropy, logs_entropy = entropy_controller.entropy_loss(losses["entropy"], cur_num_items)
                avg_entropy += losses["entropy"].item()
                actor_loss = losses["loss_objective"] + loss_entropy
                logs = actor_step_and_log(actor_loss, policy, optimizer, tcfg.optim.max_grad_norm)
                logs = {**logs, **logs_entropy}
                avg_policy_grad_norm += logs["normalization/policy_grad_norm"]
                multistep_metrics.append(logs)
    avg_entropy /= tcfg.ppo.num_minibatches * tcfg.ppo.num_epochs
    avg_policy_grad_norm /= tcfg.ppo.num_minibatches * tcfg.ppo.num_epochs
    update_results = {"avg_entropy": avg_entropy, "avg_policy_grad_norm": avg_policy_grad_norm,
                      "multistep_metrics": multistep_metrics}
    return update_results


def a2c_update_on_data(data_flat, loss_module, optimizer, value_optimizer, advantage_module, scalarized_module,
                       beta_adjuster, entropy_controller, policy, tcfg, cur_num_items) -> dict:
    value_cfg = tcfg.value_loss
    multistep_metrics = []

    with torch.no_grad():
        advantage_module(data_flat)
        if policy.does_use_popart:
            popart_logs = update_popart_and_log(data_flat["value_target"], policy)
            multistep_metrics.append(popart_logs)
        scalarized_module(data_flat)

    if value_cfg.optimize_jointly:
        losses = loss_module(data_flat)
        loss_entropy, logs_entropy = entropy_controller.entropy_loss(losses["entropy"], cur_num_items)
        actor_loss = losses["loss_objective"] + loss_entropy
        critic_loss = losses["loss_critic"]
        avg_entropy = losses["entropy"].item()
        # Make a single step on both actor and critic params while controlling for the relative norm
        # of gradients using beta_adjuster
        logs = joint_actor_critic_step_and_log(actor_loss, critic_loss, beta_adjuster, policy,
                                               optimizer, tcfg.optim.max_grad_norm)
        logs = {**logs, **logs_entropy}
        avg_policy_grad_norm = logs["normalization/policy_grad_norm"]
        multistep_metrics.append(logs)
    else:
        # Make multiple steps over the value predictor loss
        for ci in range(value_cfg.num_iter):
            critic_loss = loss_module.loss_critic(data_flat)
            logs = separate_critic_step_and_log(critic_loss, policy, value_optimizer,
                                                tcfg.optim.max_grad_norm)
            multistep_metrics.append(logs)

        losses = loss_module(data_flat, compute_critic_loss=False)
        loss_entropy, logs_entropy = entropy_controller.entropy_loss(losses["entropy"], cur_num_items)
        avg_entropy = losses["entropy"].item()
        actor_loss = losses["loss_objective"] + loss_entropy
        logs = actor_step_and_log(actor_loss, policy, optimizer, tcfg.optim.max_grad_norm)
        logs = {**logs, **logs_entropy}
        avg_policy_grad_norm = logs["normalization/policy_grad_norm"]
        multistep_metrics.append(logs)

    update_results = {"avg_entropy": avg_entropy, "avg_policy_grad_norm": avg_policy_grad_norm,
                      "multistep_metrics": multistep_metrics}
    return update_results
