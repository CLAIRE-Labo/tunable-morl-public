from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import numpy as np
from omegaconf import DictConfig

from policy import MultiRewardPolicy


class EntropySchedule(ABC):
    @abstractmethod
    def target_value(self, cur_num_items: int) -> float:
        pass

    # Currently for discrete action spaces
    @staticmethod
    def from_config(schedule_config: DictConfig, max_num_items: int, num_actions: int) -> EntropySchedule:
        max_entropy = np.log(num_actions)
        max_entropy = min(max_entropy, schedule_config.max)
        min_entropy = max(0, schedule_config.min)
        num_resets = schedule_config.num_resets
        if schedule_config.type == "cosine":
            return CosineSchedule(max_entropy, min_entropy, max_num_items, num_resets)
        elif schedule_config.type == "linear":
            assert num_resets == 0, "Linear schedule does not support resets"
            return LinearSchedule(max_entropy, min_entropy, max_num_items)
        elif schedule_config.type == "fun1":
            assert num_resets == 0, "Fun1 schedule does not support resets"
            return Fun1Schedule(max_entropy, min_entropy, max_num_items)
        else:
            raise ValueError(f"Unknown entropy control method: {schedule_config.method}")


# Start out flat and then decrease, with the decrease being faster at the end
class CosineSchedule(EntropySchedule):
    def __init__(self, init_value: float, final_value: float, max_num_items: int, num_resets: int):
        self.init_value = init_value
        self.final_value = final_value
        self.max_num_items = max_num_items
        self.num_resets = num_resets

    def target_value(self, cur_num_items: int) -> float:
        x = cur_num_items / self.max_num_items * (self.num_resets + 1)
        x = x - int(x)
        return self.final_value + (self.init_value - self.final_value) * np.cos(np.pi / 2 * x)


# Start out flat, decrease, then flatten out. An empirically selected schedule
class Fun1Schedule(EntropySchedule):
    def __init__(self, init_value: float, final_value: float, max_num_items: int):
        self.init_value = init_value
        self.final_value = final_value
        self.max_num_items = max_num_items

    def target_value(self, cur_num_items: int) -> float:
        x = np.clip(cur_num_items / self.max_num_items, 0, 0.9999)
        ynorm = 0.5 - np.cos(np.pi * ((1.0 - x) ** 1.3)) / 2
        return self.final_value + (self.init_value - self.final_value) * ynorm


class LinearSchedule(EntropySchedule):
    def __init__(self, init_value: float, final_value: float, max_num_items: int):
        self.init_value = init_value
        self.final_value = final_value
        self.max_num_items = max_num_items

    def target_value(self, cur_num_items: int) -> float:
        return self.init_value - (self.init_value - self.final_value) * cur_num_items / self.max_num_items


class EntropyController:
    # max_num_items can be either iterations or number of samples, depending on the termination criterion
    def __init__(self, policy: MultiRewardPolicy, entropy_config: DictConfig, max_num_items: int):
        self.policy = policy
        self.config = entropy_config
        self.damping = self.config.damping
        self.coef_lr = self.config.coef_lr
        self.cur_iter = 0
        self.entropy_coef = self.config.init_coef
        self.max_num_items = max_num_items
        self.schedule = EntropySchedule.from_config(self.config.schedule, max_num_items, policy.num_actions)

    # TODO add support for larger batches for coef updates
    # Updates the entropy coef and returns the loss to be added to the total loss and wandb logs
    def entropy_loss(self, entropy: torch.Tensor, cur_num_items: int) -> (torch.Tensor, dict):
        logs = {"entropy_controller/entropy": entropy.item()}
        if self.config.method == "const_coef":
            loss = -self.entropy_coef * entropy
            logs["entropy_controller/loss"] = loss.item()
            return loss, logs
        assert self.config.method == "mdmm", f"Unknown entropy control method: {self.config.method}"

        target_entropy = self.schedule.target_value(cur_num_items)
        logs["entropy_controller/target_entropy"] = target_entropy
        entropy_sg = entropy.detach().item()
        entropy_loss = (self.entropy_coef + self.damping * (entropy_sg - target_entropy)) * entropy
        logs["entropy_controller/loss"] = entropy_loss.item()
        self.entropy_coef += self.coef_lr * (entropy_sg - target_entropy)
        logs["entropy_controller/entropy_coef"] = self.entropy_coef
        return entropy_loss, logs
