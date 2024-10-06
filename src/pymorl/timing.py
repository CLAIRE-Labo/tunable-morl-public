from __future__ import annotations

from time import time
from typing import Optional
import logging

from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Usage:
# timer = TrainingTimer(config)
# for _ in timer.step_iterator():
#     with timer.train_iter():
#         # perform training iteration
#         timer.used_samples(num_used_env_transitions)
#     # log timer.train_iter_time, timer.expected_remaining_iters()
#     if should_eval_x:
#         with timer.eval_iter("x"):
#            # perform evaluation
#     if should_eval_y:
#         with timer.eval_iter("y"):
#            # perform evaluation
#     # log timer.eval_time_for, timer.margin_for_final_eval()
class TrainingTimer:
    def __init__(self, config: DictConfig):
        self.start_time = time()
        self.train_beta = config.training.timer.iter_average_beta
        self.eval_beta = config.training.timer.iter_average_beta
        self.train_iter_time = None
        self.allocated_time = config.training.timer.allocated_time
        self.num_iter = 0
        self.num_samples = 0
        self.last_used_samples = 10
        self.avg_used_samples = None
        self.max_num_samples = config.training.num_samples
        # Only used if allocated_time is None and num_samples is None
        self.max_num_iter = config.training.num_iter

        self.eval_time_for = {}
        self.num_eval_for = {}
        self.eval_frac_for = {}
        num_eval_traj = config.eval.num_pareto_points * config.eval.num_unrolls
        num_final_traj = config.eval.num_final_pareto_points * config.eval.num_unrolls
        self.final_eval_safety_factor = config.training.timer.final_eval_safety_factor * num_final_traj / num_eval_traj
        logger.info(f"Final eval safety factor: {self.final_eval_safety_factor:.2f}")

    def should_do_another_iter(self) -> bool:
        if self.max_num_samples is not None:
            return self.num_samples < self.max_num_samples
        if self.allocated_time is None:
            return self.num_iter < self.max_num_iter
        elapsed = time() - self.start_time
        train_tm = self.train_iter_time if self.train_iter_time is not None else 0
        return elapsed + train_tm + self.margin_for_final_eval() < self.allocated_time

    def expected_remaining_iters(self):
        if self.max_num_samples is not None:
            div = self.avg_used_samples if self.avg_used_samples is not None else 1
            return (self.max_num_samples - self.num_samples) / div
        if self.allocated_time is None:
            return self.max_num_iter - self.num_iter
        if self.train_iter_time is None:
            return 1000
        elapsed = time() - self.start_time
        eval_time_per_iter = {name: self.eval_frac_for[name] * tm for name, tm in self.eval_time_for.items()}
        total_time_per_iter = sum(eval_time_per_iter.values()) + self.train_iter_time

        return (self.allocated_time - elapsed - self.margin_for_final_eval()) / total_time_per_iter

    # TODO rewrite using yield
    def train_iter(self) -> _TrainIter:
        return self._TrainIter(self)

    def eval_iter(self, eval_name: str) -> _EvalIter:
        return self._EvalIter(self, eval_name)

    def step_iterator(self) -> _StepIterator:
        return self._StepIterator(self)

    def add_samples(self, num_used_env_transitions: int):
        self.last_used_samples = num_used_env_transitions
        if self.avg_used_samples is None:
            self.avg_used_samples = num_used_env_transitions
        else:
            self.avg_used_samples = (1 - self.train_beta) * num_used_env_transitions \
                                    + self.train_beta * self.avg_used_samples
        self.num_samples += num_used_env_transitions

    def _register_train_iter(self, elapsed: float):
        self.num_iter += 1
        if self.train_iter_time is None:
            self.train_iter_time = elapsed
        else:
            self.train_iter_time = (1 - self.train_beta) * elapsed + self.train_beta * self.train_iter_time

    def _register_eval_iter(self, elapsed: float, eval_name: str):
        cur_eval_time = elapsed
        if eval_name in self.eval_time_for.keys():
            # Conservative estimation of the time that will be spent on evaluation to leave enough in the end
            # self.eval_time_for[eval_name] = max(cur_eval_time, self.eval_time_for[eval_name])
            # Well run:ai does not have hard deadlines on jobs, so we can be more aggressive
            self.eval_time_for[eval_name] = (1 - self.eval_beta) * cur_eval_time \
                                            + self.eval_beta * self.eval_time_for[eval_name]
            self.num_eval_for[eval_name] += 1
            self.eval_frac_for[eval_name] = (self.num_eval_for[eval_name] - 1) / self.num_iter
        else:
            self.eval_time_for[eval_name] = cur_eval_time
            self.num_eval_for[eval_name] = 1
            self.eval_frac_for[eval_name] = 0.01

    def margin_for_final_eval(self) -> float:
        if self.allocated_time is None:
            return 0
        return self.final_eval_safety_factor * sum(v for v in self.eval_time_for.values())

    class _TrainIter:
        def __init__(self, timer: TrainingTimer):
            self.timer = timer

        def __enter__(self):
            self.start_time = time()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.elapsed = time() - self.start_time
            self.timer._register_train_iter(self.elapsed)

    class _EvalIter:
        def __init__(self, timer: TrainingTimer, eval_name: str):
            self.timer = timer
            self.eval_name = eval_name

        def __enter__(self):
            self.start_time = time()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.elapsed = time() - self.start_time
            self.timer._register_eval_iter(self.elapsed, self.eval_name)

    class _StepIterator:
        def __init__(self, timer: TrainingTimer):
            self.timer = timer
            self.pbar = tqdm(desc="Training")

        def __iter__(self):
            return self

        def __next__(self):
            remaining_iters = self.timer.expected_remaining_iters()
            postfix_str = f"Approx. remaining iter: {remaining_iters:.0f}"
            if self.timer.max_num_samples is not None:
                postfix_str += f", remaining samples: {self.timer.max_num_samples}-{self.timer.num_samples}" \
                               f"={self.timer.max_num_samples - self.timer.num_samples}"
            self.pbar.set_postfix_str(postfix_str)
            self.pbar.update()

            if self.timer.should_do_another_iter():
                return self.timer.num_iter
            else:
                self.pbar.close()
                raise StopIteration
