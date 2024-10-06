from __future__ import annotations

from typing import List, Optional
from enum import Enum
import logging

import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator

from morl_baselines.common.pareto import filter_pareto_dominated
from morl_baselines.common.performance_indicators import hypervolume, igd, sparsity, expected_utility, \
    maximum_utility_loss


logger = logging.getLogger(__name__)


class ParetoPointStatus(Enum):
    DOMINATED = 0
    DOMINATES = 1
    INCOMPARABLE = 2


def remove_dominated_nonstrict(points: np.ndarray) -> np.ndarray:
    assert points.ndim == 2, f"points must be 2D, got {points.shape=}"

    non_redundant_inds = []
    for i in range(points.shape[0]):
        is_dominated_by = np.all(points[i, np.newaxis, :] <= points, axis=1)
        is_redundant = np.any(is_dominated_by[:i]) or np.any(is_dominated_by[i + 1:])
        if not is_redundant:
            non_redundant_inds.append(i)
    return points[non_redundant_inds, :]


def sum_linf_igd(points_from: np.ndarray, points_to: np.ndarray) -> float:
    assert points_from.ndim == 2, f"points_from must be 2D, got {points_from.shape=}"
    assert points_to.ndim == 2, f"points_to must be 2D, got {points_to.shape=}"
    assert points_from.shape[1] == points_to.shape[1], \
        f"points_from and points_to must have the same number of dimensions, got" \
        f" {points_from.shape=} and {points_to.shape=}"
    dists = np.min(np.linalg.norm(points_from[np.newaxis, :, :] - points_to[:, np.newaxis, :], axis=-1, ord=np.inf),
                   axis=0)
    return np.sum(dists)


class ParetoFront:
    def __init__(self, points: np.ndarray | List[np.ndarray], num_samples: int = 10000, num_expansions: int = 5000):
        points = np.array(points)
        assert points.ndim == 2
        self.dim = points.shape[1]
        assert self.dim >= 2, f"points must have at least 2 dimensions, got {self.dim}"

        self.input_points = points
        num_points = points.shape[0]
        points = filter_pareto_dominated(points)
        logger.info(f'Created PF with {points.shape[0]} non-redundant points out of {num_points} total points')
        # print(f'Created PF with {points.shape[0]} non-redundant points out of {num_points} total points', flush=True)

        if points.shape[0] <= self.dim + 1:
            # The PF is too small to interpolate, let's expand!
            logger.info(f'PF is too small to interpolate, expanding by {num_expansions} points')
            for _ in range(num_expansions):
                start_point = np.random.randint(0, points.shape[0])
                subtr_dim = np.random.randint(0, self.dim - 1)
                newp = points[start_point, :].copy()
                newp[subtr_dim] -= np.random.uniform(0, 1)
                is_dominated_by = np.all(newp[np.newaxis, :] <= points, axis=1)
                if not np.any(is_dominated_by[:start_point]) and not np.any(is_dominated_by[start_point + 1:]):
                    points = np.concatenate([points, newp[np.newaxis, :]], axis=0)
            logger.info(f'Expanded:\n{points}')

        self.filtered_points = points
        num_points = points.shape[0]
        self.means = np.mean(points, axis=0, keepdims=True)
        self.stds = np.std(points, axis=0, keepdims=True) if num_points > 1 else np.ones((1, self.dim))
        self.points_normalized = (points - self.means) / self.stds
        try:
            if self.dim > 2:
                self.interp = LinearNDInterpolator(self.points_normalized[:, :-1], self.points_normalized[:, -1], )
            else:
                self.interp = interp1d(self.points_normalized[:, 0], self.points_normalized[:, 1])

            self.minp = np.min(self.points_normalized, axis=0)
            self.maxp = np.max(self.points_normalized, axis=0)
            num_side = int(num_samples ** (1 / (self.dim - 1)))
            coords = [np.linspace(self.minp[i], self.maxp[i], num_side) for i in range(self.dim - 1)]
            grid = np.meshgrid(*coords)
            last_coord = self.interp(*grid)
            dense_points_normalized = np.stack([*grid, last_coord], axis=-1).reshape((-1, self.dim))
            ok_inds = np.where(np.isfinite(dense_points_normalized[:, -1]))[0]
            self.dense_points_normalized = dense_points_normalized[ok_inds, ...]
            self.dense_points = self.dense_points_normalized * self.stds + self.means
        except Exception as e:
            logger.error(f'Failed to interpolate PF:\n{points}, {e}')
            self.interp = None
            self.dense_points = self.filtered_points
            self.dense_points_normalized = self.points_normalized

    def predict_last_reward(self, point: np.ndarray | float) -> Optional[float]:
        if not isinstance(point, np.ndarray):
            point = np.array([point])
        if point.ndim == 0:
            point = np.array([point])
        assert point.ndim == 1, f"point must be 1D, got {point.shape=}"
        assert point.shape[0] == self.points_normalized.shape[1] - 1, \
            f"point must have {self.points_normalized.shape[1] - 1} dimensions, got {point.shape[0]}"
        point_normalized = (point - self.means[0, :-1]) / self.stds[0, :-1]
        if self.dim == 2 and (point_normalized[0] < self.minp[0] or point_normalized[0] > self.maxp[0]):
            return None
        if self.interp is None:
            return None
        val = self.interp(point_normalized)
        if isinstance(val, np.ndarray):
            val = val.item()
        if np.isnan(val):
            return None
        else:
            return val * self.stds[0, -1] + self.means[0, -1]

    def point_status(self, point: np.ndarray) -> ParetoPointStatus:
        assert point.ndim == 1, f"point must be 1D, got {point.shape=}"
        assert point.shape[0] == self.points_normalized.shape[1], \
            f"point must have {self.points_normalized.shape[1]} dimensions, got {point.shape[0]}"

        # our point dominates if there exists a point on the PF that is worse or equal in all dimensions
        dominates = np.any(np.all(point[np.newaxis, :] >= self.dense_points, axis=1))
        if dominates:
            return ParetoPointStatus.DOMINATES
        # our point is dominated if there exists a point on the PF that is not worse in all dimensions
        is_dominated = np.any(np.all(point[np.newaxis, :] <= self.dense_points, axis=1))
        if is_dominated:
            return ParetoPointStatus.DOMINATED
        else:
            return ParetoPointStatus.INCOMPARABLE

    def hypervolume(self, ref_point: np.ndarray, use_dense_sample: bool = False,
                    normalizing_pf: Optional[ParetoFront] = None) -> float:
        sample = self.dense_points if use_dense_sample else self.filtered_points

        if normalizing_pf is not None:
            ref_point = ref_point.reshape((1, -1))
            maxp = np.max(normalizing_pf.filtered_points, axis=0, keepdims=True)
            sample = (sample - ref_point) / (maxp - ref_point)
            ref_point = np.zeros_like(ref_point)
        ref_point = ref_point.flatten()

        return hypervolume(ref_point, sample)

    @staticmethod
    def suboptimality(est_pf: np.ndarray, true_pf: ParetoFront) -> float:
        assert est_pf.ndim == 2, f"est_pf must be 2D, got {est_pf.shape=}"
        assert est_pf.shape[1] == true_pf.dim, \
            f"est_pf must have {true_pf.dim} dimensions, got {est_pf.shape[1]}"

        est_pf = filter_pareto_dominated(est_pf)
        est_pf_norm = (est_pf - true_pf.means) / true_pf.stds
        statuses = [true_pf.point_status(p) for p in est_pf]
        dominated = np.array([est_pf_norm[i] for i, s in enumerate(statuses) if s == ParetoPointStatus.DOMINATED])
        dominating = np.array([est_pf_norm[i] for i, s in enumerate(statuses) if s == ParetoPointStatus.DOMINATES])
        # incomparable = np.ndarray(
        #     [est_pf_norm[i] for i, s in enumerate(statuses) if s == ParetoPointStatus.INCOMPARABLE])

        # TODO separately report incomparable points
        sum_dominated = sum_linf_igd(dominated, true_pf.dense_points_normalized) if dominated.shape[0] > 0 else 0
        sum_dominating = sum_linf_igd(dominating, true_pf.dense_points_normalized) if dominating.shape[0] > 0 else 0
        return (sum_dominated - sum_dominating) / np.max([dominated.shape[0] + dominating.shape[0], 1])

    @staticmethod
    def incompleteness(est_pf: np.ndarray, true_pf: ParetoFront) -> float:
        assert est_pf.ndim == 2, f"est_pf must be 2D, got {est_pf.shape=}"
        assert est_pf.shape[1] == true_pf.dim, \
            f"est_pf must have {true_pf.dim} dimensions, got {est_pf.shape[1]}"

        est_pf = filter_pareto_dominated(est_pf)
        est_pf = (est_pf - true_pf.means) / true_pf.stds
        sum_dom = sum_linf_igd(true_pf.dense_points_normalized, est_pf) if est_pf.shape[0] > 0 else 0
        return sum_dom / np.max([true_pf.dense_points_normalized.shape[0], 1])


def pf_metrics(rewards: np.ndarray, true_pf: Optional[ParetoFront], hypervolume_reference: np.ndarray,
               reward_importance: np.ndarray) -> dict:
    rewards_pf = ParetoFront(rewards)
    sparsity_score = sparsity(rewards)

    eum = expected_utility(rewards, reward_importance)

    # We don't measure the volume of the interpolated PF (would set use_dense_sample=True) to be consistent with
    # MORL Benchmarks
    hv_unnorm = rewards_pf.hypervolume(hypervolume_reference, use_dense_sample=False)
    metric_dict = {"sparsity": sparsity_score, "unnormalized hypervolume": hv_unnorm, "expected utility": eum}
    if true_pf is not None:
        hv_norm = rewards_pf.hypervolume(hypervolume_reference, use_dense_sample=False, normalizing_pf=true_pf)
        suboptimality = ParetoFront.suboptimality(rewards, true_pf)
        incompleteness = ParetoFront.incompleteness(rewards, true_pf)
        igd_score = igd(true_pf.dense_points, rewards)
        mul = maximum_utility_loss(rewards, true_pf.filtered_points, reward_importance)
        metric_dict.update(
            {"normalized hypervolume": hv_norm, "suboptimality": suboptimality, "incompleteness": incompleteness,
             "inverse generational distance": igd_score, "maximum utility loss": mul})
    return metric_dict
