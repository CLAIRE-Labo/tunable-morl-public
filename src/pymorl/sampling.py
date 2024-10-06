from typing import Optional
import logging

import numpy as np


logger = logging.getLogger(__name__)


def uniform_simplex(shape, dtype=np.float32, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    assert 1 <= len(shape), f"Expected shape of length at least 1, got {shape}"
    if rng is None:
        x = np.random.exponential(scale=1.0, size=shape).astype(dtype)
    else:
        x = rng.exponential(scale=1.0, size=shape).astype(dtype)
    x /= x.sum(axis=-1, keepdims=True)
    return x


# This produces random points on the simplex that are approximately equally spaced
# Useful to visualize Pareto fronts with more than two objectives
def stratified_uniform_simplex(shape, dtype=np.float32, initial_dense_factor: float = 50,
                               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    assert 2 == len(shape), f"Expected shape of length 2, got {shape}"
    sampled_shape = (int(shape[0] * initial_dense_factor), shape[1])
    x = uniform_simplex(sampled_shape, dtype=dtype, rng=rng)
    distances_sq = np.sum((x[np.newaxis, :, :] - x[:, np.newaxis, :]) ** 2, axis=-1)

    def sample_points(dist):
        sampled_points = [0]
        for i in range(1, sampled_shape[0]):
            if np.all(distances_sq[sampled_points, i] >= dist):
                sampled_points.append(i)
        return sampled_points

    # Binary search to find min distance s.t. we can sample shape[0] points min_distance apart
    min_distance = 0.0
    max_distance = np.max(distances_sq)
    while max_distance - min_distance > 1e-8:
        mid_distance = (min_distance + max_distance) / 2
        # greedily select points that are at least mid_distance apart
        pnts = sample_points(mid_distance)
        if len(pnts) >= shape[0]:
            min_distance = mid_distance
        else:
            max_distance = mid_distance
    logger.info(f"Found min distance {min_distance} in for {shape[0]} points")
    print(f"Found min distance {min_distance} for {shape[0]} points")

    pnts = sample_points(min_distance)
    return x[pnts, :]


def generate_fixed_importance(num_points: int, num_rewards: int, stratified: bool = True) -> np.ndarray:
    # The newly created rng ensures that the alphas will stay the same between runs
    rng = np.random.default_rng(239)
    if stratified:
        return stratified_uniform_simplex((num_points, num_rewards), rng=rng)
    else:
        return uniform_simplex((num_points, num_rewards), rng=rng)
