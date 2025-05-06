"""Parallel‑friendly helpers that wrap lattice_ray_big11.compute_image."""
from __future__ import annotations

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence, Tuple

import lattice_ray_big11  # compiled cython module

from .intersections import lattice_ray_intersect_plane_spherical

__all__ = [
    "prepare_chunk",
    "compute_image_chunk",
    "ray_trace_image",
]


RayBundle = np.ndarray  # alias


def prepare_chunk(
    P_chunk: RayBundle,
    direction: np.ndarray,
    domain_left_edge: Sequence[float],
    domain_right_edge: Sequence[float],
    shape: Tuple[int, int, int],
):
    rrst = lattice_ray_intersect_plane_spherical("r", domain_left_edge[0], domain_right_edge[0], direction, P_chunk, step=shape[0] + 1)
    trst = lattice_ray_intersect_plane_spherical("theta", domain_left_edge[1], domain_right_edge[1], direction, P_chunk, step=shape[1] + 1)
    prst = lattice_ray_intersect_plane_spherical("phi", domain_left_edge[2], domain_right_edge[2], direction, P_chunk, step=shape[2] + 1)
    return np.vstack((rrst, trst, prst))


def compute_image_chunk(args):
    intp, kai, J, domain_left_edge, domain_right_edge, radius = args
    return lattice_ray_big11.compute_image(intp, kai, J, domain_left_edge, domain_right_edge, radius)


def ray_trace_image(
    P_splits: Sequence[RayBundle],
    direction: np.ndarray,
    domain_left_edge: Sequence[float],
    domain_right_edge: Sequence[float],
    rho_shape: Tuple[int, int, int],
    kai: np.ndarray,
    J: np.ndarray,
    radius: int,
    *,
    max_workers: int | None = None,
):
    """High‑level helper that maps *prepare_chunk* → Cython *compute_image* in parallel."""
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        intp_futs = [
            ex.submit(prepare_chunk, Pc, direction, domain_left_edge, domain_right_edge, rho_shape)
            for Pc in P_splits
        ]
        intps = [f.result() for f in intp_futs]
        args = [
            (intp, kai, J, domain_left_edge, domain_right_edge, radius // 2) for intp in intps
        ]
        img_futs = [ex.submit(compute_image_chunk, a) for a in args]
        images = [f.result() for f in img_futs]
    # Re‑assemble quadrants → full image
    n = radius
    final = np.zeros((2 * n, 2 * n))
    final[:n, :n], final[:n, n:], final[n:, :n], final[n:, n:] = images
    return final