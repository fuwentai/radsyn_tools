"""Geometry helpers: coordinate conversions, Gram–Schmidt, lattice
   construction in a right‑handed screen coordinate system.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

__all__ = [
    "xyz_to_sphere",
    "line_parameter_equation_xyz",
    "R3GramSchmidt",
    "choose_lattice",
]


Array3D = np.ndarray  # (…, 3) shaped float64


def xyz_to_sphere(xyz: Array3D) -> Array3D:
    """Convert ⟨x,y,z⟩ Cartesian coordinates to spherical ⟨r,θ,φ⟩.

    Parameters
    ----------
    xyz : ndarray[..., 3]
        Cartesian coordinates.

    Returns
    -------
    ndarray[..., 3]
        Spherical coordinates where r ≥ 0, θ ∈ [0,π], φ ∈ [0,2π).
    """
    x, y, z = (xyz[..., i] for i in range(3))
    r = np.linalg.norm(xyz, axis=-1)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return np.stack((r, theta, phi), axis=-1)


def line_parameter_equation_xyz(P: Array3D, direction: np.ndarray, s: np.ndarray) -> Array3D:
    """Parametric line through *P* running opposite *direction* by length *s*.

    The ray is defined as 𝑟(𝑠) = P − \hat{d}·s, so positive *s* marches *toward*
    the observer (opposite to the simulation‑to‑pixel direction).
    """
    n = -direction / np.linalg.norm(direction)
    x = P[..., 0] + n[0] * s
    y = P[..., 1] + n[1] * s
    z = P[..., 2] + n[2] * s
    return np.stack((x, y, z), axis=-1)


def R3GramSchmidt(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors (u,v) spanning the plane ⟂ *n* (left‑handed).

    The triple (u,v,n) forms a **left**‑handed basis so that (u×v)·n < 0 for
    compatibility with earlier routines.
    """
    n = np.asarray(n, dtype=float)
    # Choose two arbitrary non‑collinear seed vectors:
    u = np.array([1.0, 0.0, 0.0]) if n[0] == 0 else np.array([0.0, 1.0, 0.0])
    v = np.array([0.0, 1.0, 0.0]) if n[0] == 0 else np.array([0.0, 0.0, 1.0])
    # Project out n from u and v, then orthonormalise.
    u -= (u @ n) / (n @ n) * n
    v -= (v @ n) / (n @ n) * n
    v -= (v @ u) / (u @ u) * u
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    # Ensure left‑handedness (swap if necessary)
    if np.dot(np.cross(u, v), n) > 0:
        u, v = v, u
    return u, v


def choose_lattice(
    view_dir: np.ndarray,
    north_dir: np.ndarray,
    radius: int = 400,
    lattice_dist: float = 1.0,
    center: Tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
):
    """Generate a square lattice of *foot‑points* in the image plane.

    The image plane is perpendicular to *view_dir*. The +Y axis of the image is
    the projection of *north_dir* onto this plane; +X completes the right‑handed
    screen basis (east‑ward).

    Returns
    -------
    lattice : ndarray[(2r+1),(2r+1),3]
        Cartesian coordinates of foot‑points.
    north_vec : ndarray[3]
        Unit vector of +Y axis in real‑space coordinates.
    """
    u, v = R3GramSchmidt(view_dir)
    # Protect against view_dir ∥ north_dir → fallback to arbitrary (v)
    if np.allclose(np.cross(view_dir, north_dir), 0):
        north_vec = v
        east_vec = u
    else:
        nor_y = north_dir @ v
        nor_x = north_dir @ u
        norm = math.hypot(nor_x, nor_y)
        nor_y, nor_x = nor_y / norm, nor_x / norm
        north_vec = nor_y * v + nor_x * u
        east_vec = -nor_x * v + nor_y * u  # 90° CW from north
    east_step = lattice_dist * east_vec
    north_step = lattice_dist * north_vec

    grid = np.arange(-radius, radius + 1)
    ix, iy = np.meshgrid(grid, grid, indexing="ij")
    lattice = (
        ix[..., None] * east_step[None, None, :] +
        iy[..., None] * north_step[None, None, :]
    ) + center
    return lattice, north_vec