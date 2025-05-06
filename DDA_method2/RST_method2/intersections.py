"""Ray / curvilinear‑mesh intersection helpers."""
from __future__ import annotations

import numpy as np
from typing import Literal, Tuple

from .geometry import line_parameter_equation_xyz, xyz_to_sphere

__all__ = [
    "quad_equa_solve",
    "intersect_point_parameter",
    "lattice_ray_intersect_plane_spherical",
]


PlaneType = Literal["r", "theta", "phi"]


def quad_equa_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, *, larger: bool = True) -> np.ndarray:
    """Solve *ax² + bx + c = 0* element‑wise and return the chosen root."""
    disc = np.maximum(b ** 2 - 4 * a * c, 0.0)
    root = np.sqrt(disc)
    if larger:
        return (-b + root) / (2 * a)
    return (-b - root) / (2 * a)


def intersect_point_parameter(
    P: np.ndarray,  # lattice foot‑points
    direction: np.ndarray,
    case: PlaneType,
    value: float,
    *,
    larger_root: bool = True,
) -> np.ndarray:
    """Return ⟨s,r,θ,φ⟩ for the intersection of a ray with a constant‑*case* surface."""
    n = -direction / np.linalg.norm(direction)
    t0, t1, _ = P.shape
    if case == "r":
        r0 = value
        a = np.ones((t0, t1))
        b = 2 * (P[..., 0] * n[0] + P[..., 1] * n[1] + P[..., 2] * n[2])
        c = (P[..., 0] ** 2 + P[..., 1] ** 2 + P[..., 2] ** 2) - r0 ** 2
        s = quad_equa_solve(a, b, c, larger=larger_root)
    elif case == "theta":
        th0 = value
        cos2 = math.cos(th0) ** 2
        a = (cos2 - n[2] ** 2) * np.ones((t0, t1))
        b = 2 * (P[..., 0] * n[0] + P[..., 1] * n[1] + P[..., 2] * n[2]) * cos2 - 2 * P[..., 2] * n[2]
        c = (P[..., 0] ** 2 + P[..., 1] ** 2 + P[..., 2] ** 2) * cos2 - P[..., 2] ** 2
        if a == 0:
            s = -c / b
        else:
            s = quad_equa_solve(a, b, c, larger=larger_root)
    elif case == "phi":
        ph0 = value
        k = (n[0] * math.sin(ph0) - n[1] * math.cos(ph0)) * np.ones((t0, t1))
        b = P[..., 0] * math.sin(ph0) - P[..., 1] * math.cos(ph0)
        s = -b / k
    else:
        raise ValueError("case must be 'r', 'theta', or 'phi'")

    xyz = line_parameter_equation_xyz(P, direction, s)
    sph = xyz_to_sphere(xyz)
    s_exp = s[..., None]
    return np.concatenate((s_exp, sph), axis=-1)


def lattice_ray_intersect_plane_spherical(
    case: PlaneType,
    Lvalue: float,
    Rvalue: float,
    direction: np.ndarray,
    P: np.ndarray,
    *,
    step: int = 401,
):
    """Return intersections of a bundle of rays with evenly spaced *case*-surfaces."""
    delta = (Rvalue - Lvalue) / step
    values = np.linspace(Lvalue, Rvalue - delta, step)  # left‑inclusive
    results = []
    for val in values:
        results.append(intersect_point_parameter(P, direction, case, val))
        if case != "phi":  # r & θ surfaces are quadratic → two intersections
            results.append(intersect_point_parameter(P, direction, case, val, larger_root=False))
    return np.array(results)