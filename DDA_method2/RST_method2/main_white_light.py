"""Command‑line entry point to render a single EUV channel image.

Usage (example):
    $ python main.py --normal 0,1,1 --radius 200 --width 2 \
                    --rho npz/rho.npz --R npz/R.npz --domain-left 1,0,0 --domain-right 3,pi,2*pi
"""
from __future__ import annotations

import argparse
import math
import gc
import time
from pathlib import Path

import numpy as np


from .geometry import choose_lattice, R3GramSchmidt
from .processing import ray_trace_image
from .radiation import absorption_EUV, radiation_EUV, absorption_radio, radiation_radio, radiation_white_light

def parse_vec3(text: str) -> tuple[float, float, float]:

    try:
        numbers = text.strip("()").split(",")   # 去掉括号再按逗号拆分
        if len(numbers) != 3:
            raise ValueError
        return tuple(float(n) for n in numbers)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "(x,y,z) or x,y,z  ，--normal 0,1,-1  or  --normal '(0,1,-1)'"
        )

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solar coronal ray‑tracing renderer")
    p.add_argument("--normal", type=parse_vec3, default=(1.0, 1.0, 1.0),
               metavar="(x,y,z)", help="--normal '(1,1,1)'" )
    p.add_argument("--radius", type=int, default=200)
    p.add_argument("--width", type=float, default=2.0)
    p.add_argument("--rho", type=Path, required=True)
    p.add_argument("--Te", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("image.npy"))
    p.add_argument("--domain-left",  type=parse_vec3,
               default=(0.0, 0.0, 0.0),
               metavar="r,θ,φ", help="网格最小坐标 (默认 0,0,0)",required=True)
    p.add_argument("--domain-right", type=parse_vec3,
               default=(5.0, math.pi, 2*math.pi),
               metavar="r,θ,φ", help="网格最大坐标 (默认 5,π,2π)",required=True)
    return p.parse_args()

def main() -> None:
    args = _parse_args()

    view_dir = np.array(args.normal)
    radius = args.radius
    width = args.width
    lattice_dist = width / radius

    # ‑‑‑ Load plasma variables
    Te = np.load(args.Te)
    rho = np.load(args.rho)
    shape = rho.shape

    # ‑‑‑ Build image‑plane lattice (four‑way split for 4 processes)
    P, north_vec = choose_lattice(view_dir, view_dir, radius, lattice_dist, center=(0, 0, 0))
    n = radius
    P_quads = [
        P[:n, :n],
        P[:n, n:],
        P[n:, :n],
        P[n:, n:],
    ]

    # ‑‑‑ Optical properties
    J = radiation_white_light(rho,R,7e10)
    kai = np.zeros(J.shape)
    

    # ‑‑‑ Ray‑trace image
    domain_left_edge  = np.asarray(args.domain_left,  dtype=float)
    domain_right_edge = np.asarray(args.domain_right, dtype=float)

    t0 = time.time()
    image = ray_trace_image(
        P_quads,
        view_dir,
        domain_left_edge,
        domain_right_edge,
        shape,
        kai,
        J,
        radius,
    )
    elapsed = time.time() - t0
    print(f"Ray‑tracing completed in {elapsed:.2f} s")

    # Log‑scale with NaN mask
    mask = image == 0.0
    log_img = np.where(mask, np.nan, np.log(image))
    np.save(args.out, log_img)
    print(f"Saved {args.out}")

    gc.collect()


if __name__ == "__main__":
    main()