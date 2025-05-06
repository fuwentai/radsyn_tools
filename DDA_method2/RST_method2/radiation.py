"""Radiative emissivity & absorption models."""
from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

__all__ = [
    "radiation_193A",
    "Ly_absorption",
    "white_light",
]


_AIA_RESP_DIR: Final = Path("../AIA_response")


def _load_aia_response(wavelength: int) -> np.ndarray:
    file = _AIA_RESP_DIR / f"{wavelength}_response_line.npz"
    with np.load(file) as npz:
        return npz["a"]  # shape (N,2) : logT, response


def radiation_EUV(Te: np.ndarray, rho: np.ndarray, wavelength: int, width: float) -> np.ndarray:
    """AIA instrument response × nₑ² emissivity (simplified optically thin)."""
    res_line = _load_aia_response(wavelength)
    te_log, response = res_line[:, 0], res_line[:, 1]
    temp_log = np.log10(np.nan_to_num(Te))
    res = np.interp(temp_log, te_log, response, left=0.0, right=0.0)
    emiss = res * (rho * 1e9) ** 2
    return emiss * width


def absorption_EUV(Te: np.ndarray, rho: np.ndarray, wavelength: int, width: float) -> np.ndarray:
    """Hydrogen + helium photo‑ionisation opacity (Ly‑α like) – simplified."""
    # Constants & abundances
    sigma_H1, sigma_He1, sigma_He2 = 5.16e-20, 9.25e-19, 7.17e-19
    wave_ref, rHe = 171.0, 0.1
    k_B = 1.380649e-23

    Ne = rho  # ρ already in cm⁻³ (electron density) in caller’s units
    Pe = Ne * k_B * Te * 1e7

    wl_ratio = wavelength / wave_ref
    s_H1 = wl_ratio ** 3 * sigma_H1
    s_He1 = wl_ratio ** 2 * sigma_He1
    s_He2 = wl_ratio ** 2.75 * sigma_He2

    # Saha‑Boltzmann factors (cf. Anzer & Heinzel 2005)
    def _w(E: float) -> np.ndarray:
        return 10 ** (np.log10(1.0) + 2.5 * np.log10(Te) - 5040 * E / Te - np.log10(Pe) - 0.48)

    w_H21, w_He21, w_He32 = map(_w, (13.6, 24.587, 54.416))
    w_H21, w_He21, w_He32 = (np.nan_to_num(w) for w in (w_H21, w_He21, w_He32))

    i0 = w_H21 / (1 + w_H21)
    j1 = w_He21 / (1 + w_He21 + w_He21 * w_He32)
    j2 = (w_He21 * w_He32) / (1 + w_He21 + w_He21 * w_He32)

    be = i0 + rHe * (j1 + 2 * j2)
    N_H = Ne / be
    N_H1 = N_H * (1 - i0)
    N_He1 = (1 - j1 - j2) * rHe * N_H
    N_He2 = j1 * rHe * N_H

    kai = N_H1 * s_H1 + N_He1 * s_He1 + N_He2 * s_He2
    kai = np.nan_to_num(kai)
    return kai * width


def radiation_white_light(rho: np.ndarray, R: np.ndarray, width: float) -> np.ndarray:

    a1, a2, a3 = 8.69e-7, 0.37, 0.63
    r = R
    cfunc = 4.0 / 3.0 - np.sqrt(1.0 - 1.0 / r ** 2) - (1.0 - 1.0 / r ** 2) ** 1.5 / 3.0
    df = (r - 1.0 / r) * (5.0 - 1.0 / r ** 2) * np.log((1.0 + 1.0 / r) / np.sqrt(1.0 - 1.0 / r ** 2))
    dfunc = 0.125 * (5.0 + 1.0 / r ** 2 - df)
    bf = a1 * (a2 * cfunc + a3 * dfunc)

    I0, sigma_T, u = 1.0, 7.95e-26, 0.56
    tomson = I0 * rho * 1e9 * np.pi * sigma_T / 2.0 * ((1 - u) * cfunc + u * dfunc)
    tomson = np.nan_to_num(tomson)
    return tomson * width


def radio(Te: np.ndarray, rho: np.ndarray, width: float) -> np.ndarray:

    sigma_H1, sigma_He1, sigma_He2 = 5.16e-20, 9.25e-19, 7.17e-19
    rHe = 0.1
    k_B = 1.380649e-23

    Ne = rho  # ρ already in cm⁻³ (electron density) in caller’s units
    Pe = Ne * k_B * Te * 1e7

    # Saha‑Boltzmann factors (cf. Anzer & Heinzel 2005)
    def _w(E: float) -> np.ndarray:
        return 10 ** (np.log10(1.0) + 2.5 * np.log10(Te) - 5040 * E / Te - np.log10(Pe) - 0.48)

    w_H21, w_He21, w_He32 = map(_w, (13.6, 24.587, 54.416))
    w_H21, w_He21, w_He32 = (np.nan_to_num(w) for w in (w_H21, w_He21, w_He32))

    i0 = w_H21 / (1 + w_H21)
    j1 = w_He21 / (1 + w_He21 + w_He21 * w_He32)
    j2 = (w_He21 * w_He32) / (1 + w_He21 + w_He21 * w_He32)

    be = i0 + rHe * (j1 + 2 * j2)
    N_H = Ne / be
    N_H1 = N_H * (1 - i0)
    N_He1 = (1 - j1 - j2) * rHe * N_H
    N_He2 = j1 * rHe * N_H
    
    h = sc.h*1e7
    k_b = sc.k*1e7
    niu = 3*1e8
    
    kai = np.where( Te <=2e5,(0.00978 * (rho/niu**2/Te**1.5) * (N_H2+N_He1+4*N_He2) * (18.2+1.5*np.log(Te)-np.log(niu))),(0.00978 * (rho/niu**2/Te**1.5) * (N_H2+N_He1+4*N_He2) * (24.5+np.log(Te)-np.log(niu))))*width
    kai = np.nan_to_num(kai, nan=0)
    # 设置最大值
    #max_value = 10e12
    
    # 将超过最大值的元素替换为 3e20
    #kai[kai > max_value] = 10e12
    
    J = np.where( Te <= 2e5,(3.772e-38 * (rho/Te**0.5) * np.exp(-1*h*niu/k_b/Te) * (N_H2+N_He1+4*N_He2) * (18.2+1.5*np.log(Te)-np.log(niu))),(3.772e-38 * (rho/Te**0.5) * np.exp(-1*h*niu/k_b/Te) * (N_H2+N_He1+4*N_He2) * (24.5+np.log(Te)-np.log(niu))))*width
    return kai,J
