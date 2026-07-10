"""Closed-form max-min power allocation for the adopted OFDMA model."""

from __future__ import annotations

import numpy as np


def exact_maxmin_allocation(
    channel_gains: np.ndarray,
    total_power: float,
    bandwidth: float,
    noise_power: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the exact max-min fair power allocation.

    Parameters
    ----------
    channel_gains:
        Positive channel power gains with shape (K,).
    total_power:
        Total UAV transmit power in watts.
    bandwidth:
        Total system bandwidth in Hz.
    noise_power:
        Effective noise power per user subband in watts.

    Returns
    -------
    optimal_power:
        Exact max-min power allocation with shape (K,).
    rates_mbps:
        Individual user rates in Mbps.
    min_rate_mbps:
        Exact optimal minimum user rate in Mbps.
    """
    gains = np.asarray(channel_gains, dtype=np.float64)

    if gains.ndim != 1 or gains.size == 0:
        raise ValueError("channel_gains must be a non-empty 1-D array.")

    if not np.all(np.isfinite(gains)):
        raise ValueError("channel_gains contains NaN or infinity.")

    if np.any(gains <= 0.0):
        raise ValueError("All channel gains must be strictly positive.")

    if total_power <= 0.0:
        raise ValueError("total_power must be positive.")

    if bandwidth <= 0.0 or noise_power <= 0.0:
        raise ValueError("bandwidth and noise_power must be positive.")

    num_users = gains.size
    inverse_gains = 1.0 / gains

    # Closed-form max-min allocation.
    optimal_power = (
        total_power
        * inverse_gains
        / np.sum(inverse_gains)
    )

    snr = optimal_power * gains / noise_power

    rates_mbps = (
        bandwidth
        / num_users
        * np.log2(1.0 + snr)
        / 1e6
    )

    min_rate_mbps = float(np.min(rates_mbps))

    # Numerical consistency checks.
    if not np.isclose(
        np.sum(optimal_power),
        total_power,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("Exact allocation violates total-power constraint.")

    if np.any(optimal_power < -1e-12):
        raise RuntimeError("Exact allocation contains negative power.")

    return optimal_power, rates_mbps, min_rate_mbps