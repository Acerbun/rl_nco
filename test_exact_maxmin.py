import numpy as np

from exact_maxmin import exact_maxmin_allocation


def main() -> None:
    gains = np.array(
        [1.0e-12, 2.0e-12, 4.0e-12],
        dtype=np.float64,
    )

    power, rates, min_rate = exact_maxmin_allocation(
        channel_gains=gains,
        total_power=1.0,
        bandwidth=2.0e6,
        noise_power=1.0e-14,
    )

    print("Optimal power:", power)
    print("Rates (Mbps):", rates)
    print("Minimum rate (Mbps):", min_rate)
    print("Total power:", np.sum(power))
    print("Rate difference:", np.max(rates) - np.min(rates))

    assert np.isclose(np.sum(power), 1.0)
    assert np.all(power >= 0.0)

    # Exact max-min allocation should equalize all user rates.
    assert np.max(rates) - np.min(rates) < 1e-10

    print("Exact max-min test passed.")


if __name__ == "__main__":
    main()