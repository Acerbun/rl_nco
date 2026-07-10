from pathlib import Path

import numpy as np


def create_test_positions(
    num_users: int,
    num_topologies: int = 500,
    seed: int = 2026,
) -> np.ndarray:
    """
    Generate fixed user topologies for fair evaluation.

    Parameters
    ----------
    num_users:
        Number of ground users K.
    num_topologies:
        Number of test network realizations.
    seed:
        Base random seed.

    Returns
    -------
    np.ndarray
        User positions with shape
        (num_topologies, num_users, 2).
    """
    if num_users <= 0:
        raise ValueError("num_users must be positive.")

    if num_topologies <= 0:
        raise ValueError("num_topologies must be positive.")

    # Add num_users to the seed so that different K values
    # use different but reproducible test datasets.
    rng = np.random.default_rng(seed + num_users)

    positions = rng.uniform(
        low=0.0,
        high=500.0,
        size=(num_topologies, num_users, 2),
    )

    return positions.astype(np.float64)


def save_test_positions() -> None:
    """
    Generate and save fixed test positions for
    K = 5, 10, 15, and 20.
    """
    output_dir = Path("fixed_data/exact_benchmark")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    user_numbers = [5, 10, 15, 20]

    for num_users in user_numbers:
        positions = create_test_positions(
            num_users=num_users,
            num_topologies=500,
            seed=2026,
        )

        output_path = (
            output_dir
            / f"test_positions_K{num_users}.npy"
        )

        np.save(output_path, positions)

        print(
            f"Saved K={num_users}: "
            f"{positions.shape} -> {output_path}"
        )


if __name__ == "__main__":
    save_test_positions()