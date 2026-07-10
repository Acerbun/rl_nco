"""
Evaluate existing DDQN checkpoints against the closed-form
exact max-min allocation.

This script does NOT train any model.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

import agent as agent_module
from agent import Modified_DDQN_Agent, Regular_DDQN_Agent
from env_uav_static import UAV_Emergency_Env
from exact_maxmin import exact_maxmin_allocation
from settings import DEVICE


# ============================================================
# 1. Basic settings
# ============================================================

ROOT = Path(__file__).resolve().parent

TEST_DATA_DIR = ROOT / "fixed_data" / "exact_benchmark"
OUTPUT_DIR = ROOT / "exact_benchmark_results"

# First use 50 to verify that the script works.
# After successful verification, change it to 500.
N_TEST_TOPOLOGIES = 500

SEED_IDS = (0, 1, 2)


HEIGHT_CONFIGS = [
    {
        "label": "H=50",
        "x_value": 50,
        "num_users": 10,
        "height": 50.0,
        "model_dir": ROOT / "Models_h50" / "UAV_Static",
        "positions_file": TEST_DATA_DIR / "test_positions_K10.npy",
    },
    {
        "label": "H=100",
        "x_value": 100,
        "num_users": 10,
        "height": 100.0,
        "model_dir": ROOT / "Models_h100" / "UAV_Static",
        "positions_file": TEST_DATA_DIR / "test_positions_K10.npy",
    },
    {
        "label": "H=150",
        "x_value": 150,
        "num_users": 10,
        "height": 150.0,
        "model_dir": ROOT / "Models_h150" / "UAV_Static",
        "positions_file": TEST_DATA_DIR / "test_positions_K10.npy",
    },
    {
        "label": "H=200",
        "x_value": 200,
        "num_users": 10,
        "height": 200.0,
        "model_dir": ROOT / "Models_h200" / "UAV_Static",
        "positions_file": TEST_DATA_DIR / "test_positions_K10.npy",
    },
]


USER_CONFIGS = [
    {
        "label": "K=5",
        "x_value": 5,
        "num_users": 5,
        "height": 100.0,
        "model_dir": ROOT / "Models_user05" / "UAV_Static",
        "positions_file": TEST_DATA_DIR / "test_positions_K5.npy",
    },
    {
        "label": "K=10",
        "x_value": 10,
        "num_users": 10,
        "height": 100.0,
        "model_dir": ROOT / "Models_user10" / "UAV_Static",
        "positions_file": TEST_DATA_DIR / "test_positions_K10.npy",
    },
    {
        "label": "K=15",
        "x_value": 15,
        "num_users": 15,
        "height": 100.0,
        "model_dir": ROOT / "Models_user15" / "UAV_Static",
        "positions_file": TEST_DATA_DIR / "test_positions_K15.npy",
    },
    {
        "label": "K=20",
        "x_value": 20,
        "num_users": 20,
        "height": 100.0,
        "model_dir": ROOT / "Models_user20" / "UAV_Static",
        "positions_file": TEST_DATA_DIR / "test_positions_K20.npy",
    },
]


# ============================================================
# 2. Loading utilities
# ============================================================

def load_state_dict_safely(checkpoint_path: Path) -> dict[str, Any]:
    """Load a PyTorch state dictionary across PyTorch versions."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist:\n{checkpoint_path}"
        )

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        state_dict = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Unexpected checkpoint format: {checkpoint_path}"
        )

    return state_dict


def create_loaded_agent(
    agent_class,
    env: UAV_Emergency_Env,
    model_dir: Path,
    agent_name: str,
    seed_id: int,
):
    """
    Create an agent and manually load the requested checkpoint.

    agent_name:
        "Regular"  -> sum-rate-oriented DDQN
        "Modified" -> Q-Min
    """

    # Prevent Agent.__init__ from accidentally loading Models/UAV_Static.
    agent_module.ENVIRONMENT_NAME = "__benchmark_no_autoload__"

    agent = agent_class(
        n_actions=env.n_actions,
        n_state_dims=env.n_state_dims,
        seed_ID=seed_id,
    )

    checkpoint_path = (
        model_dir
        / f"{agent_name}_UAV_Static_seed_{seed_id}.pt"
    )

    state_dict = load_state_dict_safely(checkpoint_path)

    agent._main_net.load_state_dict(state_dict)
    agent._target_net.load_state_dict(state_dict)
    agent.eval_mode()

    print(f"Loaded: {checkpoint_path}")

    return agent


def load_test_positions(
    file_path: Path,
    num_users: int,
) -> np.ndarray:
    """Load and validate fixed test topologies."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Fixed test-position file does not exist:\n{file_path}"
        )

    positions = np.load(file_path)

    if positions.ndim != 3:
        raise ValueError(
            f"Positions must have shape (trials, K, 2), "
            f"but received {positions.shape}."
        )

    if positions.shape[1:] != (num_users, 2):
        raise ValueError(
            f"Expected (*, {num_users}, 2), "
            f"but received {positions.shape}."
        )

    if positions.shape[0] < N_TEST_TOPOLOGIES:
        raise ValueError(
            f"{file_path} contains only {positions.shape[0]} "
            f"topologies, but {N_TEST_TOPOLOGIES} are requested."
        )

    return np.asarray(
        positions[:N_TEST_TOPOLOGIES],
        dtype=np.float64,
    )


# ============================================================
# 3. Evaluation functions
# ============================================================

def evaluate_terminal_min_rates(
    agent,
    env: UAV_Emergency_Env,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Run the trained policy for N=200 steps and return the
    terminal hard minimum rate for each test topology.
    """

    terminal_rates = np.empty(
        positions.shape[0],
        dtype=np.float64,
    )

    with torch.inference_mode():
        for trial_id, topology in enumerate(positions):
            state = env.reset(user_positions=topology)
            episode_done = False

            while not episode_done:
                action = agent.act_epsilon_greedy(
                    state,
                    0.0,
                )

                state, _, episode_done = env.step(action)

            if env.last_rates is None:
                raise RuntimeError(
                    "Environment did not generate terminal rates."
                )

            terminal_rates[trial_id] = float(
                np.min(env.last_rates)
            )

    return terminal_rates


def evaluate_physical_baselines(
    env: UAV_Emergency_Env,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate equal power and exact max-min allocation
    on exactly the same topologies.
    """

    equal_rates = np.empty(
        positions.shape[0],
        dtype=np.float64,
    )
    exact_rates = np.empty(
        positions.shape[0],
        dtype=np.float64,
    )

    for trial_id, topology in enumerate(positions):
        # reset() initializes equal power allocation.
        env.reset(user_positions=topology)

        if env.last_rates is None or env.channel_gains is None:
            raise RuntimeError(
                "Environment reset failed to calculate rates/gains."
            )

        equal_rates[trial_id] = float(
            np.min(env.last_rates)
        )

        _, _, exact_min_rate = exact_maxmin_allocation(
            channel_gains=env.channel_gains,
            total_power=env.P_total,
            bandwidth=env.B,
            noise_power=env.sigma2,
        )

        exact_rates[trial_id] = exact_min_rate

    return equal_rates, exact_rates


def calculate_gap_percent(
    achieved_rates: np.ndarray,
    exact_rates: np.ndarray,
) -> np.ndarray:
    """Calculate topology-wise relative optimality gap."""

    exact = np.asarray(exact_rates, dtype=np.float64)
    achieved = np.asarray(achieved_rates, dtype=np.float64)

    return (
        exact - achieved
    ) / np.maximum(exact, 1e-12) * 100.0


def summarize_learning_method(
    rates: np.ndarray,
    exact_rates: np.ndarray,
) -> dict[str, float]:
    """
    rates has shape:
        (number of seeds, number of topologies)
    """

    seed_means = np.mean(rates, axis=1)

    gaps = calculate_gap_percent(
        achieved_rates=rates,
        exact_rates=exact_rates[None, :],
    )

    ratios = (
        rates
        / np.maximum(exact_rates[None, :], 1e-12)
        * 100.0
    )

    return {
        "mean_rate": float(np.mean(seed_means)),
        "std_seed_means": float(
            np.std(seed_means, ddof=1)
        ),
        "std_all_values": float(np.std(rates)),
        "mean_gap": float(np.mean(gaps)),
        "mean_ratio": float(np.mean(ratios)),
    }


def summarize_baseline(
    rates: np.ndarray,
    exact_rates: np.ndarray,
) -> dict[str, float]:
    """Summarize a non-learning physical baseline."""

    gaps = calculate_gap_percent(
        achieved_rates=rates,
        exact_rates=exact_rates,
    )

    ratios = (
        rates
        / np.maximum(exact_rates, 1e-12)
        * 100.0
    )

    return {
        "mean_rate": float(np.mean(rates)),
        "std_seed_means": 0.0,
        "std_all_values": float(np.std(rates)),
        "mean_gap": float(np.mean(gaps)),
        "mean_ratio": float(np.mean(ratios)),
    }


# ============================================================
# 4. One configuration
# ============================================================

def evaluate_one_configuration(
    config: dict[str, Any],
) -> dict[str, Any]:

    label = config["label"]
    num_users = int(config["num_users"])
    height = float(config["height"])
    model_dir = Path(config["model_dir"])

    print("\n" + "=" * 80)
    print(f"Evaluating {label}")
    print(f"K = {num_users}, H = {height}")
    print(f"Model directory: {model_dir}")
    print("=" * 80)

    positions = load_test_positions(
        file_path=Path(config["positions_file"]),
        num_users=num_users,
    )

    # Evaluation environment.
    env = UAV_Emergency_Env(
        num_users=num_users,
        reward_type="Eval",
        use_softmin=True,
        use_db_norm=True,
    )

    # H must be assigned before reset().
    env.H = height

    equal_rates, exact_rates = evaluate_physical_baselines(
        env=env,
        positions=positions,
    )

    qsum_rates_all_seeds = []
    qmin_rates_all_seeds = []

    for seed_id in SEED_IDS:
        print(f"\n--- Seed ID {seed_id} ---")

        regular_agent = create_loaded_agent(
            agent_class=Regular_DDQN_Agent,
            env=env,
            model_dir=model_dir,
            agent_name="Regular",
            seed_id=seed_id,
        )

        qsum_rates = evaluate_terminal_min_rates(
            agent=regular_agent,
            env=env,
            positions=positions,
        )

        qsum_rates_all_seeds.append(qsum_rates)

        modified_agent = create_loaded_agent(
            agent_class=Modified_DDQN_Agent,
            env=env,
            model_dir=model_dir,
            agent_name="Modified",
            seed_id=seed_id,
        )

        qmin_rates = evaluate_terminal_min_rates(
            agent=modified_agent,
            env=env,
            positions=positions,
        )

        qmin_rates_all_seeds.append(qmin_rates)

        print(
            f"Seed {seed_id}: "
            f"Q-Sum={np.mean(qsum_rates):.6f} Mbps, "
            f"Q-Min={np.mean(qmin_rates):.6f} Mbps"
        )

    qsum_rates_all_seeds = np.asarray(
        qsum_rates_all_seeds,
        dtype=np.float64,
    )

    qmin_rates_all_seeds = np.asarray(
        qmin_rates_all_seeds,
        dtype=np.float64,
    )

    # Numerical consistency check.
    tolerance = 1e-8

    if np.any(
        qmin_rates_all_seeds
        >
        exact_rates[None, :] + tolerance
    ):
        print(
            "WARNING: Some Q-Min values exceed the exact "
            "reference. Check units, channel gains, or evaluation code."
        )

    summaries = {
        "Equal Allocation": summarize_baseline(
            equal_rates,
            exact_rates,
        ),
        "Sum-rate DDQN": summarize_learning_method(
            qsum_rates_all_seeds,
            exact_rates,
        ),
        "Proposed Q-Min": summarize_learning_method(
            qmin_rates_all_seeds,
            exact_rates,
        ),
        "Exact Max-Min": summarize_baseline(
            exact_rates,
            exact_rates,
        ),
    }

    print("\nSummary:")
    for method, values in summaries.items():
        print(
            f"{method:20s} | "
            f"rate={values['mean_rate']:.6f} Mbps | "
            f"gap={values['mean_gap']:.3f}% | "
            f"ratio={values['mean_ratio']:.3f}%"
        )

    return {
        "label": label,
        "x_value": config["x_value"],
        "num_users": num_users,
        "height": height,
        "equal_rates": equal_rates,
        "exact_rates": exact_rates,
        "qsum_rates": qsum_rates_all_seeds,
        "qmin_rates": qmin_rates_all_seeds,
        "summaries": summaries,
    }


# ============================================================
# 5. Sweep and save
# ============================================================

def save_csv(
    experiment_name: str,
    results: list[dict[str, Any]],
) -> None:

    csv_path = (
        OUTPUT_DIR
        / f"{experiment_name}_summary.csv"
    )

    fieldnames = [
        "experiment",
        "configuration",
        "x_value",
        "num_users",
        "height_m",
        "method",
        "mean_terminal_rate_mbps",
        "std_across_seed_means_mbps",
        "std_all_values_mbps",
        "mean_optimality_gap_percent",
        "mean_optimality_ratio_percent",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            for method, summary in result["summaries"].items():
                writer.writerow(
                    {
                        "experiment": experiment_name,
                        "configuration": result["label"],
                        "x_value": result["x_value"],
                        "num_users": result["num_users"],
                        "height_m": result["height"],
                        "method": method,
                        "mean_terminal_rate_mbps": (
                            summary["mean_rate"]
                        ),
                        "std_across_seed_means_mbps": (
                            summary["std_seed_means"]
                        ),
                        "std_all_values_mbps": (
                            summary["std_all_values"]
                        ),
                        "mean_optimality_gap_percent": (
                            summary["mean_gap"]
                        ),
                        "mean_optimality_ratio_percent": (
                            summary["mean_ratio"]
                        ),
                    }
                )

    print(f"CSV saved to: {csv_path}")


def save_npz(
    experiment_name: str,
    results: list[dict[str, Any]],
) -> None:

    npz_path = (
        OUTPUT_DIR
        / f"benchmark_{experiment_name}.npz"
    )

    np.savez_compressed(
        npz_path,
        labels=np.asarray(
            [result["label"] for result in results]
        ),
        x_values=np.asarray(
            [result["x_value"] for result in results],
            dtype=np.float64,
        ),
        equal_rates=np.stack(
            [result["equal_rates"] for result in results],
            axis=0,
        ),
        exact_rates=np.stack(
            [result["exact_rates"] for result in results],
            axis=0,
        ),
        qsum_rates=np.stack(
            [result["qsum_rates"] for result in results],
            axis=0,
        ),
        qmin_rates=np.stack(
            [result["qmin_rates"] for result in results],
            axis=0,
        ),
    )

    print(f"NPZ saved to: {npz_path}")


def run_sweep(
    experiment_name: str,
    configs: list[dict[str, Any]],
) -> None:

    results = []

    for config in configs:
        result = evaluate_one_configuration(config)
        results.append(result)

    save_csv(
        experiment_name=experiment_name,
        results=results,
    )

    save_npz(
        experiment_name=experiment_name,
        results=results,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Device: {DEVICE}")
    print(
        f"Number of test topologies: "
        f"{N_TEST_TOPOLOGIES}"
    )

    run_sweep(
        experiment_name="heights",
        configs=HEIGHT_CONFIGS,
    )

    run_sweep(
        experiment_name="users",
        configs=USER_CONFIGS,
    )

    print("\n" + "=" * 80)
    print("Exact benchmark evaluation completed.")
    print(f"Results directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()