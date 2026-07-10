"""
Evaluation utilities for Q-Min and comparison methods.

This module provides:
1. Average maximum Q-value evaluation on validation states.
2. Original trajectory-score evaluation.
3. Terminal hard minimum-rate evaluation.
4. Equal-allocation and exact max-min physical baselines.

The terminal and physical-baseline functions evaluate all methods
on exactly the same fixed user topologies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from exact_maxmin import exact_maxmin_allocation
from settings import EVALUATION_STATES


def _validate_fixed_user_positions(
    fixed_user_positions: np.ndarray,
    num_users: int,
    expected_trials: int | None = None,
) -> np.ndarray:
    """
    Validate and return fixed user positions as a float64 array.

    Parameters
    ----------
    fixed_user_positions:
        User coordinates with shape (n_trials, K, 2).
    num_users:
        Number of users K expected by the environment.
    expected_trials:
        Expected number of test topologies. If None, the number
        is inferred from the first array dimension.

    Returns
    -------
    np.ndarray
        Validated position array with dtype float64.

    Raises
    ------
    ValueError
        If the shape is invalid or the array contains non-finite values.
    """
    positions = np.asarray(
        fixed_user_positions,
        dtype=np.float64,
    )

    if positions.ndim != 3:
        raise ValueError(
            "fixed_user_positions 必须是三维数组，"
            "形状为 (n_trials, K, 2)。"
        )

    if positions.shape[1:] != (num_users, 2):
        raise ValueError(
            "固定测试位置形状错误："
            f"{positions.shape}。"
            f"除第一维外，预期形状为 ({num_users}, 2)。"
        )

    if expected_trials is not None:
        expected_shape = (
            expected_trials,
            num_users,
            2,
        )

        if positions.shape != expected_shape:
            raise ValueError(
                "固定测试位置形状错误："
                f"{positions.shape}，"
                f"预期为：{expected_shape}。"
            )

    if positions.shape[0] <= 0:
        raise ValueError(
            "fixed_user_positions 至少需要包含一组测试拓扑。"
        )

    if not np.all(np.isfinite(positions)):
        raise ValueError(
            "fixed_user_positions 包含 NaN 或无穷值。"
        )

    return positions


def _check_environment_attributes(
    env: Any,
    required_attributes: tuple[str, ...],
) -> None:
    """
    Check whether the environment contains required attributes.
    """
    missing_attributes = [
        name
        for name in required_attributes
        if not hasattr(env, name)
    ]

    if missing_attributes:
        missing_text = ", ".join(missing_attributes)

        raise AttributeError(
            "环境缺少评价所需属性："
            f"{missing_text}。"
        )


def compute_Qs_over_random_states(
    agent: Any,
    memory_valid: Any,
) -> float:
    """
    计算固定验证状态上的平均最大 Q 值。

    Parameters
    ----------
    agent:
        DDQN agent.
    memory_valid:
        Validation replay memory containing fixed states.

    Returns
    -------
    float
        Mean of the maximum predicted Q-value over validation states.
    """
    if EVALUATION_STATES <= 0:
        raise ValueError(
            "EVALUATION_STATES 必须大于 0。"
        )

    states = []

    for index in range(EVALUATION_STATES):
        state, _ = (
            memory_valid.get_current_and_next_state(index)
        )
        states.append(state)

    states_array = np.asarray(
        states,
        dtype=np.float32,
    )

    agent.eval_mode()

    q_values = (
        agent.main_net_predict(states_array)
        .detach()
        .cpu()
        .numpy()
    )

    if q_values.ndim != 2:
        raise RuntimeError(
            "Q 网络输出必须是二维数组，"
            "形状应为 (n_states, n_actions)，"
            f"实际形状为 {q_values.shape}。"
        )

    max_q_values = np.max(
        q_values,
        axis=1,
    )

    return float(np.mean(max_q_values))


def compute_scores(
    agent: Any,
    env: Any,
    n_trials: int,
    random: bool = False,
    fixed_user_positions: np.ndarray | None = None,
) -> float:
    """
    使用原有方式评估智能体的轨迹累计奖励。

    该函数保留用于兼容原训练与验证代码。它返回每个 episode
    中所有 step reward 的累计值，再对多个 trial 求平均。

    Parameters
    ----------
    agent:
        DDQN agent.
    env:
        UAV emergency environment.
    n_trials:
        Number of evaluation episodes.
    random:
        If True, use completely random actions. Otherwise use the
        greedy action from the trained policy.
    fixed_user_positions:
        Optional position array with shape (n_trials, K, 2).
        If None, the environment randomly generates positions.

    Returns
    -------
    float
        Mean cumulative reward over all evaluation trials.
    """
    if n_trials <= 0:
        raise ValueError(
            "n_trials 必须大于 0。"
        )

    _check_environment_attributes(
        env,
        required_attributes=("K",),
    )

    positions = None

    if fixed_user_positions is not None:
        positions = _validate_fixed_user_positions(
            fixed_user_positions=fixed_user_positions,
            num_users=env.K,
            expected_trials=n_trials,
        )

    agent.eval_mode()

    scores_all_trials: list[float] = []

    for trial_id in range(n_trials):
        score = 0.0

        if positions is None:
            state = env.reset()
        else:
            state = env.reset(
                user_positions=positions[trial_id],
            )

        while True:
            policy_epsilon = (
                1.0
                if random
                else 0.0
            )

            action = agent.act_epsilon_greedy(
                state,
                policy_epsilon,
            )

            next_state, reward, episode_done = env.step(
                action
            )

            score += float(reward)

            if episode_done:
                break

            state = next_state

        scores_all_trials.append(score)

    return float(
        np.mean(
            np.asarray(
                scores_all_trials,
                dtype=np.float64,
            )
        )
    )


def compute_terminal_min_rates(
    agent: Any,
    env: Any,
    fixed_user_positions: np.ndarray,
) -> np.ndarray:
    """
    Evaluate terminal hard minimum rates of a trained policy.

    For every fixed topology, the agent executes a complete greedy
    allocation episode. Only the hard minimum user rate at the final
    step is recorded.

    This function does not retrain or modify the model.

    Parameters
    ----------
    agent:
        Loaded DDQN/Q-Min agent.
    env:
        UAV emergency environment.
    fixed_user_positions:
        User positions with shape (n_trials, K, 2).

    Returns
    -------
    np.ndarray
        Terminal hard minimum rate for each topology, in Mbps.
        Shape: (n_trials,).
    """
    _check_environment_attributes(
        env,
        required_attributes=(
            "K",
            "last_rates",
        ),
    )

    positions = _validate_fixed_user_positions(
        fixed_user_positions=fixed_user_positions,
        num_users=env.K,
    )

    agent.eval_mode()

    terminal_min_rates = np.empty(
        positions.shape[0],
        dtype=np.float64,
    )

    for trial_id, topology in enumerate(positions):
        state = env.reset(
            user_positions=topology,
        )

        while True:
            # Greedy policy: epsilon = 0.
            action = agent.act_epsilon_greedy(
                state,
                0.0,
            )

            next_state, _, episode_done = env.step(
                action
            )

            if episode_done:
                break

            state = next_state

        final_rates = np.asarray(
            env.last_rates,
            dtype=np.float64,
        )

        if final_rates.shape != (env.K,):
            raise RuntimeError(
                "环境中的最终速率形状错误："
                f"{final_rates.shape}，"
                f"预期为 ({env.K},)。"
            )

        if not np.all(np.isfinite(final_rates)):
            raise RuntimeError(
                f"第 {trial_id} 组拓扑的最终速率"
                "包含 NaN 或无穷值。"
            )

        terminal_min_rates[trial_id] = float(
            np.min(final_rates)
        )

    return terminal_min_rates


def compute_physical_baselines(
    env: Any,
    fixed_user_positions: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Evaluate equal allocation and exact max-min allocation.

    Both physical baselines are evaluated on exactly the same user
    topologies supplied to the learning-based methods.

    Parameters
    ----------
    env:
        UAV emergency environment.
    fixed_user_positions:
        User positions with shape (n_trials, K, 2).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:

        - "equal_terminal_min_rate":
          Equal-allocation minimum rate for every topology,
          shape (n_trials,).

        - "exact_terminal_min_rate":
          Exact max-min rate for every topology,
          shape (n_trials,).

        - "equal_power":
          Equal power vector for every topology,
          shape (n_trials, K).

        - "exact_power":
          Exact max-min power vector for every topology,
          shape (n_trials, K).

        - "exact_user_rates":
          Individual rates under exact allocation,
          shape (n_trials, K).
    """
    _check_environment_attributes(
        env,
        required_attributes=(
            "K",
            "P_total",
            "B",
            "sigma2",
            "channel_gains",
        ),
    )

    positions = _validate_fixed_user_positions(
        fixed_user_positions=fixed_user_positions,
        num_users=env.K,
    )

    n_trials = positions.shape[0]

    equal_min_rates = np.empty(
        n_trials,
        dtype=np.float64,
    )

    exact_min_rates = np.empty(
        n_trials,
        dtype=np.float64,
    )

    equal_power_vectors = np.empty(
        (n_trials, env.K),
        dtype=np.float64,
    )

    exact_power_vectors = np.empty(
        (n_trials, env.K),
        dtype=np.float64,
    )

    exact_user_rates = np.empty(
        (n_trials, env.K),
        dtype=np.float64,
    )

    equal_power = np.full(
        env.K,
        env.P_total / env.K,
        dtype=np.float64,
    )

    for trial_id, topology in enumerate(positions):
        # reset() calculates the channel gains for this topology.
        env.reset(
            user_positions=topology,
        )

        channel_gains = np.asarray(
            env.channel_gains,
            dtype=np.float64,
        ).copy()

        if channel_gains.shape != (env.K,):
            raise RuntimeError(
                "环境中的信道增益形状错误："
                f"{channel_gains.shape}，"
                f"预期为 ({env.K},)。"
            )

        if not np.all(np.isfinite(channel_gains)):
            raise RuntimeError(
                f"第 {trial_id} 组拓扑的信道增益"
                "包含 NaN 或无穷值。"
            )

        if np.any(channel_gains <= 0.0):
            raise RuntimeError(
                f"第 {trial_id} 组拓扑包含非正信道增益。"
            )

        # ---------------------------------------------------------
        # Equal power allocation
        # ---------------------------------------------------------
        equal_snr = (
            equal_power
            * channel_gains
            / env.sigma2
        )

        equal_rates = (
            env.B
            / env.K
            * np.log2(1.0 + equal_snr)
            / 1e6
        )

        if not np.all(np.isfinite(equal_rates)):
            raise RuntimeError(
                f"第 {trial_id} 组拓扑的等功率速率"
                "包含 NaN 或无穷值。"
            )

        equal_power_vectors[trial_id] = equal_power
        equal_min_rates[trial_id] = float(
            np.min(equal_rates)
        )

        # ---------------------------------------------------------
        # Exact max-min allocation
        # ---------------------------------------------------------
        (
            exact_power,
            exact_rates,
            exact_min_rate,
        ) = exact_maxmin_allocation(
            channel_gains=channel_gains,
            total_power=float(env.P_total),
            bandwidth=float(env.B),
            noise_power=float(env.sigma2),
        )

        exact_power = np.asarray(
            exact_power,
            dtype=np.float64,
        )

        exact_rates = np.asarray(
            exact_rates,
            dtype=np.float64,
        )

        if exact_power.shape != (env.K,):
            raise RuntimeError(
                "精确功率向量形状错误："
                f"{exact_power.shape}，"
                f"预期为 ({env.K},)。"
            )

        if exact_rates.shape != (env.K,):
            raise RuntimeError(
                "精确用户速率形状错误："
                f"{exact_rates.shape}，"
                f"预期为 ({env.K},)。"
            )

        if not np.isclose(
            np.sum(exact_power),
            env.P_total,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"第 {trial_id} 组拓扑的精确功率分配"
                "不满足总功率约束。"
            )

        if np.any(exact_power < -1e-12):
            raise RuntimeError(
                f"第 {trial_id} 组拓扑的精确功率分配"
                "包含负值。"
            )

        exact_power_vectors[trial_id] = exact_power
        exact_user_rates[trial_id] = exact_rates
        exact_min_rates[trial_id] = float(
            exact_min_rate
        )

    return {
        "equal_terminal_min_rate": equal_min_rates,
        "exact_terminal_min_rate": exact_min_rates,
        "equal_power": equal_power_vectors,
        "exact_power": exact_power_vectors,
        "exact_user_rates": exact_user_rates,
    }


def optimality_gap_percent(
    achieved_rates: np.ndarray,
    exact_rates: np.ndarray,
) -> np.ndarray:
    """
    Compute the relative optimality gap for each topology.

    Gap = (R_exact - R_achieved) / R_exact * 100%.

    Parameters
    ----------
    achieved_rates:
        Terminal minimum rates achieved by a tested method.
    exact_rates:
        Exact max-min rates on the same topologies.

    Returns
    -------
    np.ndarray
        Optimality-gap percentages with the same shape as the inputs.
    """
    achieved = np.asarray(
        achieved_rates,
        dtype=np.float64,
    )

    exact = np.asarray(
        exact_rates,
        dtype=np.float64,
    )

    if achieved.shape != exact.shape:
        raise ValueError(
            "achieved_rates 与 exact_rates 的形状必须一致，"
            f"当前分别为 {achieved.shape} 和 {exact.shape}。"
        )

    if achieved.size == 0:
        raise ValueError(
            "评价数组不能为空。"
        )

    if not np.all(np.isfinite(achieved)):
        raise ValueError(
            "achieved_rates 包含 NaN 或无穷值。"
        )

    if not np.all(np.isfinite(exact)):
        raise ValueError(
            "exact_rates 包含 NaN 或无穷值。"
        )

    if np.any(exact <= 0.0):
        raise ValueError(
            "exact_rates 必须全部大于 0。"
        )

    gap = (
        exact - achieved
    ) / exact * 100.0

    # Small negative gaps may occur because of floating-point errors.
    return np.maximum(
        gap,
        0.0,
    )


def optimality_ratio_percent(
    achieved_rates: np.ndarray,
    exact_rates: np.ndarray,
) -> np.ndarray:
    """
    Compute the percentage of exact performance achieved.

    Ratio = R_achieved / R_exact * 100%.
    """
    achieved = np.asarray(
        achieved_rates,
        dtype=np.float64,
    )

    exact = np.asarray(
        exact_rates,
        dtype=np.float64,
    )

    if achieved.shape != exact.shape:
        raise ValueError(
            "achieved_rates 与 exact_rates 的形状必须一致。"
        )

    if achieved.size == 0:
        raise ValueError(
            "评价数组不能为空。"
        )

    if not np.all(np.isfinite(achieved)):
        raise ValueError(
            "achieved_rates 包含 NaN 或无穷值。"
        )

    if not np.all(np.isfinite(exact)):
        raise ValueError(
            "exact_rates 包含 NaN 或无穷值。"
        )

    if np.any(exact <= 0.0):
        raise ValueError(
            "exact_rates 必须全部大于 0。"
        )

    return achieved / exact * 100.0