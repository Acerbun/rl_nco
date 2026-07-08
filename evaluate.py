import numpy as np

from settings import EVALUATION_STATES


def compute_Qs_over_random_states(
    agent,
    memory_valid,
):
    """
    计算固定验证状态上的平均最大 Q 值。
    """
    states = []

    for i in range(EVALUATION_STATES):
        state, _ = (
            memory_valid.get_current_and_next_state(i)
        )

        states.append(state)

    states = np.asarray(states)

    q_values = (
        agent.main_net_predict(states)
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    return float(
        np.mean(
            np.max(q_values, axis=1)
        )
    )


def compute_scores(
    agent,
    env,
    n_trials,
    random=False,
    fixed_user_positions=None,
):
    """
    评估智能体。

    fixed_user_positions 为 None：
        每个回合随机生成位置。

    fixed_user_positions 不为 None：
        所有智能体严格使用指定的同一批用户位置。

    fixed_user_positions 的形状必须为：
        (n_trials, num_users, 2)
    """
    if n_trials <= 0:
        raise ValueError(
            "n_trials 必须大于 0。"
        )

    if fixed_user_positions is not None:
        fixed_user_positions = np.asarray(
            fixed_user_positions,
            dtype=np.float64,
        )

        expected_shape = (
            n_trials,
            env.K,
            2,
        )

        if fixed_user_positions.shape != expected_shape:
            raise ValueError(
                f"固定测试位置形状错误："
                f"{fixed_user_positions.shape}，"
                f"预期为：{expected_shape}"
            )

    scores_all_trials = []

    for trial_id in range(n_trials):
        score = 0.0

        if fixed_user_positions is None:
            state = env.reset()
        else:
            state = env.reset(
                user_positions=(
                    fixed_user_positions[trial_id]
                )
            )

        while True:
            epsilon = 1.0 if random else 0.0

            action = agent.act_epsilon_greedy(
                state,
                epsilon,
            )

            next_state, reward, episode_done = (
                env.step(action)
            )

            score += float(reward)

            if episode_done:
                break

            state = next_state

        scores_all_trials.append(score)

    return float(
        np.mean(scores_all_trials)
    )