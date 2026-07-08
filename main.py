import os
import pickle

import numpy as np
from tqdm import trange

from settings import *
from agent import (
    Regular_DDQN_Agent,
    Modified_DDQN_Agent,
)
from replay_memory import (
    Prioritized_Replay_Memory_Gym,
    Uniform_Replay_Memory_Gym,
)
from evaluate import (
    compute_Qs_over_random_states,
    compute_scores,
)
from env_uav_static import UAV_Emergency_Env
from fixed_positions import load_or_create_positions


class LinearSchedule:
    def __init__(
        self,
        initial_val,
        final_val,
        schedule_timesteps,
    ):
        self.initial_val = initial_val
        self.final_val = final_val
        self.schedule_timesteps = schedule_timesteps

    def value(self, timestep):
        fraction = max(
            min(
                float(timestep)
                / self.schedule_timesteps,
                1.0,
            ),
            0.0,
        )

        return (
            self.initial_val
            + fraction
            * (
                self.final_val
                - self.initial_val
            )
        )


def get_experiment_flags():
    """
    根据 settings.py 中的实验名称，
    返回 Softmin 和 dB-Norm 开关。
    """
    if EXPERIMENT_VARIANT == "full":
        return True, True

    if EXPERIMENT_VARIANT == "wo_softmin":
        return False, True

    if EXPERIMENT_VARIANT == "wo_dbnorm":
        return True, False

    raise ValueError(
        f"未知实验版本："
        f"{EXPERIMENT_VARIANT}"
    )


def create_memory(capacity):
    """
    根据设置创建经验回放池。
    """
    if MEMORY_TYPE == "Prioritized":
        return Prioritized_Replay_Memory_Gym(
            capacity
        )

    return Uniform_Replay_Memory_Gym(
        capacity
    )


def train(seed_id):
    use_softmin, use_db_norm = (
        get_experiment_flags()
    )

    print("=" * 80)
    print(
        f"Experiment variant : "
        f"{EXPERIMENT_VARIANT}"
    )
    print(
        f"Use Softmin        : "
        f"{use_softmin}"
    )
    print(
        f"Use dB-Norm        : "
        f"{use_db_norm}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # 1. 固定验证位置
    # --------------------------------------------------------

    validation_positions = (
        load_or_create_positions(
            file_path=(
                FIXED_VALIDATION_POSITIONS_PATH
            ),
            num_scenarios=EVALUATION_TRIALS,
            num_users=NUM_USERS,
            random_seed=(
                FIXED_VALIDATION_POSITIONS_SEED
            ),
        )
    )

    # --------------------------------------------------------
    # 2. 初始化训练环境和训练经验池
    # --------------------------------------------------------

    envs_train = {}
    memories_train = {}

    for agent_type in [
        "Regular",
        "Modified",
    ]:
        if agent_type == "Regular":
            reward_type = "Q-Sum"
        else:
            reward_type = "Q-Min"

        envs_train[agent_type] = (
            UAV_Emergency_Env(
                num_users=NUM_USERS,
                reward_type=reward_type,
                use_softmin=use_softmin,
                use_db_norm=use_db_norm,
            )
        )

        memories_train[agent_type] = (
            create_memory(
                REPLAY_MEMORY_SIZE
            )
        )

    # --------------------------------------------------------
    # 3. 初始化验证环境
    # --------------------------------------------------------

    env_valid = UAV_Emergency_Env(
        num_users=NUM_USERS,
        reward_type="Eval",
        use_softmin=use_softmin,
        use_db_norm=use_db_norm,
    )

    memory_valid = create_memory(
        EVALUATION_STATES
    )

    # --------------------------------------------------------
    # 4. 初始化智能体
    # --------------------------------------------------------

    agents = {}

    agents["Regular"] = Regular_DDQN_Agent(
        n_actions=(
            envs_train["Regular"].n_actions
        ),
        n_state_dims=(
            envs_train["Regular"].n_state_dims
        ),
        seed_ID=seed_id,
    )

    agents["Modified"] = Modified_DDQN_Agent(
        n_actions=(
            envs_train["Modified"].n_actions
        ),
        n_state_dims=(
            envs_train["Modified"].n_state_dims
        ),
        seed_ID=seed_id,
    )

    # --------------------------------------------------------
    # 5. 生成验证状态经验池
    # --------------------------------------------------------

    print(
        f"[{ENVIRONMENT_NAME}] "
        f"Generate validation states..."
    )

    state = env_valid.reset(
        user_positions=validation_positions[0]
    )

    for _ in range(EVALUATION_STATES):
        action = agents[
            "Regular"
        ].act_epsilon_greedy(
            state,
            1.0,
        )

        next_state, _, episode_done = (
            env_valid.step(action)
        )

        memory_valid.add(
            state,
            1,
            0.0,
            episode_done,
        )

        if episode_done:
            state = env_valid.reset(
                user_positions=(
                    validation_positions[0]
                )
            )
        else:
            state = next_state

    # --------------------------------------------------------
    # 6. 初始化训练状态
    # --------------------------------------------------------

    print(
        f"[{ENVIRONMENT_NAME}] "
        f"Start training..."
    )

    states = {}
    metrics = {}
    highest_scores = {}

    for agent_type in agents:
        states[agent_type] = (
            envs_train[agent_type].reset()
        )

        metrics[agent_type] = []

        highest_scores[agent_type] = -np.inf

    policy_epsilon = LinearSchedule(
        initial_val=1.0,
        final_val=0.01,
        schedule_timesteps=int(
            TRAIN_STEPS * 0.8
        ),
    )

    priority_ImpSamp_beta = LinearSchedule(
        initial_val=0.4,
        final_val=1.0,
        schedule_timesteps=TRAIN_STEPS,
    )

    # --------------------------------------------------------
    # 7. 初始验证
    # --------------------------------------------------------

    for agent_type, agent in agents.items():
        agent.eval_mode()

        avg_Q = compute_Qs_over_random_states(
            agent,
            memory_valid,
        )

        avg_score = compute_scores(
            agent=agent,
            env=env_valid,
            n_trials=EVALUATION_TRIALS,
            fixed_user_positions=(
                validation_positions
            ),
        )

        metrics[agent_type].append(
            [
                0,
                0,
                avg_Q,
                avg_score,
            ]
        )

        print(
            "Initial: "
            f"[{agent_type}] "
            f"Q: {avg_Q:.4f}; "
            f"Score: {avg_score:.4f}; "
            f"Historic highest: "
            f"{highest_scores[agent_type]:.4f}"
        )

        if (
            avg_score
            >= highest_scores[agent_type]
        ):
            highest_scores[agent_type] = (
                avg_score
            )

    # --------------------------------------------------------
    # 8. 正式训练
    # --------------------------------------------------------

    total_steps = (
        INITIAL_EXPLORE_STEPS
        + TRAIN_STEPS
    )

    for i in trange(
        1,
        total_steps + 1,
    ):
        for agent_type, agent in agents.items():
            agent.train_mode()

            epsilon = policy_epsilon.value(
                i
                - 1
                - INITIAL_EXPLORE_STEPS
            )

            action = (
                agent.act_epsilon_greedy(
                    states[agent_type],
                    epsilon,
                )
            )

            next_state, reward, episode_done = (
                envs_train[agent_type].step(
                    action
                )
            )

            memories_train[agent_type].add(
                states[agent_type],
                action,
                reward,
                episode_done,
            )

            if episode_done:
                states[agent_type] = (
                    envs_train[
                        agent_type
                    ].reset()
                )
            else:
                states[agent_type] = (
                    next_state
                )

            if i <= INITIAL_EXPLORE_STEPS:
                continue

            training_step = (
                i - INITIAL_EXPLORE_STEPS
            )

            # 网络更新
            if (
                training_step
                % TRAIN_FREQUENCY
                == 0
            ):
                loss = agent.train(
                    memories_train[agent_type],
                    priority_ImpSamp_beta.value(
                        training_step
                    ),
                )

            # 目标网络同步
            if (
                training_step
                % TARGET_NET_SYNC_FREQUENCY
                == 0
            ):
                agent.sync_target_network()

            # 固定用户位置验证
            if (
                training_step
                % EVALUATION_FREQUENCY
                == 0
            ):
                agent.eval_mode()

                avg_Q = (
                    compute_Qs_over_random_states(
                        agent,
                        memory_valid,
                    )
                )

                avg_score = compute_scores(
                    agent=agent,
                    env=env_valid,
                    n_trials=(
                        EVALUATION_TRIALS
                    ),
                    fixed_user_positions=(
                        validation_positions
                    ),
                )

                metrics[agent_type].append(
                    [
                        int(
                            training_step
                            / TRAIN_FREQUENCY
                        ),
                        loss,
                        avg_Q,
                        avg_score,
                    ]
                )

                print(
                    f"[{agent_type}] "
                    f"Q_loss: {loss:.4f}; "
                    f"Q: {avg_Q:.4f}; "
                    f"Score: {avg_score:.4f}; "
                    f"Historic highest: "
                    f"{highest_scores[agent_type]:.4f}"
                )

                if (
                    avg_score
                    >= highest_scores[agent_type]
                ):
                    print(
                        f"[{agent_type}] "
                        f"Reached highest score!"
                    )

                    highest_scores[agent_type] = (
                        avg_score
                    )

                    agent.save_trained_net()

    print(
        "############## "
        f"Finished training on "
        f"{ENVIRONMENT_NAME}, "
        f"variant={EXPERIMENT_VARIANT}, "
        f"seed id={seed_id} "
        "##############"
    )

    for agent_type, values in metrics.items():
        metrics[agent_type] = np.asarray(
            values
        )

    return metrics


if __name__ == "__main__":
    metrics_all_seeds = {}

    os.makedirs(
        f"Models/{ENVIRONMENT_NAME}",
        exist_ok=True,
    )

    print("=" * 80)
    print(
        f"Running variant: "
        f"{EXPERIMENT_VARIANT}"
    )
    print(
        f"Fixed validation positions: "
        f"{FIXED_VALIDATION_POSITIONS_PATH}"
    )
    print("=" * 80)

    for seed_id, random_seed in enumerate(
        RANDOM_SEEDS
    ):
        set_random_seed(random_seed)

        metrics = train(
            seed_id=seed_id
        )

        metrics_all_seeds[seed_id] = (
            metrics
        )

    metrics_path = (
        f"Models/{ENVIRONMENT_NAME}/"
        f"{ENVIRONMENT_NAME}_metrics_all_seeds.pkl"
    )

    with open(metrics_path, "wb") as file:
        pickle.dump(
            metrics_all_seeds,
            file,
        )

    print(
        f"Metrics saved to: "
        f"{metrics_path}"
    )

    print("Script finished!")