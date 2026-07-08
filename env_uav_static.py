import math

import gym
from gym import spaces
import numpy as np


class UAV_Emergency_Env(gym.Env):
    def __init__(
        self,
        num_users=10,
        reward_type="Q-Min",
        use_softmin=True,
        use_db_norm=True,
    ):
        super(UAV_Emergency_Env, self).__init__()

        self.reward_type = reward_type
        self.use_softmin = use_softmin
        self.use_db_norm = use_db_norm

        self.K = num_users

        self.P_total = 1.0
        self.delta_p = 0.02

        self.H = 100.0
        self.B = 2e6
        self.sigma2 = 1e-14
        self.beta0 = 1e-4
        self.alpha = 3.5

        self.max_steps = 200
        self.current_step = 0

        self.action_space = spaces.Discrete(
            self.K * (self.K - 1) + 1
        )

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(4 * self.K,),
            dtype=np.float32,
        )

        self.n_actions = self.K * (self.K - 1) + 1
        self.n_state_dims = 4 * self.K

        self.user_positions = None

        self.uav_pos = np.array(
            [250.0, 250.0],
            dtype=np.float64,
        )

        self.current_power = None
        self.last_rates = None
        self.channel_gains = None

    def _get_normalized_gains(self):
        """
        Full 与 w/o Softmin：
            使用 dB 域归一化。

        w/o dB-Norm：
            直接输入原始线性信道增益。
        """
        if self.use_db_norm:
            gains_db = 10.0 * np.log10(
                self.channel_gains + 1e-20
            )

            normalized_gains = (
                gains_db + 150.0
            ) / 50.0

            return np.clip(
                normalized_gains,
                0.0,
                1.0,
            )

        # w/o dB-Norm
        return np.clip(
            self.channel_gains,
            0.0,
            1.0,
        )

    def _calculate_channel_gains(self):
        """
        根据当前用户位置计算信道增益。
        """
        self.channel_gains = np.zeros(
            self.K,
            dtype=np.float64,
        )

        for k in range(self.K):
            dist_2d = np.linalg.norm(
                self.uav_pos - self.user_positions[k]
            )

            distance_3d = math.sqrt(
                dist_2d**2 + self.H**2
            )

            self.channel_gains[k] = (
                self.beta0
                * distance_3d ** (-self.alpha)
            )

    def _calculate_rates(self):
        """
        根据功率和信道增益计算用户速率，单位为 Mbps。
        """
        rates = np.zeros(
            self.K,
            dtype=np.float64,
        )

        for k in range(self.K):
            snr = (
                self.current_power[k]
                * self.channel_gains[k]
            ) / self.sigma2

            rates[k] = (
                (self.B / self.K)
                * math.log2(1.0 + snr)
                / 1e6
            )

        return rates

    def _build_state(self):
        """
        构造状态：
            power + rate + bottleneck one-hot + channel gains
        """
        bottleneck_idx = int(
            np.argmin(self.last_rates)
        )

        bottleneck_onehot = np.zeros(
            self.K,
            dtype=np.float64,
        )

        bottleneck_onehot[bottleneck_idx] = 1.0

        normalized_gains = (
            self._get_normalized_gains()
        )

        state = np.concatenate(
            (
                self.current_power,
                self.last_rates,
                bottleneck_onehot,
                normalized_gains,
            )
        )

        return state

    def reset(self, user_positions=None):
        """
        重置环境。

        参数
        ----
        user_positions:
            None：
                随机生成用户位置，用于训练。

            shape=(K, 2) 的数组：
                使用指定用户位置，用于固定位置验证。
        """
        self.current_step = 0

        if user_positions is None:
            self.user_positions = np.random.uniform(
                low=0.0,
                high=500.0,
                size=(self.K, 2),
            )
        else:
            positions = np.asarray(
                user_positions,
                dtype=np.float64,
            )

            expected_shape = (self.K, 2)

            if positions.shape != expected_shape:
                raise ValueError(
                    f"user_positions 形状错误："
                    f"{positions.shape}，"
                    f"预期为：{expected_shape}"
                )

            if not np.all(np.isfinite(positions)):
                raise ValueError(
                    "user_positions 包含 NaN 或无穷值。"
                )

            # 必须复制，避免环境修改固定测试数据
            self.user_positions = positions.copy()

        self.current_power = np.ones(
            self.K,
            dtype=np.float64,
        ) * (self.P_total / self.K)

        self._calculate_channel_gains()

        self.last_rates = self._calculate_rates()

        return self._build_state()

    def step(self, action):
        self.current_step += 1

        if not self.action_space.contains(action):
            raise ValueError(
                f"非法动作：{action}"
            )

        if action < self.K * (self.K - 1):
            source_user = action // (self.K - 1)

            target_temp = action % (self.K - 1)

            target_user = (
                target_temp
                if target_temp < source_user
                else target_temp + 1
            )

            if (
                self.current_power[source_user]
                > self.delta_p + 0.001
            ):
                self.current_power[source_user] -= (
                    self.delta_p
                )

                self.current_power[target_user] += (
                    self.delta_p
                )

        rates = self._calculate_rates()

        self.last_rates = rates

        min_rate = float(np.min(rates))

        beta = 20.0

        smooth_min = min_rate - (
            1.0 / beta
        ) * np.log(
            np.sum(
                np.exp(
                    -beta * (rates - min_rate)
                )
            )
        )

        if self.reward_type == "Q-Sum":
            reward = float(np.sum(rates))

        elif self.reward_type == "Q-Min":
            if self.use_softmin:
                reward = float(smooth_min)
            else:
                reward = float(min_rate)

        elif self.reward_type == "Eval":
            # 所有算法统一使用真实最低速率测试
            reward = float(min_rate)

        else:
            raise ValueError(
                f"未知 reward_type："
                f"{self.reward_type}"
            )

        next_state = self._build_state()

        done = bool(
            self.current_step >= self.max_steps
        )

        return next_state, reward, done