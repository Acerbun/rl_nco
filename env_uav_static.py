import gym
from gym import spaces
import numpy as np
import math

class UAV_Emergency_Env(gym.Env):
    def __init__(self, num_users=10, reward_type="Q-Min"):
        super(UAV_Emergency_Env, self).__init__()
        self.reward_type = reward_type 
        
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

        self.action_space = spaces.Discrete(self.K * (self.K - 1) + 1)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4 * self.K,), dtype=np.float32)

        self.n_actions = self.K * (self.K - 1) + 1 
        self.n_state_dims = 4 * self.K 

        self.user_positions = None
        self.uav_pos = np.array([250.0, 250.0]) 
        self.current_power = None
        self.last_rates = None
        self.channel_gains = None 

    # =============== 【新增核心辅助函数：dB归一化】 ===============
    def _get_normalized_gains(self):
        # # 将微小的信道增益转化为对数域 (dB)，防止神经网络输入层数值崩溃
        # # 1e-20 是防止 log10(0) 的安全垫
        # gains_db = 10.0 * np.log10(self.channel_gains + 1e-20)
        # # 在 alpha=3.5 环境下，gains_db 大约在 -140dB 到 -100dB 之间
        # # 我们用 (dB + 150) / 50 将其完美映射到 [0, 1] 之间！
        # normalized_gains = (gains_db + 150.0) / 50.0 
        # return np.clip(normalized_gains, 0.0, 1.0)

        return np.clip(self.channel_gains, 0.0, 1.0) #dB-Norm

    # ==========================================================

    def reset(self):
        self.current_step = 0
        self.user_positions = np.random.uniform(0, 500, size=(self.K, 2))
        self.current_power = np.ones(self.K) * (self.P_total / self.K)
        
        self.channel_gains = np.zeros(self.K)
        for k in range(self.K):
            dist_2d = np.linalg.norm(self.uav_pos - self.user_positions[k])
            d_k = math.sqrt(dist_2d**2 + self.H**2)
            self.channel_gains[k] = self.beta0 * (d_k ** -self.alpha)
            
        self.last_rates = np.zeros(self.K)
        for k in range(self.K):
            snr = (self.current_power[k] * self.channel_gains[k]) / self.sigma2
            self.last_rates[k] = (self.B / self.K) * math.log2(1 + snr) / 1e6
            
        bottleneck_idx = np.argmin(self.last_rates)
        bottleneck_onehot = np.zeros(self.K)
        bottleneck_onehot[bottleneck_idx] = 1.0
        
        # 【修改位置 1】：使用 dB 归一化替代暴力的 1e13 线性放大
        normalized_gains = self._get_normalized_gains()
        return np.concatenate((self.current_power, self.last_rates, bottleneck_onehot, normalized_gains))

    def step(self, action):
        self.current_step += 1

        if action < self.K * (self.K - 1):
            i = action // (self.K - 1)
            j_temp = action % (self.K - 1)
            j = j_temp if j_temp < i else j_temp + 1

            if self.current_power[i] > self.delta_p + 0.001:
                self.current_power[i] -= self.delta_p
                self.current_power[j] += self.delta_p

        rates = np.zeros(self.K)
        for k in range(self.K):
            h_k = self.channel_gains[k]
            snr = (self.current_power[k] * h_k) / self.sigma2
            rates[k] = (self.B / self.K) * math.log2(1 + snr) / 1e6

        self.last_rates = rates

        # ---------------- 4. 计算奖励 ----------------
        min_rate = np.min(rates)
        beta = 20.0 
        smooth_min = min_rate - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (rates - min_rate))))

        # =============== 【修改位置 2】：取消所有乘数放大，回归本源 ===============
        # 绝不让奖励超过 5.0，保护神经网络不会发生 Q-value 爆炸！

        if self.reward_type == "Q-Sum":
            reward = float(np.sum(rates))      # 范围约在 1.0 ~ 3.0，安全
        elif self.reward_type == "Q-Min":
            reward = float(smooth_min)         # 范围约在 0.1 ~ 0.5，绝对安全且平滑，原版
            # reward = float(min_rate)           # Softmin
        elif self.reward_type == "Eval":
            reward = float(min_rate)
        else:
            reward = float(min_rate)
        # ======================================================================

        bottleneck_idx = np.argmin(rates)
        bottleneck_onehot = np.zeros(self.K)
        bottleneck_onehot[bottleneck_idx] = 1.0

        # 【修改位置 3】：使用 dB 归一化
        normalized_gains = self._get_normalized_gains()
        next_state = np.concatenate((self.current_power, self.last_rates, bottleneck_onehot, normalized_gains))
        done = bool(self.current_step >= self.max_steps)

        return next_state, reward, done