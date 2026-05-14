import numpy as np
import math
from env_uav_static import UAV_Emergency_Env

print("正在启动真实环境引擎，进行 1000 次蒙特卡洛仿真对决...")

# 1. 实例化你最真实的 RL 环境
env = UAV_Emergency_Env(num_users=10)

min_rates_equal = []

for _ in range(1000):
    # 重置环境（生成随机地图和位置）
    env.reset()
    
    # 【核心操作：强制接管上帝视角】
    # 我们不让 RL 智能体发号施令，而是强制把环境里所有用户的功率强行“平均分配”
    env.current_power = np.ones(env.K) * (env.P_total / env.K)
    
    # 随便传一个动作（比如0）进去，仅仅是为了触发环境内部的 step() 走一遍香农公式
    env.step(0)
    
    # 绕开所有花里胡哨的 Reward 包装，直接从底层提取绝对真实的最低速率
    real_min_rate = np.min(env.last_rates)
    min_rates_equal.append(real_min_rate)

# 计算平均值并转换为 Mbps
true_baseline = np.mean(min_rates_equal) / 1e6

print(f"\n==================================================")
print(f"破案结果：环境内部【绝对真实】的 Equal Allocation 是: {true_baseline:.4f} Mbps")
print(f"==================================================\n")