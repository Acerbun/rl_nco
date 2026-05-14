import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import os



# 1. 物理参数与换算系数
B = 2e6              
P_total = 1.0        
sigma2 = 1e-14
beta0 = 1e-4
alpha = 3.5
H = 100.0            
SCALE_FACTOR = 200.0

# 【极其重要】：你跑了几个用户的数据，这里就改成几！
# 比如跑 K=5 的数据，这里就改成 5，紫线会自动对齐到最真实的物理水平！
K = 10 

# =============== 【新增：EMA 平滑函数】 ===============
def smooth_ema(scalars, weight=0.85):
    """
    EMA (指数移动平均) 平滑函数。
    weight 越接近 1，曲线越平滑；越接近 0，越保留原始震荡。
    0.85 是学术界黄金平滑系数。
    """
    if len(scalars) == 0: return np.array([])
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)
# ====================================================


# 2. 计算纯视距(LoS)下的平均分配基准线（无瑞利衰落）
def calc_equal_allocation_rate(K, num_episodes=2000):
    min_rates = []
    for _ in range(num_episodes):
        user_positions = np.random.uniform(0, 500, size=(K, 2))
        uav_pos = np.array([250.0, 250.0])
        rates = np.zeros(K)
        for k in range(K):
            dist_2d = np.linalg.norm(uav_pos - user_positions[k])
            d_k = math.sqrt(dist_2d**2 + H**2)
            g_k = beta0 * (d_k ** -alpha)
            h_k = g_k # 【已关闭瑞利衰落】
            p_k = P_total / K
            b_k = B / K
            snr = (p_k * h_k) / sigma2
            rates[k] = b_k * math.log2(1 + snr)
        min_rates.append(np.min(rates))
    return np.mean(min_rates) / 1e6

print(f"正在计算 K={K} 下的 Equal Allocation 理论基准线...")
equal_rate = calc_equal_allocation_rate(K)
print(f"理论基准线计算完毕: {equal_rate:.4f} Mbps")

# 3. 读取你跑完的数据
data_path = f"Models_user{K:02d}/UAV_Static/UAV_Static_metrics_all_seeds.pkl"

if not os.path.exists(data_path):
    print(f"找不到文件: {data_path}。请检查你的文件夹路径！")
else:
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    seeds = list(data.keys())
    agents = ["Regular", "Modified"]
    scores = {agent: [] for agent in agents}

    for seed in seeds:
        for agent in agents:
            # 提取数据并直接换算为单步 Mbps
            scaled_scores = data[seed][agent][:, 3] / SCALE_FACTOR
            scores[agent].append(scaled_scores)

    # 4. 开始画验证图
    plt.figure(figsize=(8, 6))
    colors = {"Regular": "#E24A33", "Modified": "#348ABD"}
    labels = {"Regular": "Standard DQN (Q-Sum)", "Modified": "Proposed Q-Min"}

    # 画 RL 收敛曲线
    for agent in agents:
        agent_scores = np.array(scores[agent])
        mean_scores = np.mean(agent_scores, axis=0)
        std_scores = np.std(agent_scores, axis=0)
        steps = data[seeds[0]][agent][:, 0]
        
        # === 【核心学术微雕：执行数据平滑】 ===
        smoothed_mean = smooth_ema(mean_scores, weight=0.85)
        smoothed_std = smooth_ema(std_scores, weight=0.85)
        # =======================================
        
        # 1. 画出极其隐晦的底层原始数据（证明数据真实性，不是造假）
        plt.plot(steps, mean_scores, color=colors[agent], alpha=0.15, linewidth=1)
        
        # 2. 画平滑后的主线（视觉焦点）
        plt.plot(steps, smoothed_mean, label=labels[agent], color=colors[agent], linewidth=2.5)
        
        # 3. 画平滑后的方差阴影带
        plt.fill_between(steps, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, color=colors[agent], alpha=0.2)

    # 画出 Equal Allocation 的水平基准线
    plt.axhline(y=equal_rate, color="#988ED5", linestyle='--', linewidth=2.5, 
                label=f"Baseline: Equal Allocation")

    # 标题动态显示 K 的数量
    plt.title(f"Convergence Validation at K={K} (LoS Channel)", fontsize=15, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=13)
    plt.ylabel("Max-Min Rate (Mbps)", fontsize=13)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    # 根据 K 值自动生成保存的名字，防止误覆盖
    save_name = f"test_h{int(H):03d}_user{K:02d}.png"
    plt.savefig(save_name, dpi=300)
    print(f"画图成功！请在左侧目录查看 {save_name}")
    plt.show()