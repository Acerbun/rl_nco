import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# 1. 物理参数严谨设定
B = 2e6              
P_total = 1.0        
sigma2 = 1e-14
beta0 = 1e-4
alpha = 3.5

# 【核心修改】：高度实验中，K 通常固定为 10
K = 10   
SCALE_FACTOR = 200.0

# 高度变量列表
H_list = [50.0, 100.0, 150.0, 200.0]
# ⚠️ 请确保这里的文件夹名字与你电脑里的完全一致！
folders = ["Models_h50", "Models_h100", "Models_h150", "Models_h200"]

# 2. 计算物理大锅饭基准（注意：现在 H 是变量，K 是常量）
def calc_equal_allocation_rate(H_val, num_episodes=2000):
    min_rates = []
    for _ in range(num_episodes):
        user_positions = np.random.uniform(0, 500, size=(K, 2))
        uav_pos = np.array([250.0, 250.0])
        rates = np.zeros(K)
        for k in range(K):
            dist_2d = np.linalg.norm(uav_pos - user_positions[k])
            # 这里的 H_val 是随着循环传进来的动态高度
            d_k = math.sqrt(dist_2d**2 + H_val**2) 
            g_k = beta0 * (d_k ** -alpha)
            p_k = P_total / K
            b_k = B / K
            snr = (p_k * g_k) / sigma2
            rates[k] = b_k * math.log2(1 + snr)
        min_rates.append(np.min(rates))
    return np.mean(min_rates) / 1e6

final_results = {"Equal": [], "Regular": [], "Modified": []}

print("="*40)
print("🚀 开始提取高度 H 的数据与绘制...")
print("="*40)

for idx, current_H in enumerate(H_list):
    # 动态计算当前高度下的紫线基准
    eq_rate = calc_equal_allocation_rate(current_H)
    final_results["Equal"].append(eq_rate)
    
    folder = folders[idx]
    path_option_1 = os.path.join(folder, "UAV_Static", "UAV_Static_metrics_all_seeds.pkl")
    path_option_2 = os.path.join(folder, "UAV_Static_metrics_all_seeds.pkl")
    
    data_path = None
    if os.path.exists(path_option_1):
        data_path = path_option_1
    elif os.path.exists(path_option_2):
        data_path = path_option_2
        
    if data_path is None:
        print(f"❌ 警告：H={current_H} 的数据包未找到！文件夹名是对的吗？")
        final_results["Regular"].append(0.0)
        final_results["Modified"].append(0.0)
        continue
        
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    for agent in ["Regular", "Modified"]:
        agent_final_scores = []
        for seed in data.keys():
            # 依然使用 200 进行完美还原
            raw_scores = data[seed][agent][:, 3] / SCALE_FACTOR
            converged_score = np.mean(raw_scores[-100:])
            agent_final_scores.append(converged_score)
        
        final_results[agent].append(np.mean(agent_final_scores))

# ================= 数据透视仪 =================
print("\n" + "="*40)
print("📊 还原后的真实收敛数值 (Mbps)：")
print(f"横坐标 (H高度): {H_list}")
print(f"🟪 紫线 (基准) : {np.round(final_results['Equal'], 4)}")
print(f"🟥 红线 (Q-Sum): {np.round(final_results['Regular'], 4)}")
print(f"🟦 蓝线 (Q-Min): {np.round(final_results['Modified'], 4)}")
print("="*40 + "\n")

# 3. 开始 IEEE 顶刊级绘图
plt.figure(figsize=(9, 6))

plt.plot(H_list, final_results["Equal"], marker='s', markersize=8, linestyle='--', color="#988ED5", linewidth=2.5, label="Baseline: Equal Allocation")
plt.plot(H_list, final_results["Regular"], marker='o', markersize=8, color="#E24A33", linewidth=2.5, label="Standard DQN (Q-Sum)")
plt.plot(H_list, final_results["Modified"], marker='^', markersize=10, color="#348ABD", linewidth=3.0, label="Proposed Q-Min")

plt.title(f"Max-Min Rate vs. UAV Height (K={K})", fontsize=16, fontweight='bold')
plt.xlabel("UAV Height H (meters)", fontsize=14)
plt.ylabel("Converged Max-Min Rate (Mbps)", fontsize=14)

plt.xticks(H_list, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc="upper right", fontsize=12)
plt.grid(True, linestyle='-.', alpha=0.7)

plt.tight_layout()
plt.savefig("Summary_Heights_H.png", dpi=300)
print("🎉 高度宏观神图绘制完毕！请查看 Summary_Heights_H.png")
plt.show()