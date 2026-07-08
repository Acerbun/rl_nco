import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import os

plt.rcParams.update({
    "font.family": "serif",              # 使用衬线字体
    "font.serif": ["Times New Roman"],   # 强制使用 Times New Roman
    "font.size": 12,                     # 全局基础字号
    "axes.labelsize": 14,                # 坐标轴标签字号放大
    "legend.fontsize": 11,               # 图例字号适中
    "xtick.labelsize": 11,               # 刻度字号
    "ytick.labelsize": 11,
    "grid.alpha": 0.5,                   # 让网格线颜色变淡，不喧宾夺主
    "grid.linestyle": "--"               # 网格线用虚线
})


# 1. 物理参数严谨设定
B = 2e6              
P_total = 1.0        
sigma2 = 1e-14
beta0 = 1e-4
alpha = 3.5
H = 100.0   

# 【核心修正】：重新引入 200 的缩放因子，把 AI 的成绩还原为真实 Mbps
SCALE_FACTOR = 200.0

K_list = [5, 10, 15, 20]
folders = ["Models_user05", "Models_user10", "Models_user15", "Models_user20"]

# 2. 计算物理大锅饭基准
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
            p_k = P_total / K
            b_k = B / K
            snr = (p_k * g_k) / sigma2
            rates[k] = b_k * math.log2(1 + snr)
        min_rates.append(np.min(rates))
    return np.mean(min_rates) / 1e6

final_results = {"Equal": [], "Regular": [], "Modified": []}

print("="*40)
print("🚀 开始数据提取与绘制...")
print("="*40)

for idx, K in enumerate(K_list):
    eq_rate = calc_equal_allocation_rate(K)
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
        print(f"❌ 警告：K={K} 的数据包未找到！")
        final_results["Regular"].append(0.0)
        final_results["Modified"].append(0.0)
        continue
        
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    for agent in ["Regular", "Modified"]:
        agent_final_scores = []
        for seed in data.keys():
            # 【修复位置】：除以 SCALE_FACTOR，还原真实网速！
            raw_scores = data[seed][agent][:, 3] / SCALE_FACTOR
            converged_score = np.mean(raw_scores[-100:])
            agent_final_scores.append(converged_score)
        
        final_results[agent].append(np.mean(agent_final_scores))

# ================= 数据透视仪 =================
print("\n" + "="*40)
print("📊 还原后的真实收敛数值 (Mbps)：")
print(f"横坐标 (K值) : {K_list}")
print(f"🟪 紫线 (大锅饭) : {np.round(final_results['Equal'], 4)}")
print(f"🟥 红线 (Q-Sum)  : {np.round(final_results['Regular'], 4)}")
print(f"🟦 蓝线 (Q-Min)  : {np.round(final_results['Modified'], 4)}")
print("="*40 + "\n")

# 3. 开始 IEEE 顶刊级绘图
plt.figure(figsize=(9, 6))

plt.plot(K_list, final_results["Equal"], marker='s', markersize=8, linestyle='--', color="#988ED5", linewidth=2.5, label="Baseline: Equal Allocation")
plt.plot(K_list, final_results["Regular"], marker='o', markersize=8, color="#E24A33", linewidth=2.5, label="Standard DQN (Q-Sum)")
plt.plot(K_list, final_results["Modified"], marker='^', markersize=10, color="#348ABD", linewidth=3.0, label="Proposed Q-Min")

# plt.title(f"Max-Min Rate vs. Number of Users (H={H}m)", fontsize=16, fontweight='bold')
# plt.xlabel("Number of Users (K)", fontsize=14)
plt.xlabel(r"Number of Users $K$", fontsize=14)
plt.ylabel("Converged Max-Min Rate (Mbps)", fontsize=14)

plt.xticks(K_list, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc="upper right", fontsize=12)
plt.grid(True, linestyle='-.', alpha=0.7)

# plt.tight_layout()
# plt.savefig("Summary_Users_K_Fixed.png", dpi=300)
# print("🎉 神图绘制完毕！请查看 Summary_Users_K_Fixed.png")
# plt.show()-+*

plt.tight_layout()
plt.savefig('fig_users_trend.pdf', format='pdf', bbox_inches='tight')
plt.show()