import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*50)
print("📊 正在从训练日志提取真实数据并绘制消融实验图...")
print("="*50)

# ==========================================
# 1. 定义数据提取逻辑
# ==========================================
SCALE_FACTOR = 200.0  # 你的物理换算系数：将单回合总分转化为单步 Mbps

def get_converged_rate(pkl_path, agent_type):
    if not os.path.exists(pkl_path):
        print(f"❌ 找不到文件: {pkl_path}")
        return 0.0
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    # 提取所有 seed 的数据并取平均
    seeds = list(data.keys())
    final_scores = []
    for seed in seeds:
        # 提取最后 100 回合的平均值作为稳态收敛值
        raw_scores = data[seed][agent_type][:, 3] / SCALE_FACTOR
        converged_score = np.mean(raw_scores[-100:])
        final_scores.append(converged_score)
    
    return np.mean(final_scores)

# ==========================================
# 2. 读取四个模型的真实收敛网速
# ==========================================
# K=10 的完整模型日志路径 (包含 Q-Sum 和 完全体 Q-Min)
path_full = "Models_user10/UAV_Static/UAV_Static_metrics_all_seeds.pkl"
# 残缺版 A 日志路径
path_softmin = "Models_Softmin/UAV_Static/UAV_Static_metrics_all_seeds.pkl"
# 残缺版 B 日志路径
path_dbnorm = "Models_dB-Norm/UAV_Static/UAV_Static_metrics_all_seeds.pkl"

print("🔍 开始读取各模型的收敛速率...")
# Regular 对应 Q-Sum，Modified 对应你写的 Q-Min 或其变体
rate_q_sum      = get_converged_rate(path_full, "Regular")
print(f"✅ Standard DQN (Q-Sum) : {rate_q_sum:.4f} Mbps")

rate_no_softmin = get_converged_rate(path_softmin, "Modified")
print(f"✅ w/o Softmin 变体     : {rate_no_softmin:.4f} Mbps")

rate_no_dbnorm  = get_converged_rate(path_dbnorm, "Modified")
print(f"✅ w/o dB-Norm 变体     : {rate_no_dbnorm:.4f} Mbps")

rate_q_min_full = get_converged_rate(path_full, "Modified")
print(f"✅ Proposed Q-Min (完全体): {rate_q_min_full:.4f} Mbps")

rates = [rate_q_sum, rate_no_softmin, rate_no_dbnorm, rate_q_min_full]

# ==========================================
# 3. IEEE TWC 风格绘图设置
# ==========================================
labels = [
    'Standard DQN\n(Baseline)', 
    'Q-Min\n(w/o Softmin)', 
    'Q-Min\n(w/o dB-Norm)', 
    'Proposed Q-Min\n(Full Method)'
]

x = np.arange(len(labels))
width = 0.55

fig, ax = plt.subplots(figsize=(8, 5.5))

colors = ['#E24A33', '#FBC15E', '#8EBA42', '#348ABD']
hatches = ['//', '\\\\', 'xx', '']

# 绘制柱状图，加上黑色边框增加立体感
bars = ax.bar(x, rates, width, color=colors, edgecolor='black', linewidth=1.5)

# 填充底纹
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# 在上方标注网速数值
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), 
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Converged Max-Min Rate (Mbps)', fontsize=13)
ax.set_title('Ablation Study of Core Innovations ($K=10, H=100m$)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11, fontweight='500')

ax.set_ylim(0, max(rates) * 1.3) 
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('Fig7_Ablation_Study.pdf', format='pdf', dpi=300)
print("\n🎉 图 7 (消融实验图) 绘制完成！请查看 Fig7_Ablation_Study.pdf")
plt.show()