import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 填入您消融实验最终收敛的 Max-Min Rate 数值 (Mbps)
# ==========================================
# 标签定义
variants = [
    "Standard DQN\n(Baseline)", 
    "w/o Softmin\nDense Reward", 
    "w/o dB-Domain\nNormalization", 
    "Proposed Q-Min\n(Full Method)"
]

# 模拟最终收敛的速率 (请替换为您实际跑出来的真实数据！)
converged_rates = [0.12, 0.18, 0.22, 0.41] 

# ==========================================
# 2. IEEE TWC 风格柱状图绘制
# ==========================================
plt.figure(figsize=(8, 5))

# 为不同的变体设置渐进颜色
colors = ['#E24A33', '#FBC15E', '#8EBA42', '#348ABD']
x_pos = np.arange(len(variants))

# 绘制柱状图，加入边缘颜色增强立体感
bars = plt.bar(x_pos, converged_rates, color=colors, edgecolor='black', linewidth=1.2, width=0.55)

# 在每个柱子上方标注具体数值 (TWC 非常喜欢这种严谨的数据展示)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
             f'{yval:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('Converged Max-Min Rate (Mbps)', fontsize=13)
plt.xticks(x_pos, variants, fontsize=11)
plt.ylim(0, max(converged_rates) * 1.3) # 留出顶部空间显示文字
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("Ablation_Study.pdf", format='pdf', dpi=300)
print("✅ 消融实验图绘制完成！请查看 Ablation_Study.pdf")
plt.show()