import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# 导入你的环境和智能体
from env_uav_static import UAV_Emergency_Env
from agent import Regular_DDQN_Agent, Modified_DDQN_Agent

def extract_multiple_episodes_rates(agent, env, num_episodes=500):
    """
    运行超大规模多轮测试 (Monte Carlo Simulation)。
    500 轮 * 10 用户 = 5000 个数据点！大数定理将自动抹平锯齿，生成丝滑曲线。
    """
    all_rates = []
    for ep in range(num_episodes):
        state = env.reset()
        while True:
            action = agent.act_epsilon_greedy(state, 0.0)
            next_state, reward, done = env.step(action)
            if done:
                all_rates.extend(env.last_rates.tolist())
                break
            state = next_state
    return np.array(all_rates)

print("="*60)
print("🚀 开始运行超大规模蒙特卡洛测试 (500 Episodes)...")
print("⏳ 正在收集 5000 个个体网速样本，请耐心等待 10~20 秒...")
print("="*60)

K = 10  # 使用表现最好的典型拥挤场景 K=10
env_qsum = UAV_Emergency_Env(num_users=K, reward_type="Q-Sum")
env_qmin = UAV_Emergency_Env(num_users=K, reward_type="Q-Min")

# 提取基线 (Equal Allocation) 的速率 (跑 500 轮)
all_equal_rates = []
for _ in range(500):
    env_qsum.reset()
    all_equal_rates.extend(env_qsum.last_rates.tolist())
rates_equal = np.array(all_equal_rates)
print(f"✅ Baseline (Equal Allocation) 提取完成！")

n_actions = env_qsum.n_actions
n_state_dims = env_qsum.n_state_dims

# 你的模型路径 (K=10)
qsum_model_path = "Models_user10/UAV_Static/Regular_UAV_Static_seed_0.pt"
qmin_model_path = "Models_user10/UAV_Static/Modified_UAV_Static_seed_0.pt"

rates_q_sum = None
rates_q_min = None

try:
    agent_qsum = Regular_DDQN_Agent(n_actions, n_state_dims, seed_ID=0)
    agent_qsum._main_net.load_state_dict(torch.load(qsum_model_path, map_location=torch.device('cpu')))
    agent_qsum._main_net.eval()
    rates_q_sum = extract_multiple_episodes_rates(agent_qsum, env_qsum, num_episodes=500)
    print(f"✅ Q-Sum 模型 (10用户) 提取完成！")
except Exception as e:
    print(f"❌ Q-Sum 模型加载失败: {e}")

try:
    agent_qmin = Modified_DDQN_Agent(n_actions, n_state_dims, seed_ID=0)
    agent_qmin._main_net.load_state_dict(torch.load(qmin_model_path, map_location=torch.device('cpu')))
    agent_qmin._main_net.eval()
    rates_q_min = extract_multiple_episodes_rates(agent_qmin, env_qmin, num_episodes=500)
    print(f"✅ Q-Min 模型 (10用户) 提取完成！")
except Exception as e:
    print(f"❌ Q-Min 模型加载失败: {e}")

# ==========================================
# 开始绘制极致丝滑的顶刊级 CDF 图
# ==========================================
if rates_q_sum is not None and rates_q_min is not None:
    print("\n📊 5000 级数据点准备就绪，开始绘制 CDF 神图...")
    
    def compute_cdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    x_eq, y_eq = compute_cdf(rates_equal)
    x_qsum, y_qsum = compute_cdf(rates_q_sum)
    x_qmin, y_qmin = compute_cdf(rates_q_min)

    plt.figure(figsize=(7, 5))

    # 因为有 5000 个密集点，使用 plt.plot 会比 plt.step 更加浑然天成
    plt.plot(x_eq, y_eq, label='Equal Allocation', color='#988ED5', linestyle='--', linewidth=2.5)
    plt.plot(x_qsum, y_qsum, label='Sum-rate DDQN', color='#E24A33', linewidth=2.5)
    plt.plot(x_qmin, y_qmin, label='Proposed Q-Min', color='#348ABD', linewidth=3.0)

    # 动态高亮危险区域
    danger_zone_limit = max(0.1, np.min(rates_equal) * 1.5)
    plt.axvspan(0, danger_zone_limit, color='gray', alpha=0.15, label='Starvation Zone (Edge Users)')

    plt.xlabel('Individual User Throughput (Mbps)', fontsize=13)
    plt.ylabel('Cumulative Distribution Function (CDF)', fontsize=13)
    
    # 将 X 轴上限卡在合理的视觉范围，避免少数土豪用户拉长 X 轴
    plt.xlim(0, max(np.percentile(rates_q_sum, 99), np.percentile(rates_q_min, 99)) * 1.1)
    plt.ylim(0, 1.05)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig("CDF_Fairness_K10_Smooth.pdf", format='pdf', dpi=300)
    print("🎉 极致丝滑的 CDF 图绘制完成！请查看 CDF_Fairness_K10_Smooth.pdf")
    plt.show()