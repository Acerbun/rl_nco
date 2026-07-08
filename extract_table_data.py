import pickle
import numpy as np
import os

print("="*60)
print("📊 正在为 IEEE TWC 提取终极消融实验表格数据...")
print("="*60)

SCALE_FACTOR = 200.0  # 物理换算系数

def extract_metrics(pkl_path, agent_type, threshold_ratio=0.95, early_phase=5000):
    if not os.path.exists(pkl_path):
        return None, None, None
        
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    # 提取第一组 seed 的数据进行分析
    seed_0 = list(data.keys())[0]
    raw_scores = data[seed_0][agent_type][:, 3] / SCALE_FACTOR
    
    # 1. 计算最终收敛网速 (最后 100 轮平均)
    final_rate = np.mean(raw_scores[-100:])
    
    # 如果最终网速极低（如 Baseline），视为不收敛
    if final_rate < 0.3:
        return final_rate, "Fails to Converge", "N/A"
        
    # 2. 计算收敛回合数 (Episodes to Converge)
    # 使用滑动窗口平滑曲线，防止单次毛刺干扰
    window = 100
    smoothed = np.convolve(raw_scores, np.ones(window)/window, mode='valid')
    
    # 找到第一次达到 [最终网速 * 95%] 的回合数
    target_threshold = final_rate * threshold_ratio
    converge_idx = np.where(smoothed >= target_threshold)[0]
    
    if len(converge_idx) > 0:
        episodes_to_converge = converge_idx[0] + window
    else:
        episodes_to_converge = len(raw_scores)
        
    # 3. 计算早期震荡标准差 (Training Fluctuation)
    # 提取前 5000 步的数据算方差，方差越大说明前期越像无头苍蝇
    early_std = np.std(raw_scores[:early_phase])
    
    return final_rate, episodes_to_converge, early_std

# ==========================================
# 填入你的日志路径
# ==========================================
path_full = "Models_user10/UAV_Static/UAV_Static_metrics_all_seeds.pkl"
path_softmin = "Models_Softmin/UAV_Static/UAV_Static_metrics_all_seeds.pkl"
path_dbnorm = "Models_dB-Norm/UAV_Static/UAV_Static_metrics_all_seeds.pkl"

print(f"{'Algorithm Variant':<25} | {'Final Rate':<10} | {'Episodes':<10} | {'Std. Dev. (Fluctuation)'}")
print("-" * 75)

# 1. Baseline (Q-Sum)
rate, ep, std = extract_metrics(path_full, "Regular")
print(f"{'Standard DQN (Baseline)':<25} | {rate:<10.3f} | {str(ep):<10} | {str(std)}")

# 2. w/o Softmin
rate, ep, std = extract_metrics(path_softmin, "Modified")
std_str = f"{std:.4f}" if isinstance(std, float) else "N/A"
print(f"{'Q-Min (w/o Softmin)':<25} | {rate:<10.3f} | {str(ep):<10} | {std_str}")

# 3. w/o dB-Norm
rate, ep, std = extract_metrics(path_dbnorm, "Modified")
std_str = f"{std:.4f}" if isinstance(std, float) else "N/A"
print(f"{'Q-Min (w/o dB-Norm)':<25} | {rate:<10.3f} | {str(ep):<10} | {std_str}")

# 4. Full Method
rate, ep, std = extract_metrics(path_full, "Modified")
std_str = f"{std:.4f}" if isinstance(std, float) else "N/A"
print(f"{'Proposed Q-Min (Full)':<25} | {rate:<10.3f} | {str(ep):<10} | {std_str}")
print("=" * 75)