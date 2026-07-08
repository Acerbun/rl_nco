import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# 1. 实验结果路径
# ============================================================

PATH_FULL = Path(
    "Models_user10/UAV_Static/UAV_Static_metrics_all_seeds.pkl"
)

PATH_WO_SOFTMIN = Path(
    "Models_Softmin/UAV_Static/UAV_Static_metrics_all_seeds.pkl"
)

PATH_WO_DBNORM = Path(
    "Models_dB-Norm/UAV_Static/UAV_Static_metrics_all_seeds.pkl"
)


# ============================================================
# 2. 指标计算参数
# ============================================================

# 每个评估回合包含 200 个环境步。
# 保存的 avg_score 是 200 步奖励之和，
# 因此除以 200 得到平均最低速率，单位为 Mbps。
SCALE_FACTOR = 200.0

# 使用最后 10 个评估点计算最终性能
FINAL_WINDOW = 10

# 收敛曲线移动平均窗口
SMOOTH_WINDOW = 5

# 连续多少个评估点满足条件才认为收敛
CONVERGENCE_PATIENCE = 5

# 收敛阈值：达到最终性能的 95%
CONVERGENCE_THRESHOLD_RATIO = 0.95


# ============================================================
# 3. 实验配置
# ============================================================

EXPERIMENTS = [
    {
        "name": "Standard DQN (Baseline)",
        "path": PATH_FULL,
        "agent_type": "Regular",
    },
    {
        "name": "Q-Min (w/o Softmin)",
        "path": PATH_WO_SOFTMIN,
        "agent_type": "Modified",
    },
    {
        "name": "Q-Min (w/o dB-Norm)",
        "path": PATH_WO_DBNORM,
        "agent_type": "Modified",
    },
    {
        "name": "Proposed Q-Min (Full)",
        "path": PATH_FULL,
        "agent_type": "Modified",
    },
]


def load_pickle_file(pkl_path: Path) -> Dict:
    """
    加载 pkl 实验结果文件。
    """
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"找不到结果文件：{pkl_path.resolve()}"
        )

    with pkl_path.open("rb") as file:
        data = pickle.load(file)

    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(
            f"结果文件为空或格式错误：{pkl_path}"
        )

    return data


def extract_all_seed_curves(
    pkl_path: Path,
    agent_type: str,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    提取所有随机种子的学习曲线。

    main.py 中每一行指标应为：
        [training_step, loss, average_Q, average_score]

    返回：
        common_steps:
            所有随机种子共同拥有的训练步。

        rate_curves:
            shape = (num_seeds, num_evaluation_points)

        seed_ids:
            随机种子编号。
    """
    data = load_pickle_file(pkl_path)

    seed_curves = []
    seed_ids = []

    for seed_id in sorted(data.keys()):
        seed_data = data[seed_id]

        if agent_type not in seed_data:
            raise KeyError(
                f"{pkl_path} 的 seed={seed_id} 中"
                f"不存在算法类型 {agent_type}"
            )

        metrics = np.asarray(
            seed_data[agent_type],
            dtype=np.float64,
        )

        if metrics.ndim != 2 or metrics.shape[1] < 4:
            raise ValueError(
                f"seed={seed_id}, agent={agent_type} "
                f"的指标形状错误：{metrics.shape}"
            )

        training_steps = metrics[:, 0]
        average_rates = metrics[:, 3] / SCALE_FACTOR

        valid_mask = (
            np.isfinite(training_steps)
            & np.isfinite(average_rates)
        )

        training_steps = training_steps[valid_mask]
        average_rates = average_rates[valid_mask]

        if len(training_steps) == 0:
            raise ValueError(
                f"seed={seed_id}, agent={agent_type} "
                "没有有效数据。"
            )

        # 按训练步排序
        order = np.argsort(training_steps)

        training_steps = training_steps[order]
        average_rates = average_rates[order]

        # 删除重复训练步
        unique_steps, unique_indices = np.unique(
            training_steps,
            return_index=True,
        )

        average_rates = average_rates[unique_indices]

        seed_curves.append(
            (unique_steps, average_rates)
        )

        seed_ids.append(seed_id)

    # 找到所有随机种子共同拥有的评估步
    common_steps = seed_curves[0][0]

    for steps, _ in seed_curves[1:]:
        common_steps = np.intersect1d(
            common_steps,
            steps,
        )

    if len(common_steps) == 0:
        raise ValueError(
            f"{pkl_path} 的不同种子没有共同评估步。"
        )

    aligned_curves = []

    for steps, rates in seed_curves:
        step_to_rate = {
            int(step): float(rate)
            for step, rate in zip(steps, rates)
        }

        aligned_rate = np.array(
            [
                step_to_rate[int(step)]
                for step in common_steps
            ],
            dtype=np.float64,
        )

        aligned_curves.append(aligned_rate)

    return (
        common_steps.astype(np.int64),
        np.vstack(aligned_curves),
        seed_ids,
    )


def moving_average(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    计算一维移动平均。
    """
    if values.ndim != 1:
        raise ValueError(
            "moving_average 只接受一维数组。"
        )

    actual_window = min(
        max(1, window),
        len(values),
    )

    kernel = np.ones(
        actual_window,
        dtype=np.float64,
    ) / actual_window

    return np.convolve(
        values,
        kernel,
        mode="valid",
    )


def find_convergence_step(
    training_steps: np.ndarray,
    mean_curve: np.ndarray,
    final_rate: float,
) -> Optional[int]:
    """
    查找收敛训练步。

    收敛定义：
        移动平均性能连续 CONVERGENCE_PATIENCE 个评估点
        不低于最终性能的 CONVERGENCE_THRESHOLD_RATIO。
    """
    if len(training_steps) != len(mean_curve):
        raise ValueError(
            "training_steps 和 mean_curve 长度不一致。"
        )

    actual_window = min(
        max(1, SMOOTH_WINDOW),
        len(mean_curve),
    )

    smoothed_curve = moving_average(
        mean_curve,
        actual_window,
    )

    # 移动平均对应窗口最后一个训练步
    smoothed_steps = training_steps[
        actual_window - 1:
    ]

    target_rate = (
        final_rate
        * CONVERGENCE_THRESHOLD_RATIO
    )

    actual_patience = min(
        max(1, CONVERGENCE_PATIENCE),
        len(smoothed_curve),
    )

    for index in range(
        len(smoothed_curve)
        - actual_patience
        + 1
    ):
        segment = smoothed_curve[
            index:index + actual_patience
        ]

        if np.all(segment >= target_rate):
            return int(smoothed_steps[index])

    return None


def analyze_experiment(
    pkl_path: Path,
    agent_type: str,
) -> Dict:
    """
    计算一个算法的最终性能、收敛步数和标准差。
    """
    training_steps, curves, seed_ids = (
        extract_all_seed_curves(
            pkl_path=pkl_path,
            agent_type=agent_type,
        )
    )

    final_window = min(
        FINAL_WINDOW,
        curves.shape[1],
    )

    # 每个随机种子最后若干评估点的平均值
    seed_final_rates = np.mean(
        curves[:, -final_window:],
        axis=1,
    )

    # 所有随机种子的最终平均性能
    final_rate = float(
        np.mean(seed_final_rates)
    )

    # 不同随机种子最终性能的样本标准差
    if len(seed_final_rates) > 1:
        final_std = float(
            np.std(
                seed_final_rates,
                ddof=1,
            )
        )
    else:
        final_std = 0.0

    # 所有随机种子的平均学习曲线
    mean_curve = np.mean(
        curves,
        axis=0,
    )

    convergence_step = find_convergence_step(
        training_steps=training_steps,
        mean_curve=mean_curve,
        final_rate=final_rate,
    )

    return {
        "final_rate": final_rate,
        "final_std": final_std,
        "convergence_step": convergence_step,
        "num_seeds": len(seed_ids),
        "seed_final_rates": seed_final_rates,
    }


def format_convergence_step(
    value: Optional[int],
) -> str:
    """
    格式化收敛步数。
    """
    if value is None:
        return "Not Converged"

    return f"{value:,}"


def main() -> None:
    print("=" * 94)
    print(
        "📊 正在为 IEEE TWC 提取消融实验表格数据..."
    )
    print("=" * 94)

    results = []

    for experiment in EXPERIMENTS:
        try:
            result = analyze_experiment(
                pkl_path=experiment["path"],
                agent_type=experiment["agent_type"],
            )

            result["name"] = experiment["name"]
            results.append(result)

        except Exception as error:
            print(
                f"\n[错误] {experiment['name']}：{error}"
            )

            results.append(
                {
                    "name": experiment["name"],
                    "final_rate": None,
                    "final_std": None,
                    "convergence_step": None,
                    "num_seeds": 0,
                    "seed_final_rates": np.array([]),
                }
            )

    header = (
        f"{'Algorithm Variant':<27} | "
        f"{'Final Rate (Mbps)':<17} | "
        f"{'Convergence Steps':<18} | "
        f"{'Std. Dev.':<10}"
    )

    print("\n" + header)
    print("-" * len(header))

    for result in results:
        if result["final_rate"] is None:
            rate_text = "N/A"
            std_text = "N/A"
        else:
            rate_text = f"{result['final_rate']:.4f}"
            std_text = f"{result['final_std']:.4f}"

        convergence_text = format_convergence_step(
            result["convergence_step"]
        )

        print(
            f"{result['name']:<27} | "
            f"{rate_text:<17} | "
            f"{convergence_text:<18} | "
            f"{std_text:<10}"
        )

    print("=" * len(header))

    print("\n指标说明：")
    print(
        f"1. Final Rate：所有随机种子最后 "
        f"{FINAL_WINDOW} 个评估点的平均最低速率。"
    )
    print(
        "2. Convergence Steps：平均曲线首次稳定达到"
        f"最终性能 {CONVERGENCE_THRESHOLD_RATIO * 100:.0f}% "
        f"所需的训练步数。"
    )
    print(
        "3. Std. Dev.：不同随机种子最终性能之间的"
        "样本标准差，越小表示稳定性越好。"
    )

    print("\n各随机种子的最终速率：")

    for result in results:
        rates = result["seed_final_rates"]

        if len(rates) == 0:
            continue

        rates_text = ", ".join(
            f"{rate:.4f}"
            for rate in rates
        )

        print(
            f"- {result['name']}: {rates_text}"
        )


if __name__ == "__main__":
    main()