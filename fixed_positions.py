from pathlib import Path
import hashlib
import numpy as np


def positions_hash(positions: np.ndarray) -> str:
    """
    计算用户位置数组的 SHA-256，用于确认不同实验是否使用同一批位置。
    """
    contiguous_array = np.ascontiguousarray(positions)
    return hashlib.sha256(contiguous_array.tobytes()).hexdigest()


def load_or_create_positions(
    file_path: str,
    num_scenarios: int,
    num_users: int,
    random_seed: int = 20260708,
    area_min: float = 0.0,
    area_max: float = 500.0,
) -> np.ndarray:
    """
    加载固定用户位置。

    如果文件不存在，则使用独立随机数生成器创建并保存。
    使用独立生成器不会改变训练代码中的 NumPy 全局随机状态。

    返回形状：
        (num_scenarios, num_users, 2)
    """
    path = Path(file_path)

    expected_shape = (num_scenarios, num_users, 2)

    if path.exists():
        positions = np.load(path)

        if positions.shape != expected_shape:
            raise ValueError(
                f"固定位置文件形状错误：{positions.shape}，"
                f"预期形状为：{expected_shape}。"
                f"\n请删除旧文件后重新运行：{path}"
            )

        if not np.all(np.isfinite(positions)):
            raise ValueError(
                f"固定位置文件包含 NaN 或无穷值：{path}"
            )

        print(f"[Fixed positions] Loaded from: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(random_seed)

        positions = rng.uniform(
            low=area_min,
            high=area_max,
            size=expected_shape,
        )

        np.save(path, positions)

        print(f"[Fixed positions] Created at: {path}")

    print(f"[Fixed positions] Shape : {positions.shape}")
    print(f"[Fixed positions] SHA256: {positions_hash(positions)}")

    return positions