import os
import random

import numpy as np
import torch


# Windows/OpenMP 兼容设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# ============================================================
# 1. 环境设置
# ============================================================

ENVIRONMENT_NAME = "UAV_Static"

NUM_USERS = 10


# ============================================================
# 2. 消融实验版本
# ============================================================
#
# 每次运行前只修改这里。
#
# 可选值：
#   "full"
#   "wo_softmin"
#   "wo_dbnorm"
#
EXPERIMENT_VARIANT = "wo_dbnorm"

VALID_EXPERIMENT_VARIANTS = {
    "full",
    "wo_softmin",
    "wo_dbnorm",
}

if EXPERIMENT_VARIANT not in VALID_EXPERIMENT_VARIANTS:
    raise ValueError(
        f"未知实验版本：{EXPERIMENT_VARIANT}。"
        f"允许值为：{VALID_EXPERIMENT_VARIANTS}"
    )


# ============================================================
# 3. 固定验证用户位置
# ============================================================
#
# 三个消融实验必须使用相同路径、相同随机种子和相同用户数。
# 不要在不同实验之间修改或删除这个文件。
#
FIXED_VALIDATION_POSITIONS_PATH = (
    f"fixed_data/"
    f"uav_static_validation_positions_user{NUM_USERS}.npy"
)

FIXED_VALIDATION_POSITIONS_SEED = 20260708


# ============================================================
# 4. 经验池与设备
# ============================================================

MEMORY_TYPE = "Uniform"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 5. 状态与训练频率
# ============================================================

HISTORY_LENGTH = 1
FRAME_SKIP = 1
TRAIN_FREQUENCY = 1


# ============================================================
# 6. 强化学习超参数
# ============================================================

REWARD_DISCOUNT = 0.99

INITIAL_EXPLORE_STEPS = 2000

TRAIN_STEPS = 200000

TARGET_NET_SYNC_FREQUENCY = 1000

EVALUATION_FREQUENCY = 2000

# 为了先单独检查“测试位置是否相同”，暂时保留原来的 5。
# 后续再单独讨论是否增加到 50 或更多。
EVALUATION_TRIALS = 5

EVALUATION_TRIALS_TEST = 50

REPLAY_MEMORY_SIZE = 50000

RANDOM_SEEDS = [123, 321, 456]

EVALUATION_STATES = 200

LEARNING_RATE = 1e-4


assert EVALUATION_FREQUENCY % TRAIN_FREQUENCY == 0


def set_random_seed(rand_seed: int) -> None:
    """
    设置训练随机种子。
    """
    os.environ["PYTHONHASHSEED"] = str(rand_seed)

    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(rand_seed)
        torch.cuda.manual_seed_all(rand_seed)

    print("<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>")
    print(f"Random seed: {rand_seed}")