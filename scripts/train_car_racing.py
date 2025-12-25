import os
import sys
import random


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from scripts.car_racing.trainer import train_gym


if __name__ == "__main__":
    train_gym(
        n_envs=4,
        frame_stack=4,
        render=False,
        network="per noisy",
        exploration="linear",
        env_kwargs={"continuous": False},
        load_path="./iqn_agent_final.pth",
    )
