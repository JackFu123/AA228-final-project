from dataset import SarsaDataset
from linear_sarsa import sarsa_linear_update
import numpy as np
import time


if __name__ == "__main__":
    start_time = time.time()
    print(f"Start time: {time.strftime("%Y-%m-%d %H:%M:%S")}")
    dataset_train = SarsaDataset("/Users/yiqunf/Documents/cousess/AA228/final_project/INTERACTION-Dataset-DR-single-v1_2/train/")
    dataset_val = SarsaDataset("/Users/yiqunf/Documents/cousess/AA228/final_project/INTERACTION-Dataset-DR-single-v1_2/val/")
    # 13 features (4 agents x [x, y, psi]) + 1 bias
    theta = np.random.randn(10, 13)
    epsilon = 0.1
    for i in range(100):
        sum_abs = 0.0
        sum_sq = 0.0
        for instance in dataset_train:
            delta = sarsa_linear_update(theta, instance['features'], instance['action'], instance['reward'], instance['feature_next'], instance['next_action'])
            sum_abs += abs(delta)
            sum_sq += float(delta) * float(delta)
        n = len(dataset_train)
        avg_abs = sum_abs / n
        mse = sum_sq / n
        print(f"{time.strftime("%Y-%m-%d %H:%M:%S")} iteration: {i}")
        print(f"mean delta: {avg_abs} | mse delta: {mse}")
        if avg_abs < epsilon:
            break
    print(theta)
    end_time = time.time()
    print(f"End time: {time.strftime("%Y-%m-%d %H:%M:%S")}")
    print("Time taken(seconds): ", time.time() - start_time)
