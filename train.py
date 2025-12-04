from dataset import SarsaDataset
from linear_sarsa import sarsa_linear_update
import numpy as np
import time

if __name__ == "__main__":
    start_time = time.time()
    print("Start time: ", start_time)
    dataset_train = SarsaDataset("/Users/yiqunf/Documents/cousess/AA228/final_project/INTERACTION-Dataset-DR-single-v1_2/train/")
    dataset_val = SarsaDataset("/Users/yiqunf/Documents/cousess/AA228/final_project/INTERACTION-Dataset-DR-single-v1_2/val/")
    theta = np.random.randn(5, 12)
    for i in range(1000):
        sum_delta = 0
        for instance in dataset_train:
            delta = sarsa_linear_update(theta, instance['features'], instance['action'], instance['reward'], instance['feature_next'], instance['next_action'])
            sum_delta += delta
        print("average delta: ", sum_delta / len(dataset_train))
        if abs(sum_delta / len(dataset_train)) < 0.01:
            break
    print(theta)
    end_time = time.time()
    print("End time: ", end_time)
    print("Time taken: ", end_time - start_time)
