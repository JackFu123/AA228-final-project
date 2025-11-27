import numpy as np
from dataset import SarsaDataset
from linear_sarsa import featurize, sarsa_linear_update


if __name__ == "__main__":
    dataset_train = SarsaDataset("/Users/yiqunf/Documents/cousess/AA228/final_project/INTERACTION-Dataset-DR-single-v1_2/train/")
    dataset_val = SarsaDataset("/Users/yiqunf/Documents/cousess/AA228/final_project/INTERACTION-Dataset-DR-single-v1_2/val/")
    theta = np.random.randn(5, 12)
    for i in range(100):
        for instance in dataset_train:
            delta = sarsa_linear_update(theta, instance['features'], instance['action'], instance['reward'], instance['feature_next'], instance['next_action'])
            print(theta, delta)
            if delta < 0.01:
                break
    print(theta)