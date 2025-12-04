import numpy as np


epsilon, alpha, gamma = 0.1, 1e-2, 0.99


def featurize(state: np.ndarray):
    s = np.asarray(state, dtype=np.float32)
    X_MAX, Y_MAX = 20.0, 20.0
    s[:, 0] = np.clip(s[:, 0], -X_MAX, X_MAX) / X_MAX
    s[:, 1] = np.clip(s[:, 1], -Y_MAX, Y_MAX) / Y_MAX
    s[:, 2] = s[:, 2] / np.pi  # scale to [-1, 1]
    return s.flatten()


def discretize_action(action):
    acc = action['acceleration']
    yaw_acc = action['yaw_acceleration']
    if acc > 3:
        return 0
    elif acc > 1:
        return 1
    elif acc > -1:
        return 2
    elif acc > -3:
        return 3
    else:
        return 4


def sarsa_linear_update(theta, state, action, reward, next_state, next_action):
    s = featurize(state)
    s_prime = featurize(next_state)
    action = discretize_action(action)
    next_action = discretize_action(next_action)
    q_sa = theta[action] @ s
    delta = reward + gamma * (theta[next_action] @ s_prime) - q_sa
    theta[action] += alpha * delta * s   # semi-gradient step
    return delta
