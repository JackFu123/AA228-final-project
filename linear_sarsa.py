import numpy as np


epsilon, alpha, gamma = 0.1, 1e-2, 0.99


def featurize(state: np.ndarray):
    # normalize the state to 0-1
    state = state.flatten()
    state = (state - state.min()) / (state.max() - state.min() + 1e-8)
    return np.asarray(state, dtype=np.float32)


def discretize_action(action):
    acc = action['acceleration']
    yaw_acc = action['yaw_acceleration']
    if acc > 3:
        return 0
    elif acc > 1:
        return 1
    elif acc > -1:
        return 2
    elif yaw_acc > -3:
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
