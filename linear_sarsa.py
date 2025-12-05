import numpy as np


alpha, gamma = 5e-2, 0.99


def featurize(state: np.ndarray):
    s = np.asarray(state, dtype=np.float32)
    X_MAX, Y_MAX = 20.0, 20.0
    s[:, 0] = np.clip(s[:, 0], -X_MAX, X_MAX) / X_MAX
    s[:, 1] = np.clip(s[:, 1], -Y_MAX, Y_MAX) / Y_MAX
    s[:, 2] = s[:, 2] / np.pi  # scale heading to ~[-1, 1]
    flat = s.flatten()
    # bias term
    flat = np.concatenate([flat, np.array([1.0], dtype=flat.dtype)], axis=0)
    return flat


def discretize_action(action):
    acc = action['acceleration']
    yaw_acc = action['yaw_acceleration']
    dis_acc, dis_yaw_acc = None, None
    if acc > 3:
        dis_acc = 0
    elif acc > 1:
        dis_acc = 1
    elif acc > -1:
        dis_acc = 2
    elif acc > -3:
        dis_acc = 3
    else:
        dis_acc = 4
    
    if yaw_acc > 0.5:
        dis_yaw_acc = 5
    elif yaw_acc > 0.1:
        dis_yaw_acc = 6
    elif yaw_acc > -0.1:
        dis_yaw_acc = 7
    elif yaw_acc > -0.5:
        dis_yaw_acc = 8
    else:
        dis_yaw_acc = 9

    return dis_acc, dis_yaw_acc


def sarsa_linear_update(theta, state, action, reward, next_state, next_action):
    s = featurize(state)
    s_prime = featurize(next_state)
    action, action_yaw = discretize_action(action)
    next_action, next_action_yaw = discretize_action(next_action)
    q_sa_acc = theta[action] @ s
    delta_acc = reward + gamma * (theta[next_action] @ s_prime) - q_sa_acc
    theta[action] += alpha * delta_acc * s   # semi-gradient step
    q_sa_yaw = theta[action_yaw] @ s
    delta_yaw = reward + gamma * (theta[next_action_yaw] @ s_prime) - q_sa_yaw
    theta[action_yaw] += alpha * delta_yaw * s   # semi-gradient step

    return delta_acc + delta_yaw
