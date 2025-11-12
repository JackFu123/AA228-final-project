import numpy as np


epsilon, alpha, gamma = 0.1, 1e-2, 0.99


def featurize(state):
    # Return a 1D normalized feature vector
    return np.asarray(state, dtype=np.float32)


def sarsa_linear_update(theta, state, action, reward, next_state, next_action):
    s = featurize(state)
    s_prime = featurize(next_state)
    q_sa = theta[action] @ s
    delta = reward + gamma * (theta[next_action] @ s_prime) - q_sa
    theta[action] += alpha * delta * s   # semi-gradient step
    return delta
