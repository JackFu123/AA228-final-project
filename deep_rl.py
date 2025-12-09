"""Deep reinforcement learning utilities for ``SarsaDataset``.

This module provides a simple offline DQN/SARSA training skeleton
built on top of :class:`dataset.SarsaDataset`. It:

- combines local features from consecutive frames with ego
  information into a vector state representation;
- discretizes continuous longitudinal / yaw accelerations into a
  finite action set;
- constructs PyTorch datasets and training loops to perform Bellman
  updates.

Note: the INTERACTION dataset is large. Instantiating
``SarsaDataset`` will load many CSV files, so during development it
is recommended to use a smaller subset directory or limit
``max_samples``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dataset import SarsaDataset

DEFAULT_ACCEL_BINS = np.array([-1.2001847, -0.60009235, 0, 0.60009235, 1.2001847], dtype=np.float32)
DEFAULT_YAW_ACCEL_BINS = np.array([-0.05, -0.025, 0, 0.025, 0.05], dtype=np.float32)
STATE_NEIGHBOR_COUNT = 4
STATE_PER_NEIGHBOR = 3  # transformed_x, transformed_y, transformed_psi_rad
STATE_EGO_DIM = 4       # v_x, v_y, psi_rad, v_long
DELTA_T = 0.1           # seconds, must stay consistent with dataset.get_instance


class DiscreteActionMapper:
    """Map continuous longitudinal and yaw accelerations to a discrete action index."""

    def __init__(
        self,
        accel_bins: Sequence[float] = DEFAULT_ACCEL_BINS,
        yaw_accel_bins: Sequence[float] = DEFAULT_YAW_ACCEL_BINS,
    ) -> None:
        self.accel_bins = np.asarray(accel_bins, dtype=np.float32)
        self.yaw_bins = np.asarray(yaw_accel_bins, dtype=np.float32)
        if self.accel_bins.ndim != 1 or self.yaw_bins.ndim != 1:
            raise ValueError("Action bins must be one-dimensional arrays.")
        self.num_actions = int(self.accel_bins.size * self.yaw_bins.size)

    def encode(self, accel: float, yaw_accel: float) -> int:
        """Find the closest bins and return the corresponding action index."""
        accel_idx = int(np.argmin(np.abs(self.accel_bins - float(accel))))
        yaw_idx = int(np.argmin(np.abs(self.yaw_bins - float(yaw_accel))))
        return accel_idx * self.yaw_bins.size + yaw_idx

    def decode(self, action_idx: int) -> Tuple[float, float]:
        """Decode a discrete index back to the acceleration bin centers."""
        accel_size = self.accel_bins.size
        yaw_size = self.yaw_bins.size
        accel_idx = int(action_idx // yaw_size)
        yaw_idx = int(action_idx % yaw_size)
        return float(self.accel_bins[accel_idx]), float(self.yaw_bins[yaw_idx])


def pad_neighbors(features: np.ndarray) -> np.ndarray:
    """Pad neighbor features to fixed length ``(STATE_NEIGHBOR_COUNT, STATE_PER_NEIGHBOR)``."""
    features = np.asarray(features, dtype=np.float32)

    # [NEW] Handle the case where SarsaDataset includes ego as the last
    # row. If the feature shape is (N, 3) and N > 0, we assume the last
    # row corresponds to ego (see dataset.py modification) and we need
    # to separate neighbors from ego.
    if features.shape[0] > 0:
        # In practice we do not try to infer this from values; we rely
        # on the data construction in dataset.py where
        # ``features = concatenate([neighbors, ego], axis=0)`` and ego
        # is always the last row. ``features`` here are raw (unnormalized)
        # values.
        neighbors = features[:-1]
    else:
        neighbors = features

    if neighbors.ndim == 1:
        neighbors = neighbors.reshape(-1, STATE_PER_NEIGHBOR)
        
    out = np.zeros((STATE_NEIGHBOR_COUNT, STATE_PER_NEIGHBOR), dtype=np.float32)
    count = min(neighbors.shape[0], STATE_NEIGHBOR_COUNT)
    if count > 0:
        out[:count] = neighbors[:count]
    return out.ravel()


def reconstruct_prev_ego(ego: Dict[str, float], action: Dict[str, float]) -> np.ndarray:
    """Approximate previous-step ego state from current ego state and action."""
    v_long_curr = float(ego["v_long"])
    yaw_curr = float(ego["psi_rad"])
    accel = float(action["acceleration"])
    yaw_accel = float(action["yaw_acceleration"])
    v_long_prev = v_long_curr - accel * DELTA_T
    yaw_prev = yaw_curr - yaw_accel * DELTA_T
    # Approximately reconstruct previous longitudinal / lateral
    # velocities, ignoring lateral dynamics.
    v_x_prev = v_long_prev * np.cos(yaw_prev)
    v_y_prev = v_long_prev * np.sin(yaw_prev)
    return np.array([v_x_prev, v_y_prev, yaw_prev, v_long_prev], dtype=np.float32)


def ego_vector(ego: Dict[str, float]) -> np.ndarray:
    """Pack current-frame ego information into a vector."""
    return np.array(
        [float(ego["v_x"]), float(ego["v_y"]), float(ego["psi_rad"]), float(ego["v_long"])],
        dtype=np.float32,
    )


def build_state(instance: Dict[str, Dict], use_next: bool) -> np.ndarray:
    """Build a state vector using either the current frame or the next frame."""
    if use_next:
        # ``feature_next`` contains neighbors + ego_next (vx, vy, v_long)
        raw_feat = instance["feature_next"]
        neighbor_feat = pad_neighbors(raw_feat)
        
        # Try to extract ego-next information from ``raw_feat``
        raw_feat_arr = np.asarray(raw_feat)
        if raw_feat_arr.shape[0] > 0:
            # dataset.py layout: [ego_vx, ego_vy, v_long]
            ego_vec_3 = raw_feat_arr[-1]
            # We need 4D ego features: [vx, vy, psi, v_long]. The
            # next-frame yaw angle is not available here, so we reuse
            # the current yaw (assuming small heading change within
            # one step).
            curr_psi = float(instance["ego"]["psi_rad"])
            ego_feat = np.array([ego_vec_3[0], ego_vec_3[1], curr_psi, ego_vec_3[2]], dtype=np.float32)
        else:
            ego_feat = np.zeros(STATE_EGO_DIM, dtype=np.float32)
            
    else:
        # Current frame
        raw_feat = instance["features"]
        neighbor_feat = pad_neighbors(raw_feat)
        ego_feat = ego_vector(instance["ego"])

    return np.concatenate([neighbor_feat, ego_feat], axis=0)


def default_reward_fn(instance):
    accel = float(instance["action"]["acceleration"])
    yaw_acc = float(instance["action"]["yaw_acceleration"])
    v_long = float(instance["ego"]["v_long"])

    v_target = 4.5  # target cruising speed in m/s (around 30 km/h)

    # 1) Speed reward: closer to v_target is better
    r_speed = - ((v_long - v_target) ** 2)

    # 2) Smoothness: penalize large accelerations / steering
    r_smooth = - (0.04 * accel**2 + 0.02 * yaw_acc**2)

    reward = 0.1 * r_speed + r_smooth
    return float(reward)


class PlanItTransitionDataset(Dataset):
    """Wrap ``SarsaDataset`` samples into ``(s, a, r, s', done)`` transitions."""

    def __init__(
        self,
        data_dir: str,
        action_mapper: DiscreteActionMapper,
        reward_fn: Callable[[Dict[str, Dict]], float] = default_reward_fn,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.planit = SarsaDataset(data_dir)
        self.instances = self.planit.instances
        if max_samples is not None:
            max_samples = min(max_samples, len(self.instances))
            self.instances = self.instances[:max_samples]
        self.action_mapper = action_mapper
        self.reward_fn = reward_fn

        # Compute mean and std of states for normalization.
        states = []
        # To avoid excessive memory usage, randomly sample at most
        # 2000 instances to estimate statistics.
        sample_indices = np.random.choice(len(self.instances), size=min(len(self.instances), 2000), replace=False)
        for i in sample_indices:
            states.append(build_state(self.instances[i], use_next=False))
        states = np.stack(states, axis=0)
        self.state_mean = np.mean(states, axis=0).astype(np.float32)
        self.state_std = np.std(states, axis=0).astype(np.float32) + 1e-5
        self.state_dim = self.state_mean.shape[0]

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inst = self.instances[idx]
        state = build_state(inst, use_next=False)
        next_state = build_state(inst, use_next=True)

        # Normalization
        state = (state - self.state_mean) / self.state_std
        next_state = (next_state - self.state_mean) / self.state_std

        accel = inst["action"]["acceleration"]
        yaw_accel = inst["action"]["yaw_acceleration"]
        action_idx = self.action_mapper.encode(accel, yaw_accel)
        reward = self.reward_fn(inst)
        done = 1.0 if int(inst["meta"]["frame_id"]) >= 39 else 0.0

        return {
            "state": torch.from_numpy(state),
            "action": torch.tensor(action_idx, dtype=torch.long),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_state": torch.from_numpy(next_state),
            "done": torch.tensor(done, dtype=torch.float32),
        }


class QNetwork(nn.Module):
    """Simple fully-connected Q-network."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_actions: int) -> None:
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            last_dim = hidden
        layers.append(nn.Linear(last_dim, num_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class TrainingConfig:
    """Training hyper-parameters for offline DQN."""

    data_dir: str
    batch_size: int = 256
    lr: float = 1e-3
    gamma: float = 0.99
    num_epochs: int = 5
    hidden_dims: Sequence[int] = field(default_factory=lambda: (256, 256))
    target_update_interval: int = 500
    max_samples: Optional[int] = 50000
    num_workers: int = 0
    device: str = "cpu"
    gradient_clip: Optional[float] = 10.0
    save_every_epochs: int = 0
    model_prefix: str = "dqn_planit"


def train_dqn(config: TrainingConfig) -> Dict[str, float]:
    """Run offline DQN training once and return basic statistics."""
    device = torch.device(config.device)
    mapper = DiscreteActionMapper()
    dataset = PlanItTransitionDataset(
        data_dir=config.data_dir,
        action_mapper=mapper,
        max_samples=config.max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    q_net = QNetwork(dataset.state_dim, config.hidden_dims, mapper.num_actions).to(device)
    target_net = QNetwork(dataset.state_dim, config.hidden_dims, mapper.num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.lr)
    loss_fn = nn.SmoothL1Loss()

    global_step = 0
    stats = {"loss": 0.0, "steps": 0}

    for epoch in tqdm(range(config.num_epochs)):
        epoch_loss = 0.0
        for batch in dataloader:
            states = batch["state"].to(device)
            actions = batch["action"].to(device)
            rewards = batch["reward"].to(device)
            next_states = batch["next_state"].to(device)
            dones = batch["done"].to(device)

            q_values = q_net(states)
            q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Double DQN logic: use q_net to select action, target_net to evaluate it
            with torch.no_grad():
                # 1. Select best action from next state using online network
                next_q_online = q_net(next_states)
                _, next_actions_online = next_q_online.max(dim=1)
                
                # 2. Evaluate that action using target network
                next_q_targets = target_net(next_states)
                next_q_value = next_q_targets.gather(1, next_actions_online.unsqueeze(1)).squeeze(1)
                
                targets = rewards + config.gamma * (1.0 - dones) * next_q_value

            loss = loss_fn(q_sa, targets)
            optimizer.zero_grad()
            loss.backward()
            if config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(q_net.parameters(), config.gradient_clip)
            optimizer.step()

            epoch_loss += float(loss.item()) * states.size(0)
            global_step += 1
            if global_step % config.target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

        epoch_avg_loss = epoch_loss / len(dataset)
        stats["loss"] = epoch_avg_loss
        stats["steps"] = global_step
        print(f"[Epoch {epoch + 1}/{config.num_epochs}] loss={epoch_avg_loss:.4f}, steps={global_step}")

        if config.save_every_epochs > 0 and (epoch + 1) % config.save_every_epochs == 0:
            ckpt_path = f"{config.model_prefix}_epoch{epoch + 1}.pt"
            torch.save(q_net.state_dict(), ckpt_path)
            print(f"Model checkpoint saved to {ckpt_path}")

    target_net.load_state_dict(q_net.state_dict())
    final_path = f"{config.model_prefix}_final.pt"
    torch.save(q_net.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    return stats


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Offline DQN training for SarsaDataset.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the directory that contains INTERACTION CSV files.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Training device, e.g. cpu or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Maximum number of samples to load (None means all).")
    parser.add_argument("--target-update", type=int, default=500,
                        help="Number of steps between target network updates.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128],
                        help="Hidden layer dimensions for the Q-network.")
    parser.add_argument("--gamma", type=float, default=0.8,
                        help="Discount factor for future rewards.")
    parser.add_argument("--grad-clip", type=float, default=0.5,
                        help="Gradient clipping threshold; None disables clipping.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of DataLoader workers.")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save checkpoint every N epochs; <=0 saves only the final model.")
    parser.add_argument("--model-prefix", type=str, default="dqn_planit",
                        help="Prefix for saved model checkpoints.")
    args = parser.parse_args(argv)
    return TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        num_epochs=args.epochs,
        hidden_dims=tuple(args.hidden_dims),
        target_update_interval=args.target_update,
        max_samples=None if args.max_samples <= 0 else args.max_samples,
        num_workers=args.num_workers,
        device=args.device,
        gradient_clip=None if args.grad_clip <= 0 else args.grad_clip,
        save_every_epochs=args.save_every,
        model_prefix=args.model_prefix,
    )


if __name__ == "__main__":
    config = parse_args()
    train_dqn(config)

