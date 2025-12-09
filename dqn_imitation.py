"""Imitation-regularized offline DQN training.

Compared with the vanilla DQN in ``deep_rl.py``, this script adds
an extra regularization term that encourages the policy to imitate
human discrete actions on the offline dataset:

    L_total = L_TD + lambda_im * CE(softmax(Q(s, ·)), a_human_idx)

Here ``a_human_idx`` is obtained by discretizing the continuous
``(accel, yaw_acc)`` actions from the dataset using the same
``DiscreteActionMapper`` that is used during interaction.

Example usage::

    python dqn_imitation.py \
      --data-dir /path/to/sample_train \
      --device mps \
      --epochs 20 \
      --batch-size 256 \
      --lr 1e-4 \
      --gamma 0.95 \
      --max-samples 50000 \
      --hidden-dims 256 256 \
      --lambda-im 0.1 \
      --save-every 5 \
      --model-prefix dqn_imi

For evaluation, we directly reuse :mod:`evaluate.py`::

    python evaluate.py \
      --model-path dqn_imi_final.pt \
      --data-dir /path/to/sample_val \
      --output-dir eval_dqn_imi \
      --device mps \
      --hidden-dims 256 256
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deep_rl import (
    DiscreteActionMapper,
    PlanItTransitionDataset,
    QNetwork,
    default_reward_fn,
)


@dataclass
class ImitationConfig:
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

    # imitation regularizer
    lambda_im: float = 0.1
    temperature: float = 1.0

    # checkpointing
    save_every_epochs: int = 0
    model_prefix: str = "dqn_imi"


def train_dqn_imitation(config: ImitationConfig) -> Dict[str, float]:
    device = torch.device(config.device)

    mapper = DiscreteActionMapper()
    dataset = PlanItTransitionDataset(
        data_dir=config.data_dir,
        action_mapper=mapper,
        reward_fn=default_reward_fn,
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
    td_loss_fn = nn.SmoothL1Loss()

    global_step = 0
    stats = {"loss": 0.0, "td_loss": 0.0, "im_loss": 0.0, "steps": 0}

    for epoch in tqdm(range(config.num_epochs)):
        epoch_td_loss = 0.0
        epoch_im_loss = 0.0
        epoch_total_loss = 0.0

        for batch in dataloader:
            states = batch["state"].to(device)          # [B, S]
            actions = batch["action"].to(device)        # [B]
            rewards = batch["reward"].to(device)        # [B]
            next_states = batch["next_state"].to(device)
            dones = batch["done"].to(device)            # [B]

            # Q(s, ·)
            q_values = q_net(states)                    # [B, A]
            q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

            # Double DQN target
            with torch.no_grad():
                next_q_online = q_net(next_states)      # [B, A]
                _, next_actions_online = next_q_online.max(dim=1)  # [B]

                next_q_targets = target_net(next_states)
                next_q_value = next_q_targets.gather(1, next_actions_online.unsqueeze(1)).squeeze(1)

                targets = rewards + config.gamma * (1.0 - dones) * next_q_value

            # TD loss
            td_loss = td_loss_fn(q_sa, targets)

            # Imitation loss: cross-entropy between softmax(Q/temperature) and teacher action index
            if config.lambda_im > 0.0:
                logits = q_values / max(config.temperature, 1e-6)
                log_probs = F.log_softmax(logits, dim=1)
                im_loss = F.nll_loss(log_probs, actions)
                loss = td_loss + config.lambda_im * im_loss
            else:
                im_loss = torch.tensor(0.0, device=device)
                loss = td_loss

            optimizer.zero_grad()
            loss.backward()
            if config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(q_net.parameters(), config.gradient_clip)
            optimizer.step()

            batch_size = states.size(0)
            epoch_td_loss += float(td_loss.item()) * batch_size
            epoch_im_loss += float(im_loss.item()) * batch_size
            epoch_total_loss += float(loss.item()) * batch_size

            global_step += 1
            if global_step % config.target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

        epoch_td_loss /= len(dataset)
        epoch_im_loss /= len(dataset)
        epoch_total_loss /= len(dataset)

        stats["loss"] = epoch_total_loss
        stats["td_loss"] = epoch_td_loss
        stats["im_loss"] = epoch_im_loss
        stats["steps"] = global_step

        print(
            f"[Epoch {epoch + 1}/{config.num_epochs}] "
            f"total={epoch_total_loss:.4f}, td={epoch_td_loss:.4f}, im={epoch_im_loss:.4f}, steps={global_step}"
        )

        if config.save_every_epochs > 0 and (epoch + 1) % config.save_every_epochs == 0:
            ckpt_path = f"{config.model_prefix}_epoch{epoch + 1}.pt"
            torch.save(q_net.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    # Synchronize and save the final model
    target_net.load_state_dict(q_net.state_dict())
    final_path = f"{config.model_prefix}_final.pt"
    torch.save(q_net.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    return stats


def parse_args(argv: Optional[Sequence[str]] = None) -> ImitationConfig:
    parser = argparse.ArgumentParser(description="Offline DQN with imitation regularizer.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the directory that contains INTERACTION CSV files.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Training device, e.g. cpu / cuda:0 / mps.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Maximum number of samples to load (<=0 means all).")
    parser.add_argument("--target-update", type=int, default=500,
                        help="Number of steps between target network updates.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer dimensions for the Q-network.")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount factor for future rewards.")
    parser.add_argument("--grad-clip", type=float, default=0.5,
                        help="Gradient clipping threshold; <=0 disables clipping.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of DataLoader workers.")

    parser.add_argument("--lambda-im", type=float, default=0.1,
                        help="Coefficient for the imitation regularization term.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature used in the imitation softmax.")

    parser.add_argument("--save-every", type=int, default=0,
                        help="Save checkpoint every N epochs; <=0 saves only the final model.")
    parser.add_argument("--model-prefix", type=str, default="dqn_imi",
                        help="Prefix for saved model checkpoints.")

    args = parser.parse_args(argv)

    return ImitationConfig(
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
        lambda_im=args.lambda_im,
        temperature=args.temperature,
        save_every_epochs=args.save_every,
        model_prefix=args.model_prefix,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_dqn_imitation(cfg)
