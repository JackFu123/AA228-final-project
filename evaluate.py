"""Evaluation and visualization script for PlanIt DQN.

This script:

1. Loads a trained DQN model (e.g. ``dqn_planit.pt``).
2. Reads several cases from the specified dataset.
3. For each case, uses the model to predict actions and compares them
   against the ground-truth actions/trajectories from the dataset.
4. Produces visualization figures:

   - longitudinal acceleration comparison (ground truth vs prediction),
   - yaw acceleration comparison,
   - ego velocity profile (and optionally Q-value heatmaps).
"""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import SarsaDataset as PlanItDataset
from deep_rl import (
    DEFAULT_ACCEL_BINS,
    DEFAULT_YAW_ACCEL_BINS,
    DiscreteActionMapper,
    QNetwork,
    build_state,
)

# Configure fonts so that plots can render UTF-8 text on macOS/Linux
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_model(model_path: str, input_dim: int, num_actions: int, hidden_dims=(128, 128), device="cpu"):
    """Load a trained Q-network from disk."""
    model = QNetwork(input_dim, hidden_dims, num_actions)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_case(
    model: QNetwork,
    dataset: PlanItDataset,
    case_idx: int,
    action_mapper: DiscreteActionMapper,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device="cpu",
) -> Dict:
    """Evaluate the model on a single case.

    This performs **open-loop single-step prediction**, i.e. each step
    predicts the action based on the true previous state, rather than
    rolling out closed-loop trajectories.
    """
    # Retrieve all frames for the given case.
    # ``PlanItDataset`` currently flattens all cases into a single
    # ``instances`` list, so we need to select the contiguous segment
    # belonging to a specific ``case_id``.
    target_instance = dataset.instances[case_idx]
    case_id = target_instance["meta"]["case_id"]
    file_name = target_instance["meta"]["file_name"]
    
    # Select all frames that match the same case_id and file_name
    case_instances = [
        inst for inst in dataset.instances 
        if inst["meta"]["case_id"] == case_id and inst["meta"]["file_name"] == file_name
    ]
    # Sort by frame_id
    case_instances.sort(key=lambda x: x["meta"]["frame_id"])

    timestamps = []
    gt_accels = []
    pred_accels = []
    gt_yaw_accels = []
    pred_yaw_accels = []
    velocities = []
    
    # Store Q-value vectors for later analysis / visualization
    q_values_history = []

    with torch.no_grad():
        for inst in case_instances:
            # Build state for this frame
            state = build_state(inst, use_next=False)
            # Normalize using dataset statistics
            norm_state = (state - state_mean) / state_std
            state_tensor = torch.from_numpy(norm_state).float().unsqueeze(0).to(device)

            # Forward pass through the model
            q_out = model(state_tensor)
            action_idx = q_out.argmax(dim=1).item()
            pred_acc, pred_yaw_acc = action_mapper.decode(action_idx)

            # Record data (assume sampling rate is 10 Hz)
            timestamps.append(inst["meta"]["frame_id"] * 0.1)
            gt_accels.append(inst["action"]["acceleration"])
            pred_accels.append(pred_acc)
            gt_yaw_accels.append(inst["action"]["yaw_acceleration"])
            pred_yaw_accels.append(pred_yaw_acc)
            velocities.append(inst["ego"]["v_long"])
            q_values_history.append(q_out.cpu().numpy()[0])

    return {
        "case_id": case_id,
        "timestamps": timestamps,
        "gt_accel": gt_accels,
        "pred_accel": pred_accels,
        "gt_yaw_accel": gt_yaw_accels,
        "pred_yaw_accel": pred_yaw_accels,
        "velocity": velocities,
        "q_values": np.array(q_values_history)
    }


def plot_evaluation(results: Dict, output_dir: str):
    """Plot evaluation curves for a single case."""
    case_id = results["case_id"]
    t = results["timestamps"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Longitudinal acceleration comparison
    axs[0].plot(t, results["gt_accel"], label="Ground Truth", color="black", linestyle="--")
    axs[0].plot(t, results["pred_accel"], label="Model Prediction", color="red", alpha=0.8)
    axs[0].set_ylabel("Acceleration (m/s^2)")
    axs[0].set_title(f"Case {case_id}: Longitudinal Acceleration")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # 2. Yaw acceleration comparison
    axs[1].plot(t, results["gt_yaw_accel"], label="Ground Truth", color="black", linestyle="--")
    axs[1].plot(t, results["pred_yaw_accel"], label="Model Prediction", color="blue", alpha=0.8)
    axs[1].set_ylabel("Yaw Acceleration (rad/s^2)")
    axs[1].set_title(f"Case {case_id}: Yaw Acceleration")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # 3. Ego velocity profile (for reference)
    axs[2].plot(t, results["velocity"], color="green", label="Ego Velocity")
    axs[2].set_ylabel("Velocity (m/s)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title(f"Case {case_id}: Velocity Profile")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"case_{int(case_id)}_eval.png")
    plt.savefig(save_path)
    print(f"Saved evaluation figure to: {save_path}")
    plt.close()


def compute_stats(dataset: PlanItDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Re-compute normalization statistics (mean/std) for states.

    In a production setup these should be saved during training and
    reloaded here; for simplicity we recompute them by random
    subsampling.
    """
    print("Computing normalization statistics for states ...")
    states = []
    indices = np.random.choice(len(dataset), size=min(len(dataset), 2000), replace=False)
    for i in indices:
        states.append(build_state(dataset.instances[i], use_next=False))
    states = np.stack(states, axis=0)
    mean = np.mean(states, axis=0).astype(np.float32)
    std = np.std(states, axis=0).astype(np.float32) + 1e-5
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Evaluate PlanIt DQN model on offline dataset.")
    parser.add_argument("--model-path", type=str, default="dqn_planit.pt",
                        help="Path to the trained DQN model file.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory that contains the evaluation CSV files.")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Directory to store evaluation figures.")
    parser.add_argument("--num-cases", type=int, default=5,
                        help="Number of distinct cases to evaluate.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128],
                        help="Hidden layer dimensions for the Q-network.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Prepare dataset and model components
    print(f"Loading dataset from: {args.data_dir} ...")
    dataset = PlanItDataset(args.data_dir)
    mapper = DiscreteActionMapper()
    
    # Compute normalization statistics (ideally load from training,
    # but here we recompute for simplicity)
    state_mean, state_std = compute_stats(dataset)
    input_dim = state_mean.shape[0]

    # 2. Load model
    print(f"Loading model from: {args.model_path} ...")
    if not os.path.exists(args.model_path):
        print(f"Error: model file not found: {args.model_path}")
        return

    model = load_model(
        args.model_path, 
        input_dim, 
        mapper.num_actions, 
        hidden_dims=tuple(args.hidden_dims),
        device=args.device
    )

    # 3. Select several cases for evaluation.
    # Logic: first select indices over instances, then group by case_id.
    indices = np.linspace(0, len(dataset)-1, args.num_cases, dtype=int)
    
    processed_cases = set()
    count = 0
    
    for idx in indices:
        case_id = dataset.instances[idx]["meta"]["case_id"]
        if case_id in processed_cases:
            continue
            
        print(f"Evaluating case_id={case_id} ...")
        results = evaluate_case(
            model, dataset, int(idx), mapper, state_mean, state_std, args.device
        )
        plot_evaluation(results, args.output_dir)
        
        processed_cases.add(case_id)
        count += 1
        if count >= args.num_cases:
            break

    print("Evaluation finished.")


if __name__ == "__main__":
    main()


