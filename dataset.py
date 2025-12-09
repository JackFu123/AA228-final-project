import os

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def list_files_with_suffix(dir_path, suffix):
    """List all sub-files with suffix in given dir_path."""
    return [os.path.join(root, f) for root, _, files
            in os.walk(dir_path) for f in files if f.endswith(suffix)]


class SarsaDataset(Dataset):

    @staticmethod
    def _wrap_angle(angle_rad: [np.ndarray, float]):
        """Wrap angles to [-pi, pi]."""
        return (angle_rad + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def transform_to_ego(frame: pd.DataFrame, ego_x: float, ego_y: float, ego_psi_rad: float) -> pd.DataFrame:
        """
        Add ego-centric coordinates to a single-frame dataframe.
        Returns: DataFrame with transformed_x, transformed_y, transformed_psi_rad.
        """
        # Relative positions
        dx = frame["x"].to_numpy(dtype=np.float64) - float(ego_x)
        dy = frame["y"].to_numpy(dtype=np.float64) - float(ego_y)

        # Rotation by -ego_psi (world -> ego frame)
        c = np.cos(-ego_psi_rad)
        s = np.sin(-ego_psi_rad)
        tx = c * dx - s * dy
        ty = s * dx + c * dy

        # Relative heading
        rel_psi = SarsaDataset._wrap_angle(frame["psi_rad"].to_numpy(dtype=np.float64) - float(ego_psi_rad))

        out = frame.copy()
        out["transformed_x"] = tx.astype(np.float64)
        out["transformed_y"] = ty.astype(np.float64)
        out["transformed_psi_rad"] = rel_psi.astype(np.float64)
        out["distance"] = np.hypot(out["transformed_x"], out["transformed_y"])
        return out

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.vehicle_columns = ['x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width', 'track_id']
        self.vru_columns = ['x', 'y', 'vx', 'vy', 'track_id']
        self.max_vehicle = 48
        self.max_vru = 14
        # Map data has not been used for now
        self.max_map = 4
        self.max_lane_point_size = 20
        self.files = list_files_with_suffix(self.data_dir, '.csv')
        self.instances = self.get_instance(self.files)
        self.total_num_samples = len(self.instances)

    def __len__(self):
        return self.total_num_samples

    def __getitem__(self, idx):
        return self.instances[idx]

    def sarsa_sample(self, instance_list: list):
        sarsa_sample = []
        last_sample = None
        for instance in instance_list:
            if last_sample is None:
                last_sample = instance
                continue
            current_sample = last_sample
            current_sample['next_action'] = instance['action']
            sarsa_sample.append(current_sample)
            last_sample = instance
        return sarsa_sample

    def get_reward(self, nearest4, min_distance, ego_psi_t, v_long_t, long_acc, yaw_acc):
        safety_distance = 10.0
        p_dist = float(max(0.0, (safety_distance - float(min_distance)) / safety_distance))

        # Time-to-collision (TTC) approximation for agents ahead along ego x-axis
        tx_ahead = nearest4["transformed_x"].to_numpy(dtype=np.float64)
        agent_v_long = nearest4["vx"].to_numpy(dtype=np.float64) * np.cos(ego_psi_t) \
            + nearest4["vy"].to_numpy(dtype=np.float64) * np.sin(ego_psi_t)
        rel_v_long = agent_v_long - v_long_t
        valid_mask = (tx_ahead > 0.0) & (rel_v_long < -1e-3)
        if np.any(valid_mask):
            ttc_vals = tx_ahead[valid_mask] / (-rel_v_long[valid_mask])
            min_ttc = float(np.minimum(np.min(ttc_vals), 1e6))
        else:
            min_ttc = float(np.inf)
        ttc_threshold = 5.0
        p_ttc = float(max(0.0, (ttc_threshold - min_ttc) / ttc_threshold)) if np.isfinite(min_ttc) else 0.0

        # Comfort penalty (quadratic in accelerations)
        a_max = 3.5
        yaw_acc_max = 0.5
        p_comfort = float((long_acc / a_max) ** 2 + (yaw_acc / yaw_acc_max) ** 2)
        
        # Collision penalty (hard penalty for near-collision)
        collision_penalty = -10.0 if float(min_distance) < 1.0 else 0.0
        
        # Weighted sum (negative penalties)
        w_dist, w_ttc, w_comfort = 1.0, 1.0, 0.5
        reward = collision_penalty - (w_dist * p_dist + w_ttc * p_ttc + w_comfort * p_comfort)
        return reward

    def get_instance(self, csv_files:list):
        instances = []
        for csv_file in csv_files:
            print(csv_file)
            df = pd.read_csv(csv_file)
            grouped_df = df.groupby('case_id')
            for case_id, group in grouped_df:
                # plt.figure()
                ego_track_id = None
                track_df = group.groupby('track_id')
                for track_id, track in track_df:
                    # plt.plot(track['x'], track['y'])
                    if ego_track_id is None and len(track) == 40 and track.iloc[0]['agent_type'] == "car":
                        ego_track_id = track_id
                        break
                frame_df = group.groupby('frame_id')
                case_instance = []

                last_v_long_t = None
                last_ego_yaw_rad = None
                last_features = None
                for frame_id, frame in frame_df:
                    # Use ego row within this frame to get ego pose at this timestamp
                    ego_row = frame[frame['track_id'] == ego_track_id].iloc[0]
                    ego_x_t = float(ego_row['x'])
                    ego_y_t = float(ego_row['y'])
                    ego_psi_t = float(ego_row['psi_rad'])
                    ego_vx_t = float(ego_row['vx'])
                    ego_vy_t = float(ego_row['vy'])
                    v_long_t = ego_vx_t * np.cos(ego_psi_t) + ego_vy_t * np.sin(ego_psi_t)

                    # Add transformed columns to the frame (ego-centric)
                    frame_with_ego = self.transform_to_ego(frame, ego_x_t, ego_y_t, ego_psi_t)

                    # Exclude ego itself and select the four nearest agents
                    non_ego = frame_with_ego[frame_with_ego["track_id"] != ego_track_id]
                    nearest4 = non_ego.nsmallest(4, "distance")
                    if len(nearest4) == 0:
                        last_v_long_t = None
                        last_ego_yaw_rad = None
                        last_features = None
                        continue
                    else:
                        while len(nearest4) < 4:
                            nearest4 = pd.concat([nearest4, nearest4.iloc[[-1]].copy()], ignore_index=True)
                    min_distance = nearest4["distance"].min()
                    features = nearest4[['transformed_x', 'transformed_y', 'transformed_psi_rad']]
                    features = features.fillna(0).to_numpy()
                    
                    # Add ego velocity information to features
                    # Structure: [[n1_x, n1_y, n1_psi], ..., [n4_x, n4_y, n4_psi], [ego_vx, ego_vy, ego_v_long]]
                    # We pad ego features to match neighbor feature width (3)
                    ego_feature_row = np.array([[ego_vx_t, ego_vy_t, float(v_long_t)]])
                    features = np.concatenate([features, ego_feature_row], axis=0)

                    if last_features is None:
                        last_v_long_t = v_long_t
                        last_features = features
                        last_ego_yaw_rad = ego_psi_t
                        continue
                    long_acc = (v_long_t - last_v_long_t) / 0.1
                    yaw_acc = SarsaDataset._wrap_angle(ego_psi_t - last_ego_yaw_rad) / 0.1
                    reward = self.get_reward(nearest4, min_distance, ego_psi_t, v_long_t, long_acc, yaw_acc)
                    case_instance.append({
                        "features": last_features,
                        "feature_next": features, # nearest 4 agents in ego frame
                        "ego": {
                            "v_x": ego_vx_t,
                            "v_y": ego_vy_t,
                            "psi_rad": ego_psi_t,
                            "v_long": float(v_long_t),
                        },
                        "action": {
                            "acceleration": long_acc,
                            "yaw_acceleration": yaw_acc,
                        },
						"reward": reward,
                        "meta": {
                            "file_name": csv_file,
                            "case_id": case_id,
                            "frame_id": int(frame_id - 1),
                            "ego_track_id": int(ego_track_id),
                        }
                    })
                    if abs(long_acc) > 3.5:
                        print("large long acc: ", long_acc, "yaw acc: ", yaw_acc)
                    last_v_long_t = v_long_t
                    last_features = features
                    last_ego_yaw_rad = ego_psi_t
                instances.extend(self.sarsa_sample(case_instance))
            # break
        return instances
