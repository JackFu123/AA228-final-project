import os

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def list_files_with_suffix(dir_path, suffix):
    """List all sub-files with suffix in given dir_path."""
    return [os.path.join(root, f) for root, _, files
            in os.walk(dir_path) for f in files if f.endswith(suffix)]


class PlanItDataset(Dataset):

    @staticmethod
    def _wrap_angle(angle_rad: np.ndarray) -> np.ndarray:
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
        rel_psi = PlanItDataset._wrap_angle(frame["psi_rad"].to_numpy(dtype=np.float64) - float(ego_psi_rad))

        out = frame.copy()
        out["transformed_x"] = tx.astype(np.float64)
        out["transformed_y"] = ty.astype(np.float64)
        out["transformed_psi_rad"] = rel_psi.astype(np.float64)
        out["distance"] = np.hypot(out["transformed_x"], out["transformed_y"])
        return out

    def __init__(self, data_dir, label_agent=False):
        self.data_dir = data_dir
        self.vehicle_columns = ['x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width', 'track_id']
        self.vru_columns = ['x', 'y', 'vx', 'vy', 'track_id']
        self.max_vehicle = 48
        self.max_vru = 14
        self.max_map = 4
        self.max_lane_point_size = 20
        self.files = list_files_with_suffix(self.data_dir, '.csv')
        self.instances = []
        self.label_agent = label_agent
        self.collision_data = []
        self.index_to_col_data = {}
        self.instances = self.get_instance(self.files)
        self.total_num_samples = len(self.instances)

    def __len__(self):
        return self.total_num_samples

    def __getitem__(self, idx):
        return self.instances[idx]

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
                    features = nearest4[['transformed_x', 'transformed_y', 'transformed_psi_rad']].to_numpy()

                    if frame_id == 1:
                        last_v_long_t = v_long_t
                        last_features = features
                        last_ego_yaw_rad = ego_psi_t
                        continue
                    long_acc = (v_long_t - last_v_long_t) / 0.1
                    yaw_acc = (ego_psi_t - last_ego_yaw_rad) / 0.1

                    instances.append({
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
                        "meta": {
                            "file_name": csv_file,
                            "case_id": case_id,
                            "frame_id": int(frame_id - 1),
                            "ego_track_id": int(ego_track_id),
                        }
                    })
            # break
        return instances


if __name__ == "__main__":
    # dataset_train = PlanItDataset("/Users/yiqunf/Documents/cousess/AA228/final_project/INTERACTION-Dataset-DR-single-v1_2/train/")
    dataset_val = PlanItDataset("/Users/yiqunf/Documents/cousess/AA228/final_project/INTERACTION-Dataset-DR-single-v1_2/val/")
    # print(len(dataset_train))
    print(len(dataset_val))
