import numpy as np
import os


if __name__ == "__main__":
    folder = "/home/stpc/clean_data/wego/pointcloud"

    lidar_files = sorted(os.listdir(folder))

    for file in lidar_files:
        pcd_path = os.path.join(folder, file)

        out_arr = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 3)

        points = np.ones((out_arr.shape[0], 4))
        points[:, 0:3] = out_arr

        points.astype("float32").tofile(pcd_path)