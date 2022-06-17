import numpy as np 
import open3d as o3d
import os

if __name__ == "__main__":
    data_folder = "/home/stpc/data/flobot/warehouse-2018-06-12-17-10-22-pcd-labels/pointcloud"
    save_location = data_folder + "_np"

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    lidar_files = sorted(os.listdir(data_folder))
    lidar_paths = [os.path.join(data_folder, f) for f in lidar_files]

    for path, file in zip(lidar_paths, lidar_files):
        pcd = o3d.io.read_point_cloud(path)
        out_arr = np.asarray(pcd.points)

        # add 1 as intensity if not available
        if out_arr.shape[1] == 3:
            points = np.ones((out_arr.shape[0], 4))
        else:
            points = np.ones(out_arr.shape)

        points[:, 0:3] = out_arr

        filename = file[:len(file) - 4] + ".bin"
        points.astype("float32").tofile(os.path.join(save_location, filename))
         