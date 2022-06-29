import json
import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
    name = "tressider-2019-04-26_2"
    data_folder = "/home/stpc/data/jrdb_train/train_dataset_with_activity/pointclouds/upper_velodyne/" + name
    label_file = "/home/stpc/data/jrdb_train/train_dataset_with_activity/labels/labels_3d/" + name + ".json"

    lidar_save_folder = "/home/stpc/clean_data/jrdb/pointcloud"
    label_save_folder = "/home/stpc/clean_data/jrdb/label"

    count = 13450

    if not os.path.exists(lidar_save_folder):
        os.makedirs(lidar_save_folder)
    
    if not os.path.exists(label_save_folder):
        os.makedirs(label_save_folder)
    
    f = open(label_file)
    labels = json.load(f)

    lidar_files = sorted(os.listdir(data_folder))
    
    for file in lidar_files:
        pcd_path = os.path.join(data_folder, file)

        pcd = o3d.io.read_point_cloud(pcd_path)
        out_arr = np.asarray(pcd.points)

        points = np.ones((out_arr.shape[0], 4))
        points[:, 0:3] = out_arr

        pcd_name = '%06d' % count + '.bin'
        label_name = '%06d' % count + '.txt'

        points.astype("float32").tofile(os.path.join(lidar_save_folder, pcd_name))

        cam_z = 0.742092
        velo_z = 1.077382
        dz = velo_z - cam_z
        
        cam2velo = R.from_euler('xyz', [0, 0, -0.085])
        rot_arr = cam2velo.as_matrix()
        label = labels["labels"][file]
        
        with open(os.path.join(label_save_folder, label_name), "w") as f:
            for i in range(len(label)):
                box = label[i]["box"]
                x = box["cx"]
                y = box["cy"]
                z = box["cz"]

                vec = np.array([[x], [y], [z]])
                rot_vec = np.matmul(rot_arr, vec)
        
                x = rot_vec[0][0]
                y = rot_vec[1][0]
                z = rot_vec[2][0] - dz

                h = box["h"]
                w = box["w"]
                l = box["l"]
                yaw = box["rot_z"]
                
                z = z - h / 2
                if "social_activity" in label[i] and  "cycling" in label[i]["social_activity"]:
                    line = "Bicycle " + str(h) + " " + str(w) + " " + str(l) + " " + str(x) + " " + str(y) + " " + str(z) + " " + str(-yaw)
                else:
                    line = "Pedestrian " + str(h) + " " + str(w) + " " + str(l) + " " + str(x) + " " + str(y) + " " + str(z) + " " + str(-yaw)
                f.write(line)
                if i != len(label):
                    f.write("\n")
        count += 1
