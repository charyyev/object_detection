import os
import random

def split(data_folder, save_folder):
    lidar_files = sorted(os.listdir(data_folder))
    random.shuffle(lidar_files)
    train_len = int(0.9 * len(lidar_files))

    train_set = lidar_files[:train_len]
    val_set = lidar_files[train_len:]

    write_to_file(os.path.join(save_folder, "train.txt"), train_set)
    write_to_file(os.path.join(save_folder, "val.txt"), val_set)




def write_to_file(filename, dataset):
    with open(filename, 'w') as f:
        for data in dataset:
            f.write(data.split(".")[0] + ";" + "kitti")
            f.write('\n')
    



if __name__ == "__main__":
    data_folder = "/home/stpc/data/kitti/velodyne/training_reduced/velodyne"
    save_folder = "/home/stpc/data/train"
    split(data_folder, save_folder)
