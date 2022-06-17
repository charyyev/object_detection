import os
import random

def split(datas, percents, save_folder):
    train_set = []
    val_set = []
    for data in datas.keys():
        lidar_files = sorted(os.listdir(datas[data]))
        random.shuffle(lidar_files)
        train_len = int(percents[data] * len(lidar_files) / 100)
        val_len = train_len // 8

        train_set = lidar_files[:train_len]
        val_set = lidar_files[train_len: min(train_len + val_len, len(lidar_files))]
        #test_set = lidar_files[train_len + val_len: min(train_len + 2 * val_len, len(lidar_files))]
        #write_to_file(os.path.join(save_folder, "small_robot_test.txt"), train_set, data)
        write_to_file(os.path.join(save_folder, "self_train.txt"), train_set, data)
        #write_to_file(os.path.join(save_folder, "small_robot_val.txt"), val_set, data)
        #write_to_file(os.path.join(save_folder, "test.txt"), test_set, data)

    



def write_to_file(filename, dataset, datatype):
    with open(filename, 'a') as f:
        for data in dataset:
            f.write(data.split(".")[0] + ";" + datatype)
            f.write('\n')
    



if __name__ == "__main__":
    datas = {"lyft": "/home/stpc/clean_data/lyft/pointcloud",
             "kitti": "/home/stpc/clean_data/kitti/pointcloud",
             "nuscenes": "/home/stpc/clean_data/nuscenes/pointcloud",
             "small_robot": "/home/stpc/clean_data/small_robot/pointcloud"
    }
    percents = {"kitti": 15, "lyft": 15, "nuscenes": 80, "small_robot": 80}
    datas = {"auto": "/home/stpc/clean_data/auto/pointcloud"}
    percents = {"auto": 100}
    save_folder = "/home/stpc/clean_data/list"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    split(datas, percents, save_folder)
