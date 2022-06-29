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
        write_to_file(os.path.join(save_folder, "fine_tune.txt"), train_set, data)
        #write_to_file(os.path.join(save_folder, "jrdb_val.txt"), val_set, data)
        #write_to_file(os.path.join(save_folder, "test.txt"), test_set, data)

    



def write_to_file(filename, dataset, datatype):
    with open(filename, 'a') as f:
        for data in dataset:
            f.write(data.split(".")[0] + ";" + datatype)
            f.write('\n')
    



if __name__ == "__main__":
    datas = {"jrdb": "/home/stpc/clean_data/jrdb/pointcloud",
             "small_robot": "/home/stpc/clean_data/small_robot/pointcloud",
             "wego": "/home/stpc/clean_data/wego/pointcloud",
             "nuscenes": "/home/stpc/clean_data/nuscenes/pointcloud"}
    percents = {"jrdb": 50, "small_robot": 90, "wego": 90, "nuscenes": 30}

    datas = {"small_robot": "/home/stpc/clean_data/small_robot/pointcloud",
            "wego": "/home/stpc/clean_data/wego/pointcloud"}
    percents = {"small_robot": 100, "wego": 100}
    save_folder = "/home/stpc/clean_data/list"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    split(datas, percents, save_folder)
