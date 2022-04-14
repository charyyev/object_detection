import shutil
import os

if __name__ == "__main__":
    src_pcl_folder = "/home/stpc/data/kitti/velodyne/training_reduced/velodyne"
    src_label_folder = "/home/stpc/data/kitti/label_2/training/label_2_reduced"

    dst_pcl_folder = "/home/stpc/clean_data/kitti/pointcloud"
    dst_label_folder = "/home/stpc/clean_data/kitti/label"

    if not os.path.exists(dst_pcl_folder):
        os.makedirs(dst_pcl_folder)
    if not os.path.exists(dst_label_folder):
        os.makedirs(dst_label_folder)

    count = 1

    lidar_files = sorted(os.listdir(src_pcl_folder))
    for lidar_file in lidar_files:
        src_lidar_path = os.path.join(src_pcl_folder, lidar_file)
        src_label_path = os.path.join(src_label_folder, lidar_file.replace("bin", "txt"))

        dst_lidar_path = os.path.join(dst_pcl_folder, '%06d' % count + '.bin')
        dst_label_path = os.path.join(dst_label_folder, '%06d' % count + '.txt')

        shutil.copy(src_lidar_path, dst_lidar_path)
        shutil.copy(src_label_path, dst_label_path)

        count += 1

