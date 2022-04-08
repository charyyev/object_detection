import numpy as np
from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt

from utils.preprocess import transform_metric2label, trasform_label2metric, get_points_in_a_rotated_box, voxel_to_img


class KittiDataset(Dataset):
    def __init__(self, pointcloud_folder, label_folder, data_file, config, train = True) -> None:
        self.pointcloud_folder = pointcloud_folder
        self.label_folder = label_folder
        self.data_file = data_file

        self.create_data_list()

        self.geometry = config["geometry"]
        self.config = config
        self.train = train
       

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file = '{}.bin'.format(self.data_list[idx])
        lidar_path = os.path.join(self.pointcloud_folder, file)
        points = self.read_points(lidar_path)
        scan = self.voxelize(points)
        scan = torch.from_numpy(scan)
        scan = scan.permute(2, 0, 1)
        reg_map, cls_map = self.get_label(idx)
        reg_map = torch.from_numpy(reg_map).permute(2, 0, 1)
        
        if self.train:
            return {"voxel": scan, 
                    "reg_map": reg_map,
                    "cls_map": cls_map
                    }

        class_list, boxes = self.read_bbox(idx)
        return {"voxel": scan, 
                "reg_map": reg_map,
                "cls_map": cls_map,
                "cls_list": class_list,
                "points": points,
                "boxes": boxes,
                }       

    def read_points(self, lidar_path):
        return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    def voxelize(self, points):
        x_min = self.geometry["x_min"]
        x_max = self.geometry["x_max"]
        y_min = self.geometry["y_min"]
        y_max = self.geometry["y_max"]
        z_min = self.geometry["z_min"]
        z_max = self.geometry["z_max"]
        x_res = self.geometry["x_res"]
        y_res = self.geometry["y_res"]
        z_res = self.geometry["z_res"]

        x_size = int((x_max - x_min) / x_res)
        y_size = int((y_max - y_min) / y_res)
        z_size = int((z_max - z_min) / z_res)

        #clip points
        x_indexes = np.logical_and(points[:, 0] > x_min, points[:, 0] < x_max)
        y_indexes = np.logical_and(points[:, 1] > y_min, points[:, 1] < y_max)
        z_indexes = np.logical_and(points[:, 2] > z_min, points[:, 2] < z_max)
        pts = points[np.logical_and(np.logical_and(x_indexes, y_indexes), z_indexes)]

        occupancy_mask = np.zeros((pts.shape[0], 3), dtype = np.int32)
        voxels = np.zeros((x_size, y_size, z_size), dtype = np.float32)
        occupancy_mask[:, 0] = (pts[:, 0] - x_min) / x_res
        occupancy_mask[:, 1] = (pts[:, 1] - y_min) / y_res
        occupancy_mask[:, 2] = (pts[:, 2] - z_min) / z_res

        idxs = np.array([occupancy_mask[:, 0].reshape(-1), occupancy_mask[:, 1].reshape(-1), occupancy_mask[:, 2].reshape( -1)])

        voxels[idxs[0], idxs[1], idxs[2]] = 1
        return np.swapaxes(voxels, 0, 1)


    def get_label(self, idx):
        '''
        :param i: the ith velodyne scan in the train/val set
        :return: label map: <--- This is the learning target
                a tensor of shape 800 * 700 * 7 representing the expected output

        '''

        f_name = '{}.txt'.format(self.data_list[idx])
        label_path = os.path.join(self.label_folder, f_name)
        object_list = self.config["objects"]
        reg_map = np.zeros(self.geometry['label_shape'], dtype=np.float32)
        cls_map = np.zeros((self.geometry['label_shape'][0], self.geometry['label_shape'][1]), dtype = np.int64)
        with open(label_path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            for line in lines:
                bbox = []
                entry = line.split(' ')
                name = entry[0]
                if name in list(object_list.keys()):
                    bbox.append(object_list[name])
                    bbox.extend([float(e) for e in entry[1:]])
                    corners, reg_target = self.get_corners(bbox)
                    self.update_label_map(reg_map, cls_map, corners, reg_target, object_list[name])

        return reg_map, cls_map

    def get_corners(self, bbox):
        h, w, l, x, y, z, yaw = bbox[1:]
        
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]

        return bev_corners, reg_target


    def update_label_map(self, reg_map, cls_map, bev_corners, reg_target, cls):
        label_corners = (bev_corners / 4 ) / 0.1
        label_corners[:, 1] += self.geometry['label_shape'][0] / 2

        points = get_points_in_a_rotated_box(label_corners, self.geometry['label_shape'])

        for p in points:
            label_x = p[0]
            label_y = p[1]
            metric_x, metric_y = trasform_label2metric(np.array(p))
            actual_reg_target = np.copy(reg_target)
            actual_reg_target[2] = reg_target[2] - metric_x
            actual_reg_target[3] = reg_target[3] - metric_y
            actual_reg_target[4] = np.log(reg_target[4])
            actual_reg_target[5] = np.log(reg_target[5])

            cls_map[label_y, label_x] = cls
            reg_map[label_y, label_x] = actual_reg_target



    def read_bbox(self, idx):
        object_list = self.config["objects"]

        corner_list = []
        class_list = []
        f_name = '{}.txt'.format(self.data_list[idx])
        label_path = os.path.join(self.label_folder, f_name)
        with open(label_path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            for line in lines:
                bbox = []
                entry = line.split(' ')
                name = entry[0]
                if name in list(object_list.keys()):
                    bbox.append(object_list[name])
                    bbox.extend([float(e) for e in entry[1:]])
                    class_list.append(object_list[name])
                    corners = self.get3D_corners(bbox)
                    corner_list.append(corners)
        return (class_list, corner_list)

    def get3D_corners(self, bbox):
        h, w, l, x, y, z, yaw = bbox[1:]

        corners = []
        front = l / 2
        back = -l / 2
        left = w / 2
        right = -w / 2
        top = h
        bottom = 0
        corners.append([front, left, top])
        corners.append([front, left, bottom])
        corners.append([front, right, bottom])
        corners.append([front, right, top])
        corners.append([back, left, top])
        corners.append([back, left, bottom])
        corners.append([back, right, bottom])
        corners.append([back, right, top])
        
        for i in range(8):
            corners[i] = self.rotate_pointZ(corners[i], yaw)[0] + np.array([x, y, z])
       
        return corners


    def rotate_pointZ(self, point, yaw):
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw), np.cos(yaw), 0],
                                    [0, 0, 1]])

        rotated_point = np.matmul(rotation_matrix, np.reshape(point, (3, 1)))
        return np.reshape(rotated_point, (1, 3))

    def create_data_list(self):
        data_list = []
        with open(self.data_file, "r") as f:
            for line in f:
                line = line.strip()
                data_list.append(line.split(";")[0])
        
        self.data_list = data_list


if __name__ == "__main__":
    pointcloud_folder = "/home/stpc/data/kitti/velodyne/training_reduced/velodyne"
    label_folder = "/home/stpc/data/kitti/label_2/training/label_2_reduced"
    data_file = "/home/stpc/data/train/train.txt"
    dataset = KittiDataset(pointcloud_folder, label_folder, data_file)

    for data in dataset:
        label = data["label"]
        voxel = data["voxel"]
        #voxel = voxel.permute(1, 2, 0)
        #print(voxel.shape)
        #print(torch.sum(voxel, axis = 2))
        #print(label[:, 0].shape)
        #imgplot = plt.imshow(label[:, :, 0])
        #plt.imshow(torch.sum(voxel, axis = 2), cmap="brg", vmin=0, vmax=255)
        #img = voxel_to_img(voxel)
        #imgplot = plt.imshow(img)
       # plt.show()
        break