import numpy as np
from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt
import math
import json
import time

from utils.preprocess import trasform_label2metric, get_points_in_a_rotated_box
from utils.transform import Random_Rotation, Random_Scaling, OneOf, Random_Translation


class HotSpotDataset(Dataset):
    def __init__(self, data_file, config, aug_config, task = "train") -> None:
        self.data_file = data_file

        self.create_data_list()

        self.config = config
        self.task = task
        self.transforms = self.get_transforms(aug_config)
        self.augment = OneOf(self.transforms, aug_config["p"])
       

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file = '{}.bin'.format(self.data_list[idx])
        data_type = self.data_type_list[idx]

        pointcloud_folder = os.path.join(self.config[data_type]["location"], "pointcloud")
        lidar_path = os.path.join(pointcloud_folder, file)
        #print(lidar_path)
        points = self.read_points(lidar_path)

        if self.task == "test":
            scan = self.voxelize(points, self.config[data_type]["geometry"])
            scan = torch.from_numpy(scan)
            scan = scan.permute(2, 0, 1)
            return {
                "voxel": scan,
                "points": points,
                "dtype": data_type
                }

        boxes = self.get_boxes(idx)

        if self.task == "train" and boxes.shape[0] != 0:
            points, boxes[:, 1:] = self.augment(points, boxes[:, 1:8])

        scan = self.voxelize(points, self.config[data_type]["geometry"])
        scan = torch.from_numpy(scan)
        self.scan = scan.permute(2, 0, 1)
        target = self.get_label(boxes, self.config[data_type]["geometry"])
        target["reg_map"] = torch.from_numpy(target["reg_map"]).permute(2, 0, 1)
        
        if self.task == "val":
            class_list, boxes = self.read_bbox(boxes)
    
            return {
                "voxel": self.scan, 
                "reg_map": target["reg_map"],
                "cls_map": target["cls_map"],
                "argmin_map": target["argmin_map"],
                "quad_map": target["quad_map"],
                "hotspot_mask": target["hotspot_mask"],
                "cls_list": class_list,
                "points": points,
                "boxes": boxes,
                "dtype": data_type
                }   

        return {
            "voxel": self.scan,
            "reg_map": target["reg_map"],
            "cls_map": target["cls_map"],
            "argmin_map": target["argmin_map"],
            "quad_map": target["quad_map"],
            "hotspot_mask": target["hotspot_mask"],
            }

            

    def read_points(self, lidar_path):
        return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    def voxelize(self, points, geometry):
        x_min = geometry["x_min"]
        x_max = geometry["x_max"]
        y_min = geometry["y_min"]
        y_max = geometry["y_max"]
        z_min = geometry["z_min"]
        z_max = geometry["z_max"]
        x_res = geometry["x_res"]
        y_res = geometry["y_res"]
        z_res = geometry["z_res"]

        x_size = int((x_max - x_min) / x_res)
        y_size = int((y_max - y_min) / y_res)
        z_size = int((z_max - z_min) / z_res)

        eps = 0.001

        #clip points
        x_indexes = np.logical_and(points[:, 0] > x_min + eps, points[:, 0] < x_max - eps)
        y_indexes = np.logical_and(points[:, 1] > y_min + eps, points[:, 1] < y_max - eps)
        z_indexes = np.logical_and(points[:, 2] > z_min + eps, points[:, 2] < z_max - eps)
        pts = points[np.logical_and(np.logical_and(x_indexes, y_indexes), z_indexes)]

        occupancy_mask = np.zeros((pts.shape[0], 3), dtype = np.int32)
        voxels = np.zeros((x_size, y_size, z_size), dtype = np.float32)
        occupancy_mask[:, 0] = (pts[:, 0] - x_min) // x_res
        occupancy_mask[:, 1] = (pts[:, 1] - y_min) // y_res
        occupancy_mask[:, 2] = (pts[:, 2] - z_min) // z_res

        idxs = np.array([occupancy_mask[:, 0].reshape(-1), occupancy_mask[:, 1].reshape(-1), occupancy_mask[:, 2].reshape( -1)])

        voxels[idxs[0], idxs[1], idxs[2]] = 1
        return np.swapaxes(voxels, 0, 1)

    def is_hotspot(self, x, y, z_min, z_max):
        return torch.any(self.scan[z_min:z_max, y: y + 4, x: x + 4] == 1)


    def get_boxes(self, idx):
        '''
        :param i: the ith velodyne scan in the train/val set
        : return boxes of shape N:8

        '''

        f_name = '{}.txt'.format(self.data_list[idx])
        data_type = self.data_type_list[idx]
        label_folder = os.path.join(self.config[data_type]["location"], "label")
        label_path = os.path.join(label_folder, f_name)
        object_list = self.config[data_type]["objects"]
        boxes = []

        with open(label_path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            for line in lines:
                bbox = []
                entry = line.split(' ')
                name = entry[0]
                if name in list(object_list.keys()):
                    bbox.append(object_list[name])
                    bbox.extend([float(e) for e in entry[1:]])
                    boxes.append(bbox)

        return np.array(boxes)

    def get_label(self, boxes, geometry):
        '''
        :param boxes: numpy array of shape N:8
        :return: label map: <--- This is the learning target
                a tensor of shape 200 * 175 * 6 representing the expected output
        '''
        reg_map = np.zeros((geometry['label_shape'][0], geometry['label_shape'][1], 4), dtype=np.float32)
        cls_map = np.zeros((geometry['label_shape'][0], geometry['label_shape'][1]), dtype = np.int64)
        x_argmin_map = np.zeros((geometry['label_shape'][0], geometry['label_shape'][1]), dtype = np.int64)
        y_argmin_map = np.zeros((geometry['label_shape'][0], geometry['label_shape'][1]), dtype = np.int64)
        quad_map = np.zeros((geometry['label_shape'][0], geometry['label_shape'][1]), dtype = np.int64)
        hotspot_mask = np.ones((geometry['label_shape'][0], geometry['label_shape'][1]), dtype = np.int64)
        for i in range(boxes.shape[0]):
            self.update_label_map(reg_map, cls_map, x_argmin_map, y_argmin_map, quad_map, hotspot_mask, boxes[i], geometry)

        target = {
            "reg_map": reg_map,
            "cls_map": cls_map,
            "argmin_map": [x_argmin_map, y_argmin_map],
            "quad_map": quad_map,
            "hotspot_mask": hotspot_mask
        }
        return target


    def get_corners(self, bbox):
        h, w, l, x, y, z, yaw = bbox[1:]
        yaw2 = math.fmod(2 * yaw, 2 * math.pi)
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

        reg_target = [np.cos(yaw2), np.sin(yaw2), x, y, w, l]

        return bev_corners, reg_target


    def update_label_map(self, reg_map, cls_map, x_argmin_map, y_argmin_map, quad_map, hotspot_mask, box, geometry):
        bev_corners, reg_target = self.get_corners(box)
        cls, h, w, l, cx, cy, cz, yaw = box

        area = (w * l) / 2
        num_hotspots = self.config["c"] / area

        
        z_min = max(geometry["z_min"], cz)
        z_max = min(geometry["z_max"], cz + h)

        z_min = int((z_min - geometry["z_min"]) / geometry["z_res"])
        z_max = int((z_max - geometry["z_min"]) / geometry["z_res"])

        bx = int((cx - geometry["x_min"]) / geometry["x_res"] / 4)
        by = int((cy - geometry["y_min"]) / geometry["y_res"] / 4)

        label_corners = np.zeros((4, 2))
        label_corners[:, 0] = (bev_corners[:, 0] - geometry["x_min"]) / geometry["x_res"]
        label_corners[:, 1] = (bev_corners[:, 1] - geometry["y_min"]) / geometry["y_res"]
        label_corners /= 4
        
        points = get_points_in_a_rotated_box(label_corners, geometry['label_shape'])

        sorted_points = sorted(points, key=lambda p: (p[0] - bx)**2 + (p[1] - by)**2)
        num_points = 0
        for p in sorted_points:
            if num_points < num_hotspots:
                label_x = p[0]
                label_y = p[1]
                metric_x, metric_y = trasform_label2metric(np.array(p), geometry)
                if self.is_hotspot(label_x * 4, label_y * 4, z_min, z_max):
                    #actual_reg_target = np.copy(reg_target)
                    actual_reg_target = np.zeros(4)
                    actual_reg_target[0] = reg_target[0]
                    actual_reg_target[1] = reg_target[1]
                    actual_reg_target[2] = np.log(reg_target[4])
                    actual_reg_target[3] = np.log(reg_target[5])

                    cls_map[label_y, label_x] = cls
                    reg_map[label_y, label_x] = actual_reg_target
                    
                    # argmin softmax
                    bin_res = (self.config["bin_max"] - self.config["bin_min"]) / self.config["num_bin"]
                    dx = reg_target[2] - metric_x
                    dy = reg_target[3] - metric_y
                    x_argmin_map[label_y, label_x] = (dx - self.config["bin_min"]) // bin_res
                    y_argmin_map[label_y, label_x] = (dy - self.config["bin_min"]) // bin_res


                    # quadrant
                    min_dist = 10000
                    quad = 0
                    for i in range(4):
                        if ((label_corners[i, 0] - label_x) ** 2 + (label_corners[i, 1] - label_y) ** 2) < min_dist:
                            min_dist = ((label_corners[i, 0] - label_x) ** 2 + (label_corners[i, 1] - label_y) ** 2)
                            quad = i

                    quad_map[label_y, label_x] = quad

                    num_points += 1
                else:
                    hotspot_mask[label_y, label_x] = 0
            else:
                hotspot_mask[p[1], p[0]] = 0


    def sub_sample_mask(self, boxes, geometry):
        mask = np.ones((geometry['label_shape'][0], geometry['label_shape'][1]), dtype = np.int64)
        inner_radius_coeff = 0.3
        outer_radius_coeff = 1.2
        for i in range(boxes.shape[0]):
            box = boxes[i]
            
            # We will mask only cars
            if box[0] != 1:
                continue

            # convert box center to bev
            x, y = box[4:6]
            x = (x - geometry["x_min"]) / geometry["x_res"] / 4
            y = (y - geometry["y_min"]) / geometry["y_res"] / 4            
            #y, x = x, y
          
            # inner and outer radius of circle
            w = box[3] / 4 / 2 / geometry["x_res"]

            r_in = inner_radius_coeff * w
            r_out = outer_radius_coeff * w
            x_min = max(0, int(x - r_out))
            x_max = min(geometry["label_shape"][1], int(x + r_out))

            y_min = max(0, int(y - r_out))
            y_max = min(geometry["label_shape"][0], int(y + r_out))


            Y, X = np.ogrid[y_min:y_max, x_min:x_max]
            dist_from_center = np.sqrt((X - x)**2 + (Y-y)**2)

            mask[y_min:y_max, x_min:x_max][dist_from_center <= r_out] = 0
            mask[y_min:y_max, x_min:x_max][dist_from_center <= r_in] = 1

        return mask



    def read_bbox(self, boxes):
        corner_list = []
        class_list = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            class_list.append(box[0])
            corners = self.get3D_corners(box)
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

    def get_transforms(self, config):
        transforms = []
        if config["rotation"]["use"]:
            limit_angle = config["rotation"]["limit_angle"]
            p = config["rotation"]["p"]
            transforms.append(Random_Rotation(limit_angle, p))
        if config["scaling"]["use"]:
            range = config["scaling"]["range"]
            p = config["scaling"]["p"]
            transforms.append(Random_Scaling(range, p))

        if config["translation"]["use"]:
            scale = config["translation"]["scale"]
            p = config["translation"]["p"]
            transforms.append(Random_Translation(scale, p))

        return transforms

    def create_data_list(self):
        data_list = []
        data_type_list = []
        with open(self.data_file, "r") as f:
            for line in f:
                line = line.strip()
                data, data_type = line.split(";")
                data_list.append(data)
                data_type_list.append(data_type)
        
        self.data_list = data_list
        self.data_type_list = data_type_list


if __name__ == "__main__":
    data_file = "/home/stpc/clean_data/list/train.txt"

    with open("/home/stpc/proj/object_detection/configs/mixed_data.json", 'r') as f:
        config = json.load(f)

    dataset = Dataset(data_file, config["data"], config["augmentation"])

    # for data in dataset:
    #     label = data["label"]
    #     voxel = data["voxel"]
    #     #voxel = voxel.permute(1, 2, 0)
    #     #print(voxel.shape)
    #     #print(torch.sum(voxel, axis = 2))
    #     #print(label[:, 0].shape)
    #     #imgplot = plt.imshow(label[:, :, 0])
    #     #plt.imshow(torch.sum(voxel, axis = 2), cmap="brg", vmin=0, vmax=255)
    #     #img = voxel_to_img(voxel)
    #     #imgplot = plt.imshow(img)
    #    # plt.show()
    #     break