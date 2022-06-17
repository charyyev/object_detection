import numpy as np
import rosbag
import json
import torch
import torch.nn.functional as F

import os
import numpy as np
import matplotlib.pyplot as plt


from core.models.pixor import PIXOR
from core.models.mobilepixor import MobilePIXOR
from tools.lidar_conversion import pcl_to_numpy
from utils.preprocess import voxelize
from utils.postprocess import filter_pred

class Labeler():
    def __init__(self, bag_file, device, model, geometry, categories, save_folder):
        self.bag_file = bag_file
        self.device = device
        self.model = model
        self.geometry = geometry
        self.categories = categories
        self.save_folder = save_folder

        self.label()

    def label(self):
        if self.bag_file is not None:
            max_frames = 15000
            bag = rosbag.Bag(bag_file)
            lidar_topics = ["/velodyne_points"]
    
            count = 1
            for topic, msg, t in bag.read_messages():
                if count > max_frames:
                    break
                if topic in lidar_topics:
                    lidar_path = os.path.join(self.save_folder, "pointcloud")
                    label_path = os.path.join(self.save_folder, "label")
                    lidar_name = '%06d' % count + ".bin"
                    label_name = '%06d' % count + ".txt"
                    lidar_save = os.path.join(lidar_path, lidar_name)
                    label_save = os.path.join(label_path, label_name)

                    # save points
                    points = pcl_to_numpy(msg, 4)
                    points.astype("float32").tofile(lidar_save)

                    self.label_one_frame(points, label_save)
                    count += 1
                    
    
                    

    def label_one_frame(self, points, label_save):
        voxel = voxelize(points, self.geometry)
        voxel = torch.from_numpy(voxel)
        voxel = voxel.to(self.device)
        voxel = voxel.permute(2, 0, 1).unsqueeze(0)
        
        pred = self.model(voxel)
        pred["cls_map"] = F.softmax(pred["cls_map"], dim=1)
        reg_pred = pred["reg_map"].detach().cpu().numpy()
        cls_pred = pred["cls_map"].detach().cpu().numpy()
        config = {"geometry": self.geometry}
        boxes = filter_pred(reg_pred, cls_pred, config, score_threshold=0.5, nms_threshold=0.1)

        with open(label_save, "w") as f:
            for i in range(boxes.shape[0]):
                bbox = boxes[i]
                #cls, h, w, l, x, y, z, yaw = bbox
                cls, scores, x, y, l ,w, yaw = bbox
                line = self.categories[int(cls)] + " 0 " + str(w) + " " + str(l) + " " + str(x) + " " + str(y) + " 0 " + str(yaw)
                f.write(line)
                if i != boxes.shape[0]:
                    f.write("\n")

    



if __name__ == "__main__":
    with open("/home/stpc/proj/object_detection/configs/fine_tune.json", 'r') as f:
        config = json.load(f)

    model_path = "/home/stpc/experiments/mobilepixor_small_robot_15-06-2022_1/checkpoints/338epoch"
    #model_path = "/home/stpc/experiments/pixor_mixed_19-04-2022_1/159epoch"
    device = "cuda:0"
    data_type = "small_robot"
    model_type = "mobilepixor"
    geometry = config["data"][data_type]["geometry"]

    bag_file = "/home/stpc/rosbags/ai_2022-06-14-17-11-07.bag"
    categories = {1: "Car",
                  2: "Pedestrian",
                  3: "Motorcycle",
                  4: "Bus"}

    save_folder = "/home/stpc/data/auto/"
    if not os.path.exists(os.path.join(save_folder, "pointcloud")):
        os.makedirs(os.path.join(save_folder, "pointcloud"))
    if not os.path.exists(os.path.join(save_folder, "label")):
        os.makedirs(os.path.join(save_folder, "label"))

    if model_type == "mobilepixor":
        model = MobilePIXOR(config["data"]["kitti"]["geometry"])
    elif model_type == "pixor":
        model = PIXOR(config["data"]["kitti"]["geometry"])
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    Labeler(bag_file, device, model, geometry, categories, save_folder)