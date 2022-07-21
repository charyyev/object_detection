import numpy as np
import rosbag
import json
import torch
import torch.nn.functional as F

import os
import numpy as np
import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas
from vispy import app
from vispy.scene.visuals import Text

from core.models.pixor import PIXOR
from core.models.mobilepixor_aux import MobilePIXOR
from core.models.afdet import AFDet
from tools.lidar_conversion import pcl_to_numpy
from utils.preprocess import voxelize
from utils.postprocess import filter_pred, filter_pred_aux, filter_pred_afdet


class Vis():
    def __init__(self, bag_name, model, device, geometry, frames):
        self.index = 0
        self.frames = frames
        self.bag_name = bag_name
        self.model = model
        self.device = device
        self.geometry = geometry
        self.canvas = SceneCanvas(keys='interactive',
                                show=True,
                                size=(1600, 900))
        self.canvas.events.key_press.connect(self._key_press)
        self.canvas.events.draw.connect(self._draw)

        self.grid = self.canvas.central_widget.add_grid()
        self.scan_view = vispy.scene.widgets.ViewBox(parent=self.canvas.scene,
                                                    camera=TurntableCamera(distance=30.0))
        self.grid.add_widget(self.scan_view)
        self.scan_vis = visuals.Markers()
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)

        self.bbox = vispy.scene.visuals.Line(parent=self.scan_view.scene)
        
        self.timer_interval = 0.1
        self.timer = app.Timer(interval=self.timer_interval, connect=self.update_scan, start=True)
        self.paused = False
   


    def get_point_color_using_intensity(self, points):
        scale_factor = 100
        scaled_intensity = np.clip(points[:, 3] * scale_factor, 0, 255)
        scaled_intensity = scaled_intensity.astype(np.uint8)
        cmap = plt.get_cmap("viridis")

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        color_range = color_range.reshape(256, 3).astype(np.float32) / 255.0
        colors = color_range[scaled_intensity]
        return colors

    
    def update_scan(self, event):
        if self.index == len(self.frames):
            self.destroy()
            return

        points = self.frames[self.index]
        voxel = voxelize(points, geometry)
        voxel = torch.from_numpy(voxel)
        voxel = voxel.to(self.device)
        voxel = voxel.permute(2, 0, 1).unsqueeze(0)
        
        pred = self.model(voxel)

        offset_pred = pred["offset"]
        size_pred = pred["size"]
        yaw_pred = pred["yaw"]
        cls_pred = pred["cls"]
        
        threshold = 0.35
        config = {"geometry": self.geometry}
        boxes = filter_pred_afdet(cls_pred, offset_pred, size_pred, yaw_pred, config, threshold)

        colors = self.get_point_color_using_intensity(points)
        #colors = np.array([0, 1, 1])

        box_list = []
        class_list = []
        scores = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            box_list.append(self.get_corners3D(box))
            class_list.append(box[0])
            scores.append(box[1])

        if not self.paused:
            self.index += 1

        self.canvas.title = f"Frame: {self.index} / {len(self.frames)}"
        self.scan_vis.set_data(points[:, :3],
                            face_color=colors,
                            edge_color=colors,
                            size=1.0)

        self.plot_boxes3D(class_list, scores, box_list)


    def plot_boxes(self, class_list, scores, boxes):
        if len(boxes) == 0:
            self.bbox.set_data(pos=[],
                            connect=[],
                            color=[])
            return

        object_colors = {1: np.array([1, 0, 0, 1]), 
                         2: np.array([0, 1, 0, 1]), 
                         3: np.array([0, 0, 1, 1]),
                         4: np.array([1, 1, 0, 1]),
                         5: np.array([1, 1, 1, 1])}
        connect = []
        points = []
        colors = []

       
        for i, box in enumerate(boxes):
            color = np.tile(object_colors[class_list[i]], (4, 1))
            corners = np.array(box)
            j = 4 * i
            con = [[j, j + 1],
                   [j + 1, j + 2],
                   [j + 2, j + 3],
                   [j + 3, j]]
            con = np.array(con)

            if i == 0:
                points = corners
                connect = con
                colors = color
            else:
                points = np.concatenate((points, corners), axis = 0)
                connect = np.concatenate((connect, con), axis = 0)
                colors = np.concatenate((colors, color), axis = 0)


        self.bbox.set_data(pos=points,
                            connect=connect,
                            color=colors)


    def plot_boxes3D(self, class_list, scores, boxes):
        if len(boxes) == 0:
            self.bbox.set_data(pos=[],
                            connect=[],
                            color=[])
            return

        object_colors = {0: np.array([1, 0, 0, 1]), 
                         1: np.array([0, 1, 0, 1]), 
                         2: np.array([0, 0, 1, 1]),
                         3: np.array([1, 1, 0, 1]),
                         4: np.array([1, 1, 1, 1])}
        connect = []
        points = []
        colors = []
        text = []
        text_pos = []
       
        for i, box in enumerate(boxes):
            color = np.tile(object_colors[class_list[i]], (8, 1))
            corners = np.array(box)
            j = 8 * i
            con = [[j, j + 1],
                   [j + 1, j + 2],
                   [j + 2, j + 3],
                   [j + 3, j],
                   [j + 4, j + 5],
                   [j + 5, j + 6],
                   [j + 6, j + 7],
                   [j + 7, j + 4],
                   [j, j + 4],
                   [j + 1, j + 5],
                   [j + 2, j + 6],
                   [j + 3, j + 7]]
            con = np.array(con)

            if i == 0:
                points = corners
                connect = con
                colors = color
            else:
                points = np.concatenate((points, corners), axis = 0)
                connect = np.concatenate((connect, con), axis = 0)
                colors = np.concatenate((colors, color), axis = 0)

        self.bbox.visible = True
        self.bbox.set_data(pos=points,
                            connect=connect,
                            color=colors)

    def get_corners3D(self, bbox):
        cls, scores, x, y, l ,w, yaw = bbox

        corners = []
        front = l / 2
        back = -l / 2
        left = w / 2
        right = -w / 2
        top = 1
        bottom = -1

        corners.append([front, left, top])
        corners.append([front, left, bottom])
        corners.append([front, right, bottom])
        corners.append([front, right, top])
        corners.append([back, left, top])
        corners.append([back, left, bottom])
        corners.append([back, right, bottom])
        corners.append([back, right, top])
        
        for i in range(8):
            corners[i] = self.rotate_pointZ(corners[i], yaw)[0] + np.array([x, y, 0])
       
        return corners


    def get_corners(self, bbox):
        cls, scores, x, y, l ,w, yaw = bbox

        corners = []
        front = l / 2
        back = -l / 2
        left = w / 2
        right = -w / 2

        corners.append([front, left, 0])
        corners.append([back, left, 0])
        corners.append([back, right, 0])
        corners.append([front, right, 0])
        
        for i in range(4):
            corners[i] = self.rotate_pointZ(corners[i], yaw)[0] + np.array([x, y, 0])
       
        return corners

    def rotate_pointZ(self, point, yaw):
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw), np.cos(yaw), 0],
                                    [0, 0, 1]])

        rotated_point = np.matmul(rotation_matrix, np.reshape(point, (3, 1)))
        return np.reshape(rotated_point, (1, 3))

    def extract_bag(self):
        bag = rosbag.Bag(self.bag_file)
        for topic, msg, t in bag.read_messages():
            if topic in self.topics:
                self.frames.put(pcl_to_numpy(msg))

        self.frames.put("Done")


    def _key_press(self, event):
        if event.key == 'Right':
            if self.paused:
                self.index += 1
            self.update_scan(event)

        if event.key == 'Left':
            if self.paused and self.index > 0:
                self.index -= 1
            self.update_scan(event)

        if event.key == 'Space':
            if self.paused:
                self.timer.start()
            else:
                self.timer.stop()

            self.paused = not self.paused

        if event.key == "W":
            self.timer_interval -= 0.01
            self.timer_interval = max(self.timer_interval, 0)
            self.timer.interval = self.timer_interval

        if event.key == "S":
            self.timer_interval += 0.01
            self.timer.interval = self.timer_interval

        if event.key == "Down":
            if self.paused:
                self.save_frame()
                self.text.text = "Frame {} has been saved".format(self.index)
            else:
                print("Pause first to save the frame: Use space key to pause")     

        if event.key == 'Q':
            self.destroy()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        vispy.app.quit()

    def _draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def run(self):
        self.canvas.app.run()



if __name__ == "__main__":
    bag_file = "/home/stpc/rosbags/pangyo.bag"
    bag_name = bag_file.split("/")[-1].split(".")[0]
    with open("/home/stpc/proj/object_detection/configs/afdet_half_range.json", 'r') as f:
        config = json.load(f)

    model_path = "/home/stpc/experiments/scconv/107epoch"
    #model_path = "/home/stpc/experiments/afdet_half/117epoch"
    device = "cuda:0"
    data_type = "small_robot"
    model_type = "mobilepixor"
    geometry = config["data"][data_type]["geometry"]
    
    max_frames = 5000
    bag = rosbag.Bag(bag_file)
    lidar_topics = ["/points_raw"]
    frames = []
    for topic, msg, t in bag.read_messages():
        if len(frames) > max_frames:
            break
        if topic in lidar_topics:
            frames.append(pcl_to_numpy(msg))

    print("Done reading frames: ", len(frames), "frames in total")

    # if model_type == "mobilepixor":
    #     model = MobilePIXOR(config["data"]["kitti"]["geometry"])
    # elif model_type == "pixor":
    #     model = PIXOR(config["data"]["kitti"]["geometry"])
    model = AFDet(config["data"]["num_classes"])
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    vis = Vis(bag_name, model, device, geometry, frames)
    vis.run()
