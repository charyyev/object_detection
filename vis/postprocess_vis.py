import os
import numpy as np
import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas

from torch.utils.data import Dataset, DataLoader
import torch

from utils.preprocess import voxelize, voxel_to_points
from utils.postprocess import filter_pred
from core.kitti_dataset import KittiDataset
from utils.one_hot import one_hot


class Vis():
    def __init__(self, voxels, boxess):
        self.voxels = voxels
        self.boxess = boxess
        self.index = 0
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

        self.update_scan()




    def get_corners(self, bbox):
        #h, w, l, x, y, z, yaw = bbox[1:]
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



    def plot_boxes(self, class_list, boxes):
        object_colors = {1: np.array([1, 0, 0, 1]), 2: np.array([0, 1, 0, 1]), 3:np.array([0, 0, 1, 1])}
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


    def update_scan(self):
        voxel = self.voxels[self.index]
        boxes = self.boxess[self.index]
        points = voxel_to_points(voxel)
        box_list = []
        class_list = []

        for i in range(boxes.shape[0]):
            box = boxes[i]
            box_list.append(self.get_corners(box))
            class_list.append(box[0])
        

        colors = np.array([0, 0, 1])
        
        self.canvas.title = str(self.index)
        self.scan_vis.set_data(points[:, :3],
                            face_color=colors,
                            edge_color=colors,
                            size=1.0)

        self.plot_boxes(class_list, box_list)


    def _key_press(self, event):
        if event.key == 'N':
            if self.index < len(voxels) - 1:
                self.index += 1
            self.update_scan()

        if event.key == 'B':
            if self.index > 0:
                self.index -= 1
            self.update_scan()

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
    geom = {
        "L1": -40.0,
        "L2": 40.0,
        "W1": 0.0,
        "W2": 70.0,
        "H1": -2.5,
        "H2": 1.0,
        "input_shape": [800, 700, 35],
        "label_shape": [200, 175, 7]
    }

    pointcloud_folder = "/home/stpc/data/kitti/velodyne/training_reduced/velodyne"
    label_folder = "/home/stpc/data/kitti/label_2/training/label_2_reduced"
    data_file = "/home/stpc/data/train/train.txt"
    dataset = KittiDataset(pointcloud_folder, label_folder, data_file)

    data_loader = DataLoader(dataset, shuffle=True, batch_size=1)

    voxels = []
    boxess = []
    
    count = 0
    max_num = 20
    #net = PIXOR(geom, use_bn=False)

    for data in data_loader:
        cls_one_hot = one_hot(data["cls_map"], num_classes= 4 , device="cpu", dtype=data["cls_map"].dtype)
        boxes = filter_pred(data["reg_map"].numpy(), cls_one_hot.numpy(), geom)
        voxels.append(torch.squeeze(data["voxel"]).permute(2, 1, 0).numpy())
        boxess.append(boxes)

        #imgplot = plt.imshow(torch.squeeze(data["cls_map"]).permute(0, 1))
        #plt.show()
        #preds = net(data["voxel"])
        #print(preds["reg_map"].shape)
        #print(data["reg_map"].shape)
        count += 1
        if count >= max_num:
            break

    vis = Vis(voxels, boxess)
    vis.run()
        
    
    
