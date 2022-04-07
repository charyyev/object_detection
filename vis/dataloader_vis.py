import os
import numpy as np
import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas
import time

from utils.preprocess import voxelize, voxel_to_points
from core.kitti_dataset import KittiDataset


class Vis():
    def __init__(self, dataset):
        self.dataset = dataset
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

        self.canvas1 = SceneCanvas(keys='interactive',
                                show=True,
                                size=(200, 175))
        self.canvas1.events.key_press.connect(self._key_press)
        self.canvas1.events.draw.connect(self._draw)

        self.view = self.canvas1.central_widget.add_view()
        self.image = vispy.scene.visuals.Image(parent=self.view.scene)

        

        self.update_scan()



    def plot_boxes(self, class_list, boxes):
        object_colors = {1: np.array([1, 0, 0, 1]), 2: np.array([0, 1, 0, 1]), 3:np.array([0, 0, 1, 1])}
        connect = []
        points = []
        colors = []
       
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

        self.bbox.set_data(pos=points,
                            connect=connect,
                            color=colors)


    def update_scan(self):
        data = self.dataset[self.index]
        voxel = data["voxel"].permute(2, 1, 0).numpy()
        class_list = data["class_list"]
        boxes = data["boxes"]
        cls_map = data["cls_map"]
        points = voxel_to_points(voxel)
        
        #colors = self.get_point_color_using_intensity(points)
        colors = np.array([0, 1, 0])
        
        self.canvas.title = str(self.index)
        self.scan_vis.set_data(points[:, :3],
                            face_color=colors,
                            edge_color=colors,
                            size=1.0)

        self.plot_boxes(class_list, boxes)

        self.image.set_data(np.swapaxes(cls_map, 0, 1))




    def _key_press(self, event):
        if event.key == 'N':
            if self.index < len(self.dataset) - 1:
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
    pointcloud_folder = "/home/stpc/data/kitti/velodyne/training_reduced/velodyne"
    label_folder = "/home/stpc/data/kitti/label_2/training/label_2_reduced"
    data_file = "/home/stpc/data/train/train.txt"

    dataset = KittiDataset(pointcloud_folder, label_folder, data_file)
    
    vis = Vis(dataset)
    vis.run()
