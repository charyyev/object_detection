import os
import numpy as np
import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas
import json

from utils.preprocess import voxelize, voxel_to_points
from core.hotspot_dataset import HotSpotDataset


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
        object_colors = {1: np.array([1, 0, 0, 1]), 
                         2: np.array([0, 1, 0, 1]), 
                         3: np.array([0, 0, 1, 1]),
                         4: np.array([1, 1, 0, 1]),
                         5: np.array([1, 1, 1, 1])}
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


    def voxel_to_points_color(self, voxel, geometry, cls_map, hotspot_mask, quad_map):
        x_min = geometry["x_min"]
        x_max = geometry["x_max"]
        y_min = geometry["y_min"]
        y_max = geometry["y_max"]
        z_min = geometry["z_min"]
        z_max = geometry["z_max"]
        x_res = geometry["x_res"]
        y_res = geometry["y_res"]
        z_res = geometry["z_res"]

        quad_colors = [[1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]

        cp_voxel = np.zeros(voxel.shape, dtype = np.int16)

        hotspot = hotspot_mask * cls_map
        print(hotspot.shape)
        h_y, h_x = np.where(hotspot > 0)

        for i in range(h_y.shape[0]):
            x = h_x[i]
            y = h_y[i]
            cp_voxel[4 * x: 4 * x + 4, 4 * y:4 * y + 4, :] = quad_map[y][x] + 1



        xs, ys, zs = np.where(voxel.astype(int) == 1)
        points_x = xs + x_res / 2
        points_y = ys + y_res / 2
        points_z = zs + z_res / 2

        points_x = points_x * x_res + x_min
        points_y = points_y * y_res + y_min
        points_z = points_z * z_res + z_min
        points = np.transpose(np.array([points_x, points_y, points_z]))

        colors = np.zeros((xs.shape[0], 3))
        colors[:, 1] = 1

        colors[cp_voxel[xs, ys, zs] == 1] = quad_colors[0]
        colors[cp_voxel[xs, ys, zs] == 2] = quad_colors[1]
        colors[cp_voxel[xs, ys, zs] == 3] = quad_colors[2]
        colors[cp_voxel[xs, ys, zs] == 4] = quad_colors[3]
        return points, colors

    def update_scan(self):
        data = self.dataset[self.index]
        voxel = data["voxel"].permute(2, 1, 0).numpy()
        class_list = data["cls_list"]
        boxes = data["boxes"]
        cls_map = data["cls_map"]
        quad_map = data["quad_map"]
        hotspot_mask = data["hotspot_mask"]
        data_type = data["dtype"]
        points, colors = self.voxel_to_points_color(voxel, dataset.config[data_type]["geometry"], cls_map, hotspot_mask, quad_map)
        #points = data["points"]
        #print(boxes)
        #colors = self.get_point_color_using_intensity(points)
        #colors = np.array([0, 1, 0])
        
        self.canvas.title = str(self.index) + ": " + data_type
        self.scan_vis.set_data(points[:, :3],
                            face_color=colors,
                            edge_color=colors,
                            size=1.0)

        self.plot_boxes(class_list, boxes)
        # sub_map = data["sub_map"]
        # color_img = np.zeros((cls_map.shape[0], cls_map.shape[1], 3))

        # color_img[:, :, 0][cls_map == 1] = 1
        # color_img[:, :, 1][cls_map == 1] = 0
        # color_img[:, :, 2][cls_map == 1] = 0
        # color_img[:, :, 0][cls_map == 2] = 0
        # color_img[:, :, 1][cls_map == 2] = 1
        # color_img[:, :, 2][cls_map == 2] = 0
        # color_img[:, :, 0][cls_map == 3] = 1
        # color_img[:, :, 1][cls_map == 3] = 1
        # color_img[:, :, 2][cls_map == 3] = 1
        # color_img[:, :, 0][sub_map == 0] = 0
        # color_img[:, :, 1][sub_map == 0] = 1
        # color_img[:, :, 2][sub_map == 0] = 1
        

        self.image.set_data(np.swapaxes( hotspot_mask * cls_map, 0, 1))


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
    data_file = "/home/stpc/clean_data/list/val.txt"
    #data_file = "/home/stpc/clean_data/list/train_small.txt"
    with open("/home/stpc/proj/object_detection/configs/hotspot.json", 'r') as f:
        config = json.load(f)

    dataset = HotSpotDataset(data_file, config["data"], config["augmentation"], "val")

    vis = Vis(dataset)
    vis.run()
