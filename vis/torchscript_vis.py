import os
import time
import numpy as np
import json
import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Text

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

from utils.preprocess import voxelize, voxel_to_points
from utils.transform import corner_to_center_box2d
from utils.postprocess import non_max_suppression
from core.dataset import Dataset
from utils.one_hot import one_hot
from core.losses import CustomLoss
from core.models.mobilepixor_aux_torchscript import MobilePIXOR


class Vis():
    def __init__(self, data_loader, model, config, task = "val"):
        self.data_loader = data_loader
        self.model = model
        self.model.eval()
        self.config = config
        self.index = 0
        self.iter = iter(self.data_loader)
        self.task = task
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
        self.gt_bbox = vispy.scene.visuals.Line(parent=self.scan_view.scene)

        self.draw_gt = False
        self.use_current_data = False

        self.text = Text(parent=self.scan_view.scene, color='white', font_size = 50)

        self.canvas1 = SceneCanvas(keys='interactive',
                                show=True,
                                size=(200, 175))
        self.canvas1.events.key_press.connect(self._key_press)
        self.canvas1.events.draw.connect(self._draw)

        self.view = self.canvas1.central_widget.add_view()
        self.image = vispy.scene.visuals.Image(parent=self.view.scene)

        self.update_scan()




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



    def plot_boxes(self, class_list, scores, boxes):
        self.gt_bbox.visible = False
        if len(boxes) == 0:
            self.bbox.visible = False
            return

        object_colors = {1: np.array([1, 0, 0, 1]), 
                         2: np.array([0, 1, 0, 1]), 
                         3: np.array([0, 0, 1, 1]),
                         4: np.array([1, 1, 0, 1]),
                         5: np.array([1, 1, 1, 1])}
        connect = []
        points = []
        colors = []
        text = []
        text_pos = []
       
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

            text.append(str(scores[i])[:4])
            text_pos.append(corners[0])
            #Text(str(scores[i])[ :4], parent=self.scan_view.scene, color='white', pos = corners[0], font_size = 50)
        self.text.text = text
        self.text.pos = text_pos
        self.bbox.visible = True
        self.bbox.set_data(pos=points,
                            connect=connect,
                            color=colors)

    def plot_gt_boxes(self, class_list, boxes):
        self.bbox.visible = False
        if len(boxes) == 0:
            self.gt_bbox.visible = False
            return

        object_colors = {1: np.array([1, 0, 0, 1]), 
                         2: np.array([0, 1, 0, 1]), 
                         3: np.array([0, 0, 1, 1]),
                         4: np.array([1, 1, 0, 1]),
                         5: np.array([1, 1, 1, 1])}
        connect = []
        points = []
        colors = []

        for i in range(len(boxes)):
            box = boxes[i]
            if isinstance(class_list[i], float):
                break
            class_list[i] = class_list[i].tolist()[0]
            for j in range(len(box)):
                box[j] = box[j].numpy()[0]
       
       
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
        self.gt_bbox.visible = True
        self.gt_bbox.set_data(pos=points,
                            connect=connect,
                            color=colors)


    def get_point_color_using_intensity(self, points):
        scale_factor = 500
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



    def update_scan(self):
        if self.use_current_data:
            data = self.current_data
        else:
            data = next(self.iter)
            self.current_data = data

        geometry = self.config[data["dtype"][0]]["geometry"]
        x_min = geometry["x_min"]
        y_min = geometry["y_min"]
        x_res = geometry["x_res"]
        y_res = geometry["y_res"]
        
        voxel = data["voxel"]
        #voxel = voxel.half()
        #voxel = voxel.to("cuda:0")
        # boxes = self.model(voxel, x_min, y_min, x_res, y_res, 0.1).detach().cpu().numpy()
        pred = self.model(voxel, x_min, y_min, x_res, y_res, torch.tensor([1, 0.1, 0.1, 0.1, 0.1]), 1)
        # print(boxes.shape)
        # points = data["points"].squeeze().numpy()

        # box_list = []
        # class_list = []
        # scores = []
        # for i in range(boxes.shape[0]):
        #     box = boxes[i]
        #     box_list.append(self.get_corners(box))
        #     class_list.append(box[0])
        #     scores.append(box[1])

        # print(class_list) 
        cls = pred[:, 0].detach().cpu().numpy()
        scores = pred[:, 1].detach().cpu().numpy()
        corners = pred[:, 2:].reshape(-1, 4, 2).detach().cpu().numpy()

        selected_idxs = non_max_suppression(corners, scores, 0.1)
        selected_corners = corners[selected_idxs]        


        selected_boxes = corner_to_center_box2d(selected_corners)
        cls = cls[selected_idxs].reshape(-1, 1)
        scores = scores[selected_idxs].reshape(-1, 1)
        selected_boxes = np.concatenate((cls, scores , selected_boxes), axis = 1)

        points = data["points"].squeeze().numpy()

        box_list = []
        class_list = []
        scores = []
        for i in range(selected_boxes.shape[0]):
            box = selected_boxes[i]
            box_list.append(self.get_corners(box))
            class_list.append(box[0])
            scores.append(box[1])
        

        #colors = np.array([0, 1, 1])
        colors = self.get_point_color_using_intensity(points)
        
        self.canvas.title = str(self.index)
        self.scan_vis.set_data(points[:, :3],
                            face_color=colors,
                            edge_color=colors,
                            size=1.0)


        if not self.draw_gt or self.task == "test":
            self.plot_boxes(class_list, scores, box_list)
        else:
            self.plot_gt_boxes(data["cls_list"], data["boxes"])
        
        


    def _key_press(self, event):
        if event.key == 'N':
            self.use_current_data = False
            if self.index < len(self.data_loader) - 1:
                self.index += 1
            self.update_scan()

        if event.key == 'B':
            self.use_current_data = False
            if self.index > 0:
                self.index -= 1
            self.update_scan()

        if event.key == "G":
            self.draw_gt = not self.draw_gt
            self.use_current_data = True
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
    with open("/home/stpc/proj/object_detection/configs/more_classes.json", 'r') as f:
        config = json.load(f)
    #model_path = "/home/stpc/experiments/mobilepixor_more_classes_03-06-2022_1/174epoch"
    model_path = "/home/stpc/experiments/mobilepixor_aux_17-06-2022_1/354epoch"

    data_file = "/home/stpc/clean_data/list/small_robot_test.txt"
    dataset = Dataset(data_file, config["data"], config["augmentation"], "test")
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1)

    model = MobilePIXOR()
    #model.to(config['device'])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = config["device"]

    #model_path = "/home/stpc/models/pixor_half.pt"
    # model_path = "/home/stpc/models/mobilepixor.pt"
    # model = torch.jit.load(model_path)
    # model.to("cuda:0")
    vis = Vis(data_loader, model, config["data"])
    vis.run()

        
    
    
