import os
import numpy as np
import json
import open3d as o3d
import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas

from scipy.spatial.transform import Rotation as R


class Vis():
    def __init__(self, data_folder, labels):
        self.index = 0
        self.lidar_files = sorted(os.listdir(data_folder))
        self.labels = labels
        self.data_folder = data_folder

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


    def read_points(self, lidar_path):
        pcd = o3d.io.read_point_cloud(lidar_path)
        return np.asarray(pcd.points)


    def get_point_color_using_intensity(self, points):
        scale_factor = 10
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



    def get_corners(self, bbox):
        h, w, l, x, y, z, yaw = bbox[1:]

        corners = []
        front = l / 2
        back = -l / 2
        left = w / 2
        right = -w / 2
        top = h / 2
        bottom = -h / 2
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



    def plot_boxes(self, class_list, boxes):
        object_colors = {1: np.array([1, 0, 0, 1]), 2: np.array([0, 1, 0, 1]), 3 : np.array([0, 0, 1, 1]), 4: np.array([1, 1, 0, 1])}
        connect = []
        points = []
        colors = []
        if len(boxes) == 0:
            return
       
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

    def read_bbox(self, label):
        boxes = []
        cls_list = []
        cam_x = -0.019685
        cam_y = 0
        cam_z = 0.742092

        velo_x = -0.019685
        velo_y = 0
        velo_z = 1.077382

        dx = velo_x - cam_x
        dy = velo_y - cam_y
        dz = velo_z - cam_z
        
        orig2cam = R.from_euler('xyz', [0, 0, 0 ])
        orig2velo = R.from_euler('xyz', [0, 0, -0.085])

        cam2velo = orig2velo * orig2cam.inv() 

        rot_arr = cam2velo.as_matrix()
        

        for i in range(len(label)):
            box = label[i]["box"]
            x = box["cx"]
            y = box["cy"]
            z = box["cz"]

            vec = np.array([[x], [y], [z]])
            rot_vec = np.matmul(rot_arr, vec)
     
            x = rot_vec[0][0] - dx
            y = rot_vec[1][0] - dy
            z = rot_vec[2][0] - dz
            #z -= dz

            h = box["h"]
            w = box["w"]
            l = box["l"]
            yaw = box["rot_z"]
            if "social_activity" in label[i] and  "cycling" in label[i]["social_activity"]:
                cls_list.append(3)
            else:
                cls_list.append(2)
            corners = self.get_corners([2, h, w, l, x, y, z, -yaw])
            boxes.append(corners)

        return cls_list, boxes


    def update_scan(self):
        lidar_file = self.lidar_files[self.index]
        label = self.labels[lidar_file]
        
        points = self.read_points(os.path.join(self.data_folder, lidar_file))
        cls_list, boxes = self.read_bbox(label)
        
        #colors = self.get_point_color_using_intensity(points)
        colors = [0, 1, 1]
        self.canvas.title = lidar_file
        self.scan_vis.set_data(points[:, :3],
                            face_color=colors,
                            edge_color=colors,
                            size=1.0)

        self.plot_boxes(cls_list, boxes)


    def _key_press(self, event):
        if event.key == 'N':
            if self.index < len(self.lidar_files) - 1:
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
    name = "tressider-2019-04-26_2"
    data_folder = "/home/stpc/data/jrdb_train/train_dataset_with_activity/pointclouds/upper_velodyne/" + name
    label_file = "/home/stpc/data/jrdb_train/train_dataset_with_activity/labels/labels_3d/" + name + ".json"
    
    f = open(label_file)
    labels = json.load(f)

    vis = Vis(data_folder, labels["labels"])
    vis.run()
