import os
import numpy as np
import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas


class Vis():
    def __init__(self, data_folder, label_folder):
        self.index = 0
        self.lidar_paths, self.label_paths= self.read_data(data_folder, label_folder)

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

    def read_data(self, data_folder, label_folder):
        lidar_files = sorted(os.listdir(data_folder))
        lidar_paths = [os.path.join(data_folder, f) for f in lidar_files]
        label_paths = []
        for lidar_file in lidar_files:
            label_path = os.path.join(label_folder, lidar_file.replace("bin", "txt"))
            if os.path.isfile(label_path):
                label_paths.append(label_path)
            else:
                label_paths.append(None)
        return lidar_paths, label_paths

    def read_points(self, lidar_path):
        return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

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

    def read_bbox(self, label_path):
        object_list = {'Car': 1, 'Pedestrian':2, 'Person_sitting':2, 'Cyclist':3}
        #object_list = {'car': 1, 'pedestrian':2, 'person_sitting':2, 'bicycle':3}
        #object_list = {'vehicle.car': 1}

        corner_list = []
        class_list = []
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
                    corners = self.get_corners(bbox)
                    corner_list.append(corners)
        return (class_list, corner_list)

    def get_corners(self, bbox):
        h, w, l, x, y, z, yaw = bbox[1:]
        #yaw = -yaw
        #yaw = -(yaw + np.pi / 2)

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



    def plot_boxes(self, class_list, boxes):
        object_colors = {1: np.array([1, 0, 0, 1]), 2: np.array([0, 1, 0, 1]), 3:np.array([0, 0, 1, 1])}
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


    def update_scan(self):
        lidar_path = self.lidar_paths[self.index]
        label_path = self.label_paths[self.index]
        points = self.read_points(lidar_path)
        class_list, boxes = self.read_bbox(label_path)
        

        colors = self.get_point_color_using_intensity(points)
        
        self.canvas.title = f"Frame: {self.index} / {len(self.lidar_paths)} - {lidar_path}"
        self.scan_vis.set_data(points[:, :3],
                            face_color=colors,
                            edge_color=colors,
                            size=1.0)

        self.plot_boxes(class_list, boxes)


    def _key_press(self, event):
        if event.key == 'N':
            if self.index < len(self.lidar_paths) - 1:
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
    #data_folder = "/home/stpc/data/nuscenes/kitti/velodyne/"
    #label_folder = "/home/stpc/data/nuscenes/kitti/label_2/"
    data_folder = "/home/stpc/clean_data/kitti/pointcloud/"
    label_folder = "/home/stpc/clean_data/kitti/label/"
    
    vis = Vis(data_folder, label_folder)
    vis.run()
