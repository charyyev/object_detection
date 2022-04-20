import numpy as np
import rosbag
from tools.lidar_conversion import pcl_to_numpy

import os
import numpy as np
import matplotlib.pyplot as plt
import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas
from vispy import app
from vispy.scene.visuals import Text


class Vis():
    def __init__(self, bag_name, save_location, frames):
        self.index = 0
        self.frames = frames
        self.bag_name = bag_name
        self.save_location = save_location
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

        self.text = Text(parent=self.canvas.scene, color='red', font_size = 10, pos = (170, 30))
        
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
        colors = self.get_point_color_using_intensity(points)
        colors = np.array([0, 1, 1])
        if not self.paused:
            self.index += 1
            self.text.text = ""
        self.canvas.title = f"Frame: {self.index} / {len(self.frames)}"
        self.scan_vis.set_data(points[:, :3],
                            face_color=colors,
                            edge_color=colors,
                            size=1.0)

    def save_frame(self):
        points = self.frames[self.index]
        filename = self.bag_name + "_" + '%06d' % self.index + ".bin"
        points.astype("float32").tofile(os.path.join(self.save_location, filename))

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
    bag_file = "/home/stpc/stpc_ws/data/rosbag_recorder/scripts/city_ori_xt.bag"
    bag_name = bag_file.split("/")[-1].split(".")[0]
    save_location = "/home/stpc/clean_data/custom/pointcloud"
    
    bag = rosbag.Bag(bag_file)
    lidar_topics = ["/points_raw"]
    frames = []
    for topic, msg, t in bag.read_messages():
        if topic in lidar_topics:
            frames.append(pcl_to_numpy(msg))

    print("Done reading frames: ", len(frames), "frames in total")

    if not os.path.exists(save_location):
        os.makedirs(save_location)
    vis = Vis(bag_name, save_location, frames)
    vis.run()


            