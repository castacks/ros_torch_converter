import yaml
import torch
import numpy as np
#switch backend to one that works out of main thread
import matplotlib
matplotlib.use('GTK3agg')
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from ros_torch_converter.converter import ROSTorchConverter

class MPLPlotter(Node):
    def __init__(self, converter):
        super().__init__('mpl_plotter_node')
        self.converter = converter
        self.status_timer = self.create_timer(1.0, self.converter_status_callback)
        self.plot_timer = self.create_timer(0.5, self.plot_callback)

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 12))
        plt.show(block=False)

        self.pos_buf = np.zeros([0, 2])
        self.speed_buf = np.zeros([0, ])

        self.get_logger().info('plotter ready')

    def converter_status_callback(self):
        self.get_logger().info(self.converter.get_status_str())

    def plot_callback(self):
        if self.converter.can_get_data():
            data = self.converter.get_data()

            pos = data['state'][:2].numpy().reshape(1, 2)
            speed = np.linalg.norm(data['state'][7:10].numpy())
            self.pos_buf = np.append(self.pos_buf, pos, axis=0)
            self.speed_buf = np.append(self.speed_buf, speed)

            costmap_metadata = data['local_costmap']['metadata']
            extent = (
                costmap_metadata['origin'][0],
                costmap_metadata['origin'][0] + costmap_metadata['length'][0],
                costmap_metadata['origin'][1],
                costmap_metadata['origin'][1] + costmap_metadata['length'][1]
            )
            costmap = data['local_costmap']['data'][0].numpy()

            goal_arr = data['waypoints']

            self.ax.cla()

            self.ax.imshow(costmap.T, origin='lower', extent=extent, cmap='plasma')
            self.ax.plot(self.pos_buf[:, 0], self.pos_buf[:, 1], c='r')
            self.ax.scatter(goal_arr[:, 0], goal_arr[:, 1], c='r', marker='x')

            self.ax.set_xlim(extent[0], extent[1])
            self.ax.set_ylim(extent[2], extent[3])

            self.ax.set_xlabel('X(m)')
            self.ax.set_ylabel('Y(m)')

            plt.pause(1e-2)

def main(args=None):
    config_fp = '/home/tartandriver/tartandriver_ws/src/core/ros_torch_converter/ros_torch_converter/configs/costmap_speedmap.yaml'
    config = yaml.safe_load(open(config_fp, 'r'))['ros']

    rclpy.init(args=args)

    converter_node = ROSTorchConverter(config)
    plt_node = MPLPlotter(converter_node)

    executor = MultiThreadedExecutor()
    executor.add_node(converter_node)
    executor.add_node(plt_node)

    try:
        executor.spin()
    finally:
        converter_node.destroy_node()
        plt_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()