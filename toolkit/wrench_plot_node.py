#!/usr/bin/env python3

from collections import deque

import matplotlib.pyplot as plt
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node


class WrenchPlotNode(Node):
    def __init__(self):
        super().__init__("wrench_plot_node")

        self.topic_name = "/nic_insertion/processed_wrench"
        self.filtered_topic_name = "/nic_insertion/processed_wrench/filtered"
        self.window_sec = 20.0

        self.times = deque()
        self.times_filtered = deque()
        self.fx_values = deque()
        self.fy_values = deque()
        self.fz_values = deque()
        self.tx_values = deque()
        self.ty_values = deque()
        self.tz_values = deque()

        self.fx_values_filtered = deque()
        self.fy_values_filtered = deque()
        self.fz_values_filtered = deque()
        self.tx_values_filtered = deque()
        self.ty_values_filtered = deque()
        self.tz_values_filtered = deque()

        self.start_time = None

        self.subscription = self.create_subscription(
            WrenchStamped,
            self.topic_name,
            self.wrench_callback,
            10,
        )
        self.filtered_subscription = self.create_subscription(
            WrenchStamped,
            self.filtered_topic_name,
            self.filtered_wrench_callback,
            10,
        )

        self.fig, (
            self.ax_force,
            self.ax_force_filtered,
            self.ax_torque,
            self.ax_torque_filtered,
        ) = plt.subplots(
            4, 1, figsize=(10, 12), sharex=True
        )
        self.line_fx, = self.ax_force.plot([], [], label="Fx")
        self.line_fy, = self.ax_force.plot([], [], label="Fy")
        self.line_fz, = self.ax_force.plot([], [], label="Fz")

        self.line_fx_filtered, = self.ax_force_filtered.plot([], [], label="Fx (filtered)")
        self.line_fy_filtered, = self.ax_force_filtered.plot([], [], label="Fy (filtered)")
        self.line_fz_filtered, = self.ax_force_filtered.plot([], [], label="Fz (filtered)")

        self.line_tx, = self.ax_torque.plot([], [], label="Tx")
        self.line_ty, = self.ax_torque.plot([], [], label="Ty")
        self.line_tz, = self.ax_torque.plot([], [], label="Tz")

        self.line_tx_filtered, = self.ax_torque_filtered.plot([], [], label="Tx (filtered)")
        self.line_ty_filtered, = self.ax_torque_filtered.plot([], [], label="Ty (filtered)")
        self.line_tz_filtered, = self.ax_torque_filtered.plot([], [], label="Tz (filtered)")

        self.ax_force.set_title("Processed Wrench - Force")
        self.ax_force.set_ylabel("Force (N)")
        self.ax_force.legend(loc="upper right")
        self.ax_force.grid(True)

        self.ax_force_filtered.set_title("Processed Wrench - Force (1s Filtered)")
        self.ax_force_filtered.set_ylabel("Force (N)")
        self.ax_force_filtered.legend(loc="upper right")
        self.ax_force_filtered.grid(True)

        self.ax_torque.set_title("Processed Wrench - Torque")
        self.ax_torque.set_ylabel("Torque (Nm)")
        self.ax_torque.legend(loc="upper right")
        self.ax_torque.grid(True)

        self.ax_torque_filtered.set_title("Processed Wrench - Torque (1s Filtered)")
        self.ax_torque_filtered.set_xlabel("Time (s)")
        self.ax_torque_filtered.set_ylabel("Torque (Nm)")
        self.ax_torque_filtered.legend(loc="upper right")
        self.ax_torque_filtered.grid(True)

        self.get_logger().info(f"Subscribed to {self.topic_name}")
        self.get_logger().info(f"Subscribed to {self.filtered_topic_name}")

    def wrench_callback(self, msg: WrenchStamped):
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if now == 0.0:
            now = self.get_clock().now().nanoseconds / 1e9

        if self.start_time is None:
            self.start_time = now

        t = now - self.start_time
        self.times.append(t)
        self.fx_values.append(msg.wrench.force.x)
        self.fy_values.append(msg.wrench.force.y)
        self.fz_values.append(msg.wrench.force.z)
        self.tx_values.append(msg.wrench.torque.x)
        self.ty_values.append(msg.wrench.torque.y)
        self.tz_values.append(msg.wrench.torque.z)

        while self.times and (self.times[-1] - self.times[0] > self.window_sec):
            self.times.popleft()
            self.fx_values.popleft()
            self.fy_values.popleft()
            self.fz_values.popleft()
            self.tx_values.popleft()
            self.ty_values.popleft()
            self.tz_values.popleft()

    def filtered_wrench_callback(self, msg: WrenchStamped):
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if now == 0.0:
            now = self.get_clock().now().nanoseconds / 1e9

        if self.start_time is None:
            self.start_time = now

        t = now - self.start_time
        self.times_filtered.append(t)
        self.fx_values_filtered.append(msg.wrench.force.x)
        self.fy_values_filtered.append(msg.wrench.force.y)
        self.fz_values_filtered.append(msg.wrench.force.z)
        self.tx_values_filtered.append(msg.wrench.torque.x)
        self.ty_values_filtered.append(msg.wrench.torque.y)
        self.tz_values_filtered.append(msg.wrench.torque.z)

        while self.times_filtered and (self.times_filtered[-1] - self.times_filtered[0] > self.window_sec):
            self.times_filtered.popleft()
            self.fx_values_filtered.popleft()
            self.fy_values_filtered.popleft()
            self.fz_values_filtered.popleft()
            self.tx_values_filtered.popleft()
            self.ty_values_filtered.popleft()
            self.tz_values_filtered.popleft()

    def update_plot(self):
        if not self.times and not self.times_filtered:
            return

        x_data = list(self.times)
        x_data_filtered = list(self.times_filtered)

        self.line_fx.set_data(x_data, list(self.fx_values))
        self.line_fy.set_data(x_data, list(self.fy_values))
        self.line_fz.set_data(x_data, list(self.fz_values))
        self.line_tx.set_data(x_data, list(self.tx_values))
        self.line_ty.set_data(x_data, list(self.ty_values))
        self.line_tz.set_data(x_data, list(self.tz_values))

        self.line_fx_filtered.set_data(x_data_filtered, list(self.fx_values_filtered))
        self.line_fy_filtered.set_data(x_data_filtered, list(self.fy_values_filtered))
        self.line_fz_filtered.set_data(x_data_filtered, list(self.fz_values_filtered))

        self.line_tx_filtered.set_data(x_data_filtered, list(self.tx_values_filtered))
        self.line_ty_filtered.set_data(x_data_filtered, list(self.ty_values_filtered))
        self.line_tz_filtered.set_data(x_data_filtered, list(self.tz_values_filtered))

        self.ax_force.relim()
        self.ax_force.autoscale_view()
        self.ax_force_filtered.relim()
        self.ax_force_filtered.autoscale_view()
        self.ax_torque.relim()
        self.ax_torque.autoscale_view()
        self.ax_torque_filtered.relim()
        self.ax_torque_filtered.autoscale_view()


def main(args=None):
    rclpy.init(args=args)
    node = WrenchPlotNode()

    try:
        plt.ion()
        while rclpy.ok() and plt.fignum_exists(node.fig.number):
            rclpy.spin_once(node, timeout_sec=0.05)
            node.update_plot()
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close("all")


if __name__ == "__main__":
    main()
