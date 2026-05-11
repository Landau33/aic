#!/usr/bin/env python3
# 用法:
#   python3 wrench_tf_processor_node.py
#   python3 wrench_tf_processor_node.py --ros-args -p target_frame:=gripper/tcp -p source_frame:=ati/tool_link -p use_sim_time:=true
# 功能:
#   1) 订阅原始力/力矩与 tare，进行去皮 + TF 转换，发布:
#      - /nic_insertion/processed_wrench
#      - /nic_insertion/processed_wrench/filtered  (1s 滑动平均)
#   2) 同时实时绘制 processed / filtered 的 force 和 torque 曲线

from collections import deque
import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter

from tf2_ros import Buffer, TransformListener

from geometry_msgs.msg import WrenchStamped
from aic_control_interfaces.msg import ControllerState


def quaternion_to_rotation_matrix(q) -> np.ndarray:
    """q = [x, y, z, w] → 3x3 旋转矩阵（纯 numpy，无额外包）"""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


class WrenchTFProcessorNode(Node):
    def __init__(self):
        super().__init__("wrench_tf_processor_node")

        self.declare_parameter("target_frame", "gripper/tcp")
        self.declare_parameter("source_frame", "ati/tool_link")
        self.declare_parameter("raw_wrench_topic", "/fts_broadcaster/wrench")
        self.declare_parameter(
            "controller_state_topic", "/aic_controller/controller_state"
        )
        self.declare_parameter("output_topic", "/nic_insertion/processed_wrench")
        self.declare_parameter(
            "output_filtered_topic", "/nic_insertion/processed_wrench/filtered"
        )
        self.declare_parameter("filter_window_sec", 1.0)
        self.declare_parameter("enable_plot", True)
        self.declare_parameter("plot_window_sec", 20.0)

        self.target_frame = str(self.get_parameter("target_frame").value)
        self.source_frame = str(self.get_parameter("source_frame").value)
        self.raw_wrench_topic = str(self.get_parameter("raw_wrench_topic").value)
        self.controller_state_topic = str(
            self.get_parameter("controller_state_topic").value
        )
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.output_filtered_topic = str(
            self.get_parameter("output_filtered_topic").value
        )
        self.filter_window_sec = float(self.get_parameter("filter_window_sec").value)
        self.enable_plot = bool(self.get_parameter("enable_plot").value)
        self.plot_window_sec = float(self.get_parameter("plot_window_sec").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_wrench = None
        self.current_tared_wrench = None

        self.filter_time = deque()
        self.filter_wrench = deque()
        self.plot_start_time = None
        self.plot_times = deque()
        self.plot_values = [deque() for _ in range(6)]
        self.plot_times_filtered = deque()
        self.plot_values_filtered = [deque() for _ in range(6)]
        self.fig = None

        self.processed_pub = self.create_publisher(WrenchStamped, self.output_topic, 10)
        self.filtered_pub = self.create_publisher(
            WrenchStamped, self.output_filtered_topic, 10
        )

        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            self.raw_wrench_topic,
            self.wrench_callback,
            10,
        )
        self.state_sub = self.create_subscription(
            ControllerState,
            self.controller_state_topic,
            self.controller_state_callback,
            10,
        )

        self.create_timer(0.02, self.timer_callback)
        if self.enable_plot:
            self._init_plot()

        self.get_logger().info(f"target_frame: {self.target_frame}")
        self.get_logger().info(f"source_frame: {self.source_frame}")
        self.get_logger().info(f"publishing: {self.output_topic}")
        self.get_logger().info(f"publishing: {self.output_filtered_topic}")
        self.get_logger().info(f"plot enabled: {self.enable_plot}")
        self.get_logger().info(
            "wrench flow: raw -> tare -> tf(target_frame) -> filtered"
        )

    def _init_plot(self):
        self.fig, (
            self.ax_force,
            self.ax_force_filtered,
            self.ax_torque,
            self.ax_torque_filtered,
        ) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        self.lines_force = [
            self.ax_force.plot([], [], label=label)[0] for label in ("Fx", "Fy", "Fz")
        ]
        self.lines_force_filtered = [
            self.ax_force_filtered.plot([], [], label=f"{label} (filtered)")[0]
            for label in ("Fx", "Fy", "Fz")
        ]
        self.lines_torque = [
            self.ax_torque.plot([], [], label=label)[0] for label in ("Tx", "Ty", "Tz")
        ]
        self.lines_torque_filtered = [
            self.ax_torque_filtered.plot([], [], label=f"{label} (filtered)")[0]
            for label in ("Tx", "Ty", "Tz")
        ]

        self.ax_force.set_title("Processed Wrench - Force")
        self.ax_force.set_ylabel("Force (N)")
        self.ax_force.legend(loc="upper right")
        self.ax_force.grid(True)

        self.ax_force_filtered.set_title(
            f"Processed Wrench - Force ({self.filter_window_sec:.1f}s Filtered)"
        )
        self.ax_force_filtered.set_ylabel("Force (N)")
        self.ax_force_filtered.legend(loc="upper right")
        self.ax_force_filtered.grid(True)

        self.ax_torque.set_title("Processed Wrench - Torque")
        self.ax_torque.set_ylabel("Torque (Nm)")
        self.ax_torque.legend(loc="upper right")
        self.ax_torque.grid(True)

        self.ax_torque_filtered.set_title(
            f"Processed Wrench - Torque ({self.filter_window_sec:.1f}s Filtered)"
        )
        self.ax_torque_filtered.set_xlabel("Time (s)")
        self.ax_torque_filtered.set_ylabel("Torque (Nm)")
        self.ax_torque_filtered.legend(loc="upper right")
        self.ax_torque_filtered.grid(True)

    def wrench_callback(self, msg: WrenchStamped):
        self.current_wrench = msg

    def controller_state_callback(self, msg: ControllerState):
        if self.current_wrench is None:
            return

        self.current_tared_wrench = copy.deepcopy(self.current_wrench)
        tare = msg.fts_tare_offset.wrench
        self.current_tared_wrench.wrench.force.x -= tare.force.x
        self.current_tared_wrench.wrench.force.y -= tare.force.y
        self.current_tared_wrench.wrench.force.z -= tare.force.z
        self.current_tared_wrench.wrench.torque.x -= tare.torque.x
        self.current_tared_wrench.wrench.torque.y -= tare.torque.y
        self.current_tared_wrench.wrench.torque.z -= tare.torque.z

    def _update_filter(self, wrench_vec: np.ndarray, now_sec: float):
        self.filter_time.append(now_sec)
        self.filter_wrench.append(wrench_vec)

        while self.filter_time and (
            now_sec - self.filter_time[0] > self.filter_window_sec
        ):
            self.filter_time.popleft()
            self.filter_wrench.popleft()

        if len(self.filter_wrench) == 0:
            return wrench_vec

        return np.mean(np.vstack(self.filter_wrench), axis=0)

    def _publish_wrench(self, vec: np.ndarray, frame_id: str, topic_pub):
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.wrench.force.x = float(vec[0])
        msg.wrench.force.y = float(vec[1])
        msg.wrench.force.z = float(vec[2])
        msg.wrench.torque.x = float(vec[3])
        msg.wrench.torque.y = float(vec[4])
        msg.wrench.torque.z = float(vec[5])
        topic_pub.publish(msg)
        return msg

    def _record_plot_sample(self, vec: np.ndarray, now_sec: float, filtered: bool):
        if not self.enable_plot:
            return
        if self.plot_start_time is None:
            self.plot_start_time = now_sec

        times = self.plot_times_filtered if filtered else self.plot_times
        values = self.plot_values_filtered if filtered else self.plot_values
        plot_time = now_sec - self.plot_start_time
        times.append(plot_time)
        for idx, value in enumerate(vec):
            values[idx].append(float(value))

        while times and (times[-1] - times[0] > self.plot_window_sec):
            times.popleft()
            for values_for_axis in values:
                values_for_axis.popleft()

    def update_plot(self):
        if not self.enable_plot or self.fig is None:
            return
        if not self.plot_times and not self.plot_times_filtered:
            return

        x_data = list(self.plot_times)
        x_data_filtered = list(self.plot_times_filtered)

        for idx, line in enumerate(self.lines_force):
            line.set_data(x_data, list(self.plot_values[idx]))
        for idx, line in enumerate(self.lines_torque):
            line.set_data(x_data, list(self.plot_values[idx + 3]))
        for idx, line in enumerate(self.lines_force_filtered):
            line.set_data(x_data_filtered, list(self.plot_values_filtered[idx]))
        for idx, line in enumerate(self.lines_torque_filtered):
            line.set_data(x_data_filtered, list(self.plot_values_filtered[idx + 3]))

        for axis in (
            self.ax_force,
            self.ax_force_filtered,
            self.ax_torque,
            self.ax_torque_filtered,
        ):
            axis.relim()
            axis.autoscale_view()

    def timer_callback(self):
        if self.current_tared_wrench is None:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
            # 1. 提取 R 和 p
            q = transform.transform.rotation
            R = quaternion_to_rotation_matrix(q)  # 3x3
            p = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )

            # 3. 原始力/力矩（source frame）
            f_source = np.array(
                [
                    self.current_tared_wrench.wrench.force.x,
                    self.current_tared_wrench.wrench.force.y,
                    self.current_tared_wrench.wrench.force.z,
                ]
            )
            tau_source = np.array(
                [
                    self.current_tared_wrench.wrench.torque.x,
                    self.current_tared_wrench.wrench.torque.y,
                    self.current_tared_wrench.wrench.torque.z,
                ]
            )

            # 4. 完整 wrench 变换（关键！）
            f_target = R @ f_source
            tau_target = R @ tau_source + np.cross(p, f_target)  # ← 这就是缺失的 p×F

            raw_wrench = np.concatenate([f_target, tau_target])
        except Exception as exc:
            self.get_logger().warn(f"Wrench transform failed: {str(exc)}")
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        filtered_wrench = self._update_filter(raw_wrench, now_sec)

        self._publish_wrench(raw_wrench, self.target_frame, self.processed_pub)
        self._publish_wrench(filtered_wrench, self.target_frame, self.filtered_pub)
        self._record_plot_sample(raw_wrench, now_sec, filtered=False)
        self._record_plot_sample(filtered_wrench, now_sec, filtered=True)


def main(args=None):
    cli_args = list(sys.argv[1:] if args is None else args)
    rclpy.init(args=args)
    node = WrenchTFProcessorNode()
    if not any("use_sim_time:=" in arg for arg in cli_args):
        node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

    try:
        if node.enable_plot:
            plt.ion()
            while (
                rclpy.ok()
                and node.fig is not None
                and plt.fignum_exists(node.fig.number)
            ):
                rclpy.spin_once(node, timeout_sec=0.05)
                node.update_plot()
                plt.pause(0.05)
        else:
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if node.enable_plot:
            plt.close("all")


if __name__ == "__main__":
    main()
