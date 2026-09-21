#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import Buffer, TransformException, TransformListener


def _normalize_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _quat_xyzw_to_euler_xyz_degrees(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion_xyzw(quat)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


def _euler_xyz_degrees_to_quat_xyzw(euler_deg: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.radians(np.asarray(euler_deg, dtype=np.float64))
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def _quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = _normalize_quaternion_xyzw(q1)
    x2, y2, z2, w2 = _normalize_quaternion_xyzw(q2)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_inverse_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion_xyzw(quat)
    return np.array([-x, -y, -z, w], dtype=np.float64)


class AngleWrenchRecorderNode(Node):
    def __init__(self):
        super().__init__("angle_wrench_recorder")
        self._declare_parameters()
        self._load_parameters()

        self.latest_wrench_msg = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.expected_relative_quat = _euler_xyz_degrees_to_quat_xyzw(
            self.expected_relative_euler_deg
        )

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.output_csv.open("w", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "stamp_sec",
                "angle_x_deg",
                "angle_y_deg",
                "angle_z_deg",
                "abs_angle_x_deg",
                "abs_angle_y_deg",
                "abs_angle_z_deg",
                "fx",
                "fy",
                "fz",
                "tx",
                "ty",
                "tz",
            ],
        )
        self.csv_writer.writeheader()

        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            self.wrench_topic,
            self._on_wrench,
            20,
        )
        self.create_timer(1.0 / self.record_rate_hz, self._on_timer)
        self.get_logger().info(f"Recording angle/wrench samples to {self.output_csv}")
        self.get_logger().info(
            f"TF angle: {self.base_frame} -> {self.source_frame} relative to {self.target_frame}"
        )
        self.get_logger().info(f"Wrench topic: {self.wrench_topic}")

    def _declare_parameters(self):
        self.declare_parameter(
            "output_csv",
            str(Path(__file__).with_name("angle_wrench_samples.csv")),
        )
        self.declare_parameter("wrench_topic", "/nic_insertion/processed_wrench/filtered")
        self.declare_parameter("record_rate_hz", 10.0)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("source_frame", "cable_0/sfp_tip_link")
        self.declare_parameter(
            "target_frame",
            "task_board/nic_card_mount_0/sfp_port_1_link",
        )
        self.declare_parameter("expected_relative_euler_deg", [0.0, 0.0, 0.0])

    def _load_parameters(self):
        self.output_csv = Path(str(self.get_parameter("output_csv").value))
        self.wrench_topic = str(self.get_parameter("wrench_topic").value)
        self.record_rate_hz = max(1e-3, float(self.get_parameter("record_rate_hz").value))
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.source_frame = str(self.get_parameter("source_frame").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.expected_relative_euler_deg = np.asarray(
            [float(v) for v in self.get_parameter("expected_relative_euler_deg").value],
            dtype=np.float64,
        )
        if self.expected_relative_euler_deg.size != 3:
            raise ValueError("expected_relative_euler_deg must have 3 values.")

    def _on_wrench(self, msg: WrenchStamped):
        self.latest_wrench_msg = msg

    def _lookup_quat_xyzw(self, frame: str) -> np.ndarray:
        transform = self.tf_buffer.lookup_transform(
            self.base_frame,
            frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.2),
        )
        rotation = transform.transform.rotation
        return np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64)

    def _signed_angle_error_deg(self) -> np.ndarray:
        source_quat = self._lookup_quat_xyzw(self.source_frame)
        target_quat = self._lookup_quat_xyzw(self.target_frame)
        relative_quat = _quat_multiply_xyzw(
            _quat_inverse_xyzw(target_quat),
            source_quat,
        )
        relative_quat = _quat_multiply_xyzw(
            _quat_inverse_xyzw(self.expected_relative_quat),
            relative_quat,
        )
        return _quat_xyzw_to_euler_xyz_degrees(relative_quat)

    def _on_timer(self):
        if self.latest_wrench_msg is None:
            return
        try:
            angle_deg = self._signed_angle_error_deg()
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup failed: {exc}")
            return

        wrench = self.latest_wrench_msg.wrench
        now_msg = self.get_clock().now().to_msg()
        self.csv_writer.writerow(
            {
                "stamp_sec": now_msg.sec + now_msg.nanosec * 1e-9,
                "angle_x_deg": float(angle_deg[0]),
                "angle_y_deg": float(angle_deg[1]),
                "angle_z_deg": float(angle_deg[2]),
                "abs_angle_x_deg": float(abs(angle_deg[0])),
                "abs_angle_y_deg": float(abs(angle_deg[1])),
                "abs_angle_z_deg": float(abs(angle_deg[2])),
                "fx": float(wrench.force.x),
                "fy": float(wrench.force.y),
                "fz": float(wrench.force.z),
                "tx": float(wrench.torque.x),
                "ty": float(wrench.torque.y),
                "tz": float(wrench.torque.z),
            }
        )
        self.csv_file.flush()

    def cleanup(self):
        self.csv_file.close()


def main(args=None):
    cli_args = list(sys.argv[1:] if args is None else args)
    rclpy.init(args=args)
    node = None
    try:
        node = AngleWrenchRecorderNode()
        if not any("use_sim_time:=" in arg for arg in cli_args):
            node.set_parameters([
                Parameter("use_sim_time", Parameter.Type.BOOL, True)
            ])
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.cleanup()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
