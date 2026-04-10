#!/usr/bin/env python3

"""
Standalone keyboard teleop for AIC Cartesian control.

Keys:
- `a` / `d`: - / + X
- `w` / `s`: - / + Y
- `j` / `k`: - / + Z
- `q` / `e`: + / - Rx
- `u` / `i`: + / - Ry
- `o` / `p`: + / - Rz
- `n` / `m`: tcp / base frame
- `esc`: exit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from aic_control_interfaces.msg import MotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Twist, Vector3, Wrench
from pynput import keyboard
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import Buffer, TransformException, TransformListener

HIL_SERL_ROOT = Path(__file__).resolve().parents[3] / "hil-serl"
EXAMPLES_ROOT = HIL_SERL_ROOT / "examples"
SERL_LAUNCHER_ROOT = HIL_SERL_ROOT / "serl_launcher"

for path in (EXAMPLES_ROOT, SERL_LAUNCHER_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.aic_cable_insertion.config import EnvConfig
from experiments.aic_cable_insertion.wrapper import _euler_xyz_degrees_to_quat_xyzw

FAST_LINEAR_VEL = 0.01
FAST_ANGULAR_VEL = 0.06

KEY_MAPPINGS = {
    "d": (1, 0, 0, 0, 0, 0),
    "a": (-1, 0, 0, 0, 0, 0),
    "w": (0, -1, 0, 0, 0, 0),
    "s": (0, 1, 0, 0, 0, 0),
    "j": (0, 0, -1, 0, 0, 0),
    "k": (0, 0, 1, 0, 0, 0),
    "q": (0, 0, 0, 1, 0, 0),
    "e": (0, 0, 0, -1, 0, 0),
    "u": (0, 0, 0, 0, -1, 0),
    "i": (0, 0, 0, 0, 1, 0),
    "o": (0, 0, 0, 0, 0, 1),
    "p": (0, 0, 0, 0, 0, -1),
}


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


def _compute_signed_relative_euler_deg(
    source_quaternion_xyzw: np.ndarray,
    target_quaternion_xyzw: np.ndarray,
    expected_relative_quaternion_xyzw: np.ndarray | None = None,
) -> np.ndarray:
    relative_quaternion = _quat_multiply_xyzw(
        _quat_inverse_xyzw(target_quaternion_xyzw),
        source_quaternion_xyzw,
    )
    if expected_relative_quaternion_xyzw is not None:
        relative_quaternion = _quat_multiply_xyzw(
            _quat_inverse_xyzw(expected_relative_quaternion_xyzw),
            relative_quaternion,
        )
    return _quat_xyzw_to_euler_xyz_degrees(relative_quaternion).astype(np.float32)


class AICKeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__("aic_keyboard_teleop")
        config = EnvConfig()
        self.set_parameters([
            Parameter("use_sim_time", Parameter.Type.BOOL, bool(config.use_sim_time)),
        ])
        self.controller_namespace = self.declare_parameter("controller_namespace", "aic_controller").value
        self.publish_rate = float(self.declare_parameter("publish_rate", 25.0).value)
        self.fast_linear_velocity = float(
            self.declare_parameter("intervention_linear_velocity", FAST_LINEAR_VEL).value
        )
        self.fast_angular_velocity = float(
            self.declare_parameter("intervention_angular_velocity", FAST_ANGULAR_VEL).value
        )
        self.default_frame_id = self.declare_parameter("default_frame_id", "base_link").value
        self.angle_source_frame = self.declare_parameter(
            "angle_source_frame",
            config.reward_source_frame,
        ).value
        self.angle_target_frame = self.declare_parameter(
            "angle_target_frame",
            config.reward_target_frame,
        ).value
        self.angle_degrees_per_step = float(
            self.declare_parameter(
                "angle_degrees_per_step",
                config.angle_penalty_degrees_per_step,
            ).value
        )
        self.angle_penalty_per_bucket = float(
            self.declare_parameter(
                "angle_penalty_per_bucket",
                config.angle_penalty_per_bucket_per_step,
            ).value
        )
        self.angle_print_period_sec = float(
            self.declare_parameter("angle_print_period_sec", 0.5).value
        )

        self.motion_pub = self.create_publisher(
            MotionUpdate,
            f"/{self.controller_namespace}/pose_commands",
            10,
        )
        self.change_target_mode_client = self.create_client(
            ChangeTargetMode,
            f"/{self.controller_namespace}/change_target_mode",
        )

        self.active_keys = set()
        self.linear_vel = self.fast_linear_velocity
        self.angular_vel = self.fast_angular_velocity
        self.frame_id = self.default_frame_id
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
        self._expected_relative_quaternion = _euler_xyz_degrees_to_quat_xyzw(
            config.angle_expected_relative_euler_deg
        )
        self.motion_subscriber_ready = False
        self.last_wait_log_time = 0.0
        self.last_angle_log_time = 0.0
        self.last_command_log_time = 0.0

        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release,
        )
        self.keyboard_listener.start()

        self._ensure_cartesian_mode()
        self.timer = self.create_timer(1.0 / self.publish_rate, self.send_references)
        self.get_logger().info("Keyboard teleop started.")

    def on_key_press(self, key):
        try:
            if hasattr(key, "char") and key.char is not None:
                self.active_keys.add(key.char.lower())
        except AttributeError:
            return

    def on_key_release(self, key):
        try:
            if hasattr(key, "char") and key.char is not None:
                self.active_keys.discard(key.char.lower())
        except AttributeError:
            pass

        if key == keyboard.Key.esc:
            rclpy.shutdown()

    def _ensure_motion_subscriber_ready(self) -> bool:
        if self.motion_subscriber_ready:
            return True
        if self.motion_pub.get_subscription_count() > 0:
            self.motion_subscriber_ready = True
            self.get_logger().info("pose_commands subscriber detected.")
            return True
        now = time.monotonic()
        if now - self.last_wait_log_time > 1.0:
            self.get_logger().info("Waiting for subscriber to pose_commands...")
            self.last_wait_log_time = now
        return False

    def _ensure_cartesian_mode(self):
        if not self.change_target_mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warning("change_target_mode service unavailable.")
            return
        req = ChangeTargetMode.Request()
        req.target_mode.mode = TargetMode.MODE_CARTESIAN
        future = self.change_target_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result()
        if response is None or not response.success:
            self.get_logger().warning("Unable to switch controller to cartesian mode.")

    def generate_velocity_motion_update(self, twist: Twist, frame_id: str) -> MotionUpdate:
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.velocity = twist
        msg.target_stiffness = np.diag([85.0] * 6).flatten().tolist()
        msg.target_damping = np.diag([75.0] * 6).flatten().tolist()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0] * 6
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg

    def send_references(self):
        if not self._ensure_motion_subscriber_ready():
            return

        input_twist = np.zeros(6, dtype=np.float64)
        toggle_frame_id = False

        for key in self.active_keys:
            if key in KEY_MAPPINGS:
                vals = KEY_MAPPINGS[key]
                input_twist[0:3] += np.array(vals[0:3], dtype=np.float64) * self.linear_vel
                input_twist[3:6] += np.array(vals[3:6], dtype=np.float64) * self.angular_vel
            elif key == "n":
                self.frame_id = "gripper/tcp"
                toggle_frame_id = True
            elif key == "m":
                self.frame_id = "base_link"
                toggle_frame_id = True

        twist = Twist()
        twist.linear.x = float(input_twist[0])
        twist.linear.y = float(input_twist[1])
        twist.linear.z = float(input_twist[2])
        twist.angular.x = float(input_twist[3])
        twist.angular.y = float(input_twist[4])
        twist.angular.z = float(input_twist[5])

        self.motion_pub.publish(self.generate_velocity_motion_update(twist, self.frame_id))

        self._maybe_log_current_command(twist)
        self._maybe_log_current_angle()
        if toggle_frame_id:
            self.get_logger().info(f"Target frame set to {self.frame_id}")

    def _maybe_log_current_command(self, twist: Twist):
        now = time.monotonic()
        if now - self.last_command_log_time < self.angle_print_period_sec:
            return
        self.last_command_log_time = now

        self.get_logger().info(
            "cmd="
            f"[{twist.linear.x:.4f}, {twist.linear.y:.4f}, {twist.linear.z:.4f}; "
            f"{twist.angular.x:.4f}, {twist.angular.y:.4f}, {twist.angular.z:.4f}] "
            f"frame={self.frame_id} active_keys={sorted(self.active_keys)}"
        )

    def _maybe_log_current_angle(self):
        now = time.monotonic()
        if now - self.last_angle_log_time < self.angle_print_period_sec:
            return
        self.last_angle_log_time = now

        try:
            source_tf = self._tf_buffer.lookup_transform(
                "base_link",
                self.angle_source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
            target_tf = self._tf_buffer.lookup_transform(
                "base_link",
                self.angle_target_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            self.get_logger().warning(f"Angle TF lookup failed: {exc}")
            return

        source_quaternion = np.array(
            [
                source_tf.transform.rotation.x,
                source_tf.transform.rotation.y,
                source_tf.transform.rotation.z,
                source_tf.transform.rotation.w,
            ],
            dtype=np.float32,
        )
        target_quaternion = np.array(
            [
                target_tf.transform.rotation.x,
                target_tf.transform.rotation.y,
                target_tf.transform.rotation.z,
                target_tf.transform.rotation.w,
            ],
            dtype=np.float32,
        )
        euler_deg = _compute_signed_relative_euler_deg(
            source_quaternion_xyzw=source_quaternion,
            target_quaternion_xyzw=target_quaternion,
            expected_relative_quaternion_xyzw=self._expected_relative_quaternion,
        )
        self.get_logger().info(
            f"angle_deg=[{euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}]"
        )

    def cleanup(self):
        zero_twist = Twist()
        self.motion_pub.publish(self.generate_velocity_motion_update(zero_twist, self.frame_id))
        self.keyboard_listener.stop()


def main(args=None):
    print(
        """
Keyboard teleop for Cartesian control
-----------------------------------
Linear:
  a/d : -/+ X
  w/s : -/+ Y
  j/k : -/+ Z

Angular:
  q/e : + / - Rx
  u/i : + / - Ry
  o/p : + / - Rz

Modes:
  n / m : tcp / base frame
  esc   : quit
        """.strip()
    )

    node = None
    try:
        rclpy.init(args=args)
        node = AICKeyboardTeleopNode()
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
