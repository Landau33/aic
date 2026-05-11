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
- `n`: toggle tcp / base frame
- `m`: toggle normal / fast speed
- `z`: toggle auto-align angle to [0, 0, 0]
- `x`: toggle auto-align to a random target with each axis in [-2, 2] deg
- `c`: toggle auto-move to a random point above the target via its W-side waypoint
- `r`: publish a zero twist once, then exit
- `t`: toggle pause (stop publishing; release all held keys and auto modes)
- `esc`: exit
"""

from __future__ import annotations

import time
from enum import Enum

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
from std_msgs.msg import String as StdString
from tf2_ros import Buffer, TransformException, TransformListener

NORMAL_LINEAR_VEL = 0.01
NORMAL_ANGULAR_VEL = 0.06
FAST_LINEAR_VEL = 0.05
FAST_ANGULAR_VEL = 0.18
AUTO_ALIGN_ANGULAR_GAIN = 1.2
AUTO_ALIGN_MAX_ANGULAR_VEL = 0.1
AUTO_ALIGN_TOLERANCE_DEG = 0.5
RANDOM_ALIGN_MAX_DEG = 2.0
AUTO_MOVE_ABOVE_HEIGHT_MIN = 0.038
AUTO_MOVE_ABOVE_HEIGHT_MAX = 0.040
AUTO_MOVE_XY_MAX = 0.003
AUTO_MOVE_W_SIDE_OFFSET = 0.2
AUTO_MOVE_LINEAR_GAIN = 1.5
AUTO_MOVE_MAX_LINEAR_VEL = 0.05
AUTO_MOVE_TOLERANCE_M = 0.002
AUTO_MOVE_TCP_FRAME = "gripper/tcp"
DEFAULT_TASK_NAME = "task3"
TASK_ANGLE_FRAMES = {
    "task1": ("cable_0/sfp_tip_link", "task_board/nic_card_mount_0/sfp_port_0_link"),
    "trial_1": ("cable_0/sfp_tip_link", "task_board/nic_card_mount_0/sfp_port_0_link"),
    "task2": ("cable_0/sfp_tip_link", "task_board/nic_card_mount_1/sfp_port_0_link"),
    "trial_2": ("cable_0/sfp_tip_link", "task_board/nic_card_mount_1/sfp_port_0_link"),
    "task3": ("cable_1/sc_tip_link", "task_board/sc_port_0/sc_port_base_link"),
    "trial_3": ("cable_1/sc_tip_link", "task_board/sc_port_1/sc_port_base_link"),
}
DEFAULT_ANGLE_SOURCE_FRAME, DEFAULT_ANGLE_TARGET_FRAME = TASK_ANGLE_FRAMES[DEFAULT_TASK_NAME]
DEFAULT_ANGLE_EXPECTED_RELATIVE_EULER_DEG = (0.0, 0.0, 0.0)

KEY_MAPPINGS = {
    "d": (-1, 0, 0, 0, 0, 0),
    "a": (1, 0, 0, 0, 0, 0),
    "w": (0, -1, 0, 0, 0, 0),
    "s": (0, 1, 0, 0, 0, 0),
    "j": (0, 0, -1, 0, 0, 0),
    "k": (0, 0, 1, 0, 0, 0),
    "q": (0, 0, 0, 1, 0, 0),
    "e": (0, 0, 0, -1, 0, 0),
    "u": (0, 0, 0, 0, 1, 0),
    "i": (0, 0, 0, 0, -1, 0),
    "o": (0, 0, 0, 0, 0, -1),
    "p": (0, 0, 0, 0, 0, 1),
}

ANGLE_ZERO_CORRECTION_KEYS = {
    "x": ("e", "q"),
    "y": ("u", "i"),
    "z": ("p", "o"),
}


class SpeedMode(Enum):
    NORMAL = "normal"
    FAST = "fast"


def _resolve_task_angle_frames(task_name: str) -> tuple[str, str]:
    task_key = str(task_name).strip().lower().replace("_", "")
    normalized_frames = {key.replace("_", ""): frames for key, frames in TASK_ANGLE_FRAMES.items()}
    return normalized_frames.get(task_key, TASK_ANGLE_FRAMES[DEFAULT_TASK_NAME])


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


def _euler_xyz_degrees_to_quat_xyzw(
    euler_deg: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    roll, pitch, yaw = np.radians(np.asarray(euler_deg, dtype=np.float64))
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return _normalize_quaternion_xyzw(
        np.array(
            [
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ],
            dtype=np.float64,
        )
    )


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


def _quaternion_from_transform(transform) -> np.ndarray:
    return np.array(
        [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ],
        dtype=np.float32,
    )


def _quat_xyzw_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quaternion_xyzw(quat)
    if quat[3] < 0.0:
        quat = -quat
    vector = quat[:3]
    vector_norm = np.linalg.norm(vector)
    if vector_norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, quat[3])
    return vector / vector_norm * angle


def _rotate_vector_by_quat_xyzw(vector: np.ndarray, quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quaternion_xyzw(quat)
    q_vec = quat[:3]
    q_w = quat[3]
    vector = np.asarray(vector, dtype=np.float64)
    return (
        vector
        + 2.0 * q_w * np.cross(q_vec, vector)
        + 2.0 * np.cross(q_vec, np.cross(q_vec, vector))
    )


def _key_to_zero_angle(axis_name: str, angle_deg: float, tolerance_deg: float = 0.05) -> str:
    if abs(angle_deg) <= tolerance_deg:
        return "-"
    positive_key, negative_key = ANGLE_ZERO_CORRECTION_KEYS[axis_name]
    return positive_key if angle_deg > 0.0 else negative_key


class AICKeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__("aic_keyboard_teleop")
        if self.has_parameter("use_sim_time"):
            self.set_parameters([Parameter("use_sim_time", value=True)])
        else:
            self.declare_parameter("use_sim_time", True)
        self.controller_namespace = self.declare_parameter("controller_namespace", "aic_controller").value
        self.publish_rate = float(self.declare_parameter("publish_rate", 25.0).value)
        self.normal_linear_velocity = float(
            self.declare_parameter("intervention_linear_velocity", NORMAL_LINEAR_VEL).value
        )
        self.normal_angular_velocity = float(
            self.declare_parameter("intervention_angular_velocity", NORMAL_ANGULAR_VEL).value
        )
        self.fast_linear_velocity = float(
            self.declare_parameter("fast_linear_velocity", FAST_LINEAR_VEL).value
        )
        self.fast_angular_velocity = float(
            self.declare_parameter("fast_angular_velocity", FAST_ANGULAR_VEL).value
        )
        self.default_frame_id = self.declare_parameter("default_frame_id", "base_link").value
        self.task_name = self.declare_parameter("task_name", DEFAULT_TASK_NAME).value
        default_angle_source_frame, default_angle_target_frame = _resolve_task_angle_frames(
            self.task_name
        )
        self.angle_source_frame = self.declare_parameter(
            "angle_source_frame",
            default_angle_source_frame,
        ).value
        self.angle_target_frame = self.declare_parameter(
            "angle_target_frame",
            default_angle_target_frame,
        ).value
        self.angle_print_period_sec = float(
            self.declare_parameter("angle_print_period_sec", 0.5).value
        )
        self.auto_align_angular_gain = float(
            self.declare_parameter("auto_align_angular_gain", AUTO_ALIGN_ANGULAR_GAIN).value
        )
        self.auto_align_max_angular_velocity = float(
            self.declare_parameter(
                "auto_align_max_angular_velocity",
                AUTO_ALIGN_MAX_ANGULAR_VEL,
            ).value
        )
        self.auto_align_tolerance_deg = float(
            self.declare_parameter("auto_align_tolerance_deg", AUTO_ALIGN_TOLERANCE_DEG).value
        )
        self.random_align_max_deg = float(
            self.declare_parameter("random_align_max_deg", RANDOM_ALIGN_MAX_DEG).value
        )
        self.auto_move_above_height_min = float(
            self.declare_parameter(
                "auto_move_above_height_min",
                AUTO_MOVE_ABOVE_HEIGHT_MIN,
            ).value
        )
        self.auto_move_above_height_max = float(
            self.declare_parameter(
                "auto_move_above_height_max",
                AUTO_MOVE_ABOVE_HEIGHT_MAX,
            ).value
        )
        self.auto_move_xy_max = float(
            self.declare_parameter("auto_move_xy_max", AUTO_MOVE_XY_MAX).value
        )
        self.auto_move_w_side_offset = float(
            self.declare_parameter("auto_move_w_side_offset", AUTO_MOVE_W_SIDE_OFFSET).value
        )
        self.auto_move_linear_gain = float(
            self.declare_parameter("auto_move_linear_gain", AUTO_MOVE_LINEAR_GAIN).value
        )
        self.auto_move_max_linear_velocity = float(
            self.declare_parameter(
                "auto_move_max_linear_velocity",
                AUTO_MOVE_MAX_LINEAR_VEL,
            ).value
        )
        self.auto_move_tolerance_m = float(
            self.declare_parameter("auto_move_tolerance_m", AUTO_MOVE_TOLERANCE_M).value
        )
        self.angular_control_frame = self.declare_parameter(
            "angular_control_frame",
            "gripper/tcp",
        ).value
        self.auto_move_tcp_frame = self.declare_parameter(
            "auto_move_tcp_frame", AUTO_MOVE_TCP_FRAME
        ).value
        self.insertion_event_topic = self.declare_parameter(
            "insertion_event_topic",
            "/scoring/insertion_event",
        ).value
        self._random_align_target_quaternion: np.ndarray | None = None
        self._auto_move_target_position_in_base: np.ndarray | None = None
        self._auto_move_waypoints_in_base: list[np.ndarray] = []
        self.auto_move_active = False
        self._random_rng = np.random.default_rng()

        self.motion_pub = self.create_publisher(
            MotionUpdate,
            f"/{self.controller_namespace}/pose_commands",
            10,
        )
        self.change_target_mode_client = self.create_client(
            ChangeTargetMode,
            f"/{self.controller_namespace}/change_target_mode",
        )
        self.insertion_event_sub = self.create_subscription(
            StdString,
            self.insertion_event_topic,
            self._on_insertion_event,
            10,
        )

        self.active_keys = set()
        self.paused = False
        self.speed_mode = SpeedMode.NORMAL
        self.linear_vel = self.normal_linear_velocity
        self.angular_vel = self.normal_angular_velocity
        self.auto_align_active = False
        self.frame_id = self.default_frame_id
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
        self._expected_relative_quaternion = _euler_xyz_degrees_to_quat_xyzw(
            DEFAULT_ANGLE_EXPECTED_RELATIVE_EULER_DEG
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
        self.get_logger().info(
            "Using task angle frames: "
            f"task_name={self.task_name}, "
            f"source={self.angle_source_frame}, target={self.angle_target_frame}"
        )
        self.timer = self.create_timer(1.0 / self.publish_rate, self.send_references)

    def _on_insertion_event(self, msg: StdString):
        port = str(msg.data).strip()
        if port:
            self.get_logger().info(f"Insertion complete: port={port}")
        else:
            self.get_logger().info("Insertion complete")

    def on_key_press(self, key):
        try:
            if hasattr(key, "char") and key.char is not None:
                char = key.char.lower()
                if char not in self.active_keys:
                    if char == "m":
                        self._toggle_speed_mode()
                    elif char == "n":
                        self._toggle_frame_id()
                    elif char == "z":
                        self._toggle_auto_align()
                    elif char == "x":
                        self._toggle_random_align()
                    elif char == "c":
                        self._toggle_move_above_random()
                    elif char == "r":
                        self._stop_and_exit()
                        return
                    elif char == "t":
                        self._toggle_pause()
                self.active_keys.add(char)
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

    def _toggle_speed_mode(self):
        if self.speed_mode == SpeedMode.NORMAL:
            self.speed_mode = SpeedMode.FAST
            self.linear_vel = self.fast_linear_velocity
            self.angular_vel = self.fast_angular_velocity
        else:
            self.speed_mode = SpeedMode.NORMAL
            self.linear_vel = self.normal_linear_velocity
            self.angular_vel = self.normal_angular_velocity
        self.get_logger().info(
            f"Speed mode: {self.speed_mode.value} "
            f"(linear={self.linear_vel:.4f}, angular={self.angular_vel:.4f})"
        )

    def _toggle_frame_id(self):
        if self.frame_id == "gripper/tcp":
            self.frame_id = "base_link"
        else:
            self.frame_id = "gripper/tcp"
        self.get_logger().info(f"Target frame_id: {self.frame_id}")

    def _toggle_auto_align(self):
        self.auto_align_active = not self.auto_align_active
        self._random_align_target_quaternion = None
        if self.auto_align_active:
            self.get_logger().info("Auto angle align: enabled")
        else:
            self.get_logger().info("Auto angle align: disabled")

    def _toggle_move_above_random(self):
        if self.auto_move_active:
            self.auto_move_active = False
            self._auto_move_target_position_in_base = None
            self._auto_move_waypoints_in_base = []
            self.get_logger().info("Auto move path: disabled")
            return

        try:
            target_in_base_tf = self._tf_buffer.lookup_transform(
                "base_link",
                self.angle_target_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
        except TransformException as exc:
            self.get_logger().warning(
                f"Auto move above: target TF unavailable ({exc})"
            )
            return

        target_pos = np.array(
            [
                target_in_base_tf.transform.translation.x,
                target_in_base_tf.transform.translation.y,
                target_in_base_tf.transform.translation.z,
            ],
            dtype=np.float64,
        )
        sampled_xy_offset = self._random_rng.uniform(
            -self.auto_move_xy_max,
            self.auto_move_xy_max,
            size=2,
        )
        sampled_above_height = float(
            self._random_rng.uniform(
                self.auto_move_above_height_min,
                self.auto_move_above_height_max,
            )
        )
        target_above_random = target_pos + np.array(
            [sampled_xy_offset[0], sampled_xy_offset[1], sampled_above_height],
            dtype=np.float64,
        )
        waypoint_w_side = target_above_random + np.array(
            [0.0, -self.auto_move_w_side_offset, 0.0],
            dtype=np.float64,
        )
        self._auto_move_waypoints_in_base = [waypoint_w_side, target_above_random]
        self._auto_move_target_position_in_base = self._auto_move_waypoints_in_base[0].copy()
        self.auto_move_active = True
        self.get_logger().info(
            "Auto move random path: enabled "
            f"offset=[{sampled_xy_offset[0]:.4f}, {sampled_xy_offset[1]:.4f}] "
            f"above_height={sampled_above_height:.4f} "
            f"waypoint=[{waypoint_w_side[0]:.4f}, {waypoint_w_side[1]:.4f}, {waypoint_w_side[2]:.4f}] "
            f"target=[{target_above_random[0]:.4f}, {target_above_random[1]:.4f}, {target_above_random[2]:.4f}]"
        )

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.active_keys.clear()
            self.auto_align_active = False
            self.auto_move_active = False
            self._random_align_target_quaternion = None
            self._auto_move_target_position_in_base = None
            self._auto_move_waypoints_in_base = []
            zero_twist = Twist()
            self.motion_pub.publish(
                self.generate_velocity_motion_update(zero_twist, self.frame_id)
            )
            self.get_logger().info("Keyboard teleop: PAUSED (press 't' to resume)")
        else:
            self.get_logger().info("Keyboard teleop: RESUMED")

    def _stop_and_exit(self):
        self.active_keys.clear()
        self.auto_align_active = False
        self.auto_move_active = False
        self._random_align_target_quaternion = None
        self._auto_move_target_position_in_base = None
        self._auto_move_waypoints_in_base = []
        zero_twist = Twist()
        self.motion_pub.publish(
            self.generate_velocity_motion_update(zero_twist, self.frame_id)
        )
        self.get_logger().info("Keyboard teleop: sent zero twist and exiting")
        rclpy.shutdown()

    def _toggle_random_align(self):
        if self.auto_align_active and self._random_align_target_quaternion is not None:
            self.auto_align_active = False
            self._random_align_target_quaternion = None
            self.get_logger().info("Random angle align: disabled")
            return
        sampled_euler_deg = self._random_rng.uniform(
            -self.random_align_max_deg,
            self.random_align_max_deg,
            size=3,
        )
        self._random_align_target_quaternion = _euler_xyz_degrees_to_quat_xyzw(sampled_euler_deg)
        self.auto_align_active = True
        self.get_logger().info(
            "Random angle align: enabled target="
            f"[{sampled_euler_deg[0]:.2f}, {sampled_euler_deg[1]:.2f}, {sampled_euler_deg[2]:.2f}] deg"
        )

    def _ensure_motion_subscriber_ready(self) -> bool:
        if self.motion_subscriber_ready:
            return True
        if self.motion_pub.get_subscription_count() > 0:
            self.motion_subscriber_ready = True
            return True
        return False

    def _ensure_cartesian_mode(self):
        if not self.change_target_mode_client.wait_for_service(timeout_sec=5.0):
            return
        req = ChangeTargetMode.Request()
        req.target_mode.mode = TargetMode.MODE_CARTESIAN
        future = self.change_target_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result()
        if response is None or not response.success:
            return

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

    def _lookup_frame_quaternion_in_base(self, frame_name: str) -> np.ndarray | None:
        try:
            frame_in_base_tf = self._tf_buffer.lookup_transform(
                "base_link",
                frame_name,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            return None
        return _quaternion_from_transform(frame_in_base_tf).astype(np.float64)

    def send_references(self):
        if not self._ensure_motion_subscriber_ready():
            return

        if self.paused:
            self._maybe_log_current_angle()
            return

        input_twist = np.zeros(6, dtype=np.float64)
        publish_frame_id = "base_link"

        if self.auto_align_active or self.auto_move_active:
            if self.auto_align_active:
                auto_angular = self._compute_auto_align_angular_velocity()
                if auto_angular is not None:
                    input_twist[3:6] = auto_angular
            if self.auto_move_active:
                auto_linear = self._compute_auto_move_linear_velocity()
                if auto_linear is not None:
                    input_twist[0:3] = auto_linear
            publish_frame_id = "base_link"
        else:
            angular_twist_local = np.zeros(3, dtype=np.float64)
            for key in self.active_keys:
                if key in KEY_MAPPINGS:
                    vals = KEY_MAPPINGS[key]
                    input_twist[0:3] += np.array(vals[0:3], dtype=np.float64) * self.linear_vel
                    angular_twist_local += np.array(vals[3:6], dtype=np.float64) * self.angular_vel
            if np.linalg.norm(angular_twist_local) > 0.0:
                angular_frame_quat = self._lookup_frame_quaternion_in_base(
                    self.angular_control_frame
                )
                if angular_frame_quat is None:
                    self.get_logger().warning(
                        f"Angular control frame unavailable: {self.angular_control_frame}"
                    )
                else:
                    input_twist[3:6] = _rotate_vector_by_quat_xyzw(
                        angular_twist_local,
                        angular_frame_quat,
                    )

        twist = Twist()
        twist.linear.x = float(input_twist[0])
        twist.linear.y = float(input_twist[1])
        twist.linear.z = float(input_twist[2])
        twist.angular.x = float(input_twist[3])
        twist.angular.y = float(input_twist[4])
        twist.angular.z = float(input_twist[5])

        self.motion_pub.publish(self.generate_velocity_motion_update(twist, publish_frame_id))

        self._maybe_log_current_angle()

    def _lookup_angle_error_quaternion(self) -> np.ndarray | None:
        try:
            source_in_target_tf = self._tf_buffer.lookup_transform(
                self.angle_target_frame,
                self.angle_source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            return None

        relative_quaternion = _quaternion_from_transform(source_in_target_tf)
        if self._expected_relative_quaternion is not None:
            relative_quaternion = _quat_multiply_xyzw(
                _quat_inverse_xyzw(self._expected_relative_quaternion),
                relative_quaternion,
            )
        if self._random_align_target_quaternion is not None:
            relative_quaternion = _quat_multiply_xyzw(
                _quat_inverse_xyzw(self._random_align_target_quaternion),
                relative_quaternion,
            )
        return relative_quaternion

    def _lookup_tip_frame_correction_rotvec_deg(self) -> np.ndarray | None:
        relative_quaternion = self._lookup_angle_error_quaternion()
        if relative_quaternion is None:
            return None
        correction_in_source = _quat_inverse_xyzw(relative_quaternion)
        rotvec_rad = _quat_xyzw_to_rotvec(correction_in_source)
        return np.degrees(rotvec_rad).astype(np.float32)

    def _compute_auto_align_angular_velocity(self) -> np.ndarray | None:
        relative_quaternion = self._lookup_angle_error_quaternion()
        if relative_quaternion is None:
            return None

        euler_deg = _quat_xyzw_to_euler_xyz_degrees(relative_quaternion)
        if np.max(np.abs(euler_deg)) <= self.auto_align_tolerance_deg:
            self.auto_align_active = False
            mode_label = "Random angle align" if self._random_align_target_quaternion is not None else "Auto angle align"
            self._random_align_target_quaternion = None
            self.get_logger().info(
                f"{mode_label}: reached "
                f"[{euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}] deg"
            )
            return np.zeros(3, dtype=np.float64)

        correction_target = -self.auto_align_angular_gain * _quat_xyzw_to_rotvec(
            relative_quaternion
        )
        norm = np.linalg.norm(correction_target)
        if norm > self.auto_align_max_angular_velocity:
            correction_target *= self.auto_align_max_angular_velocity / norm

        try:
            target_in_base_tf = self._tf_buffer.lookup_transform(
                "base_link",
                self.angle_target_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            return None

        target_to_base_quat = _quaternion_from_transform(target_in_base_tf)
        return _rotate_vector_by_quat_xyzw(correction_target, target_to_base_quat)

    def _compute_auto_move_linear_velocity(self) -> np.ndarray | None:
        if self._auto_move_target_position_in_base is None:
            return None
        try:
            tcp_in_base_tf = self._tf_buffer.lookup_transform(
                "base_link",
                self.auto_move_tcp_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            return None
        tcp_pos = np.array(
            [
                tcp_in_base_tf.transform.translation.x,
                tcp_in_base_tf.transform.translation.y,
                tcp_in_base_tf.transform.translation.z,
            ],
            dtype=np.float64,
        )
        error = self._auto_move_target_position_in_base - tcp_pos
        distance = float(np.linalg.norm(error))
        if distance <= self.auto_move_tolerance_m:
            if self._auto_move_waypoints_in_base:
                self._auto_move_waypoints_in_base.pop(0)
            if self._auto_move_waypoints_in_base:
                self._auto_move_target_position_in_base = self._auto_move_waypoints_in_base[0].copy()
                self.get_logger().info(
                    "Auto move path: next waypoint "
                    f"[{self._auto_move_target_position_in_base[0]:.4f}, "
                    f"{self._auto_move_target_position_in_base[1]:.4f}, "
                    f"{self._auto_move_target_position_in_base[2]:.4f}]"
                )
            else:
                self.auto_move_active = False
                self._auto_move_target_position_in_base = None
                self.get_logger().info(
                    f"Auto move path: reached final target (dist={distance * 1000.0:.2f} mm)"
                )
            return np.zeros(3, dtype=np.float64)
        velocity = self.auto_move_linear_gain * error
        speed = float(np.linalg.norm(velocity))
        if speed > self.auto_move_max_linear_velocity:
            velocity *= self.auto_move_max_linear_velocity / speed
        return velocity

    def _maybe_log_current_angle(self):
        now = time.monotonic()
        if now - self.last_angle_log_time < self.angle_print_period_sec:
            return
        self.last_angle_log_time = now

        tip_correction_deg = self._lookup_tip_frame_correction_rotvec_deg()
        if tip_correction_deg is None:
            return

        display_deg = np.array(
            [
                tip_correction_deg[1],
                tip_correction_deg[0],
                tip_correction_deg[2],
            ],
            dtype=np.float32,
        )
        zero_keys = {
            "first": _key_to_zero_angle("x", float(display_deg[0])),
            "second": _key_to_zero_angle("y", float(display_deg[1])),
            "third": _key_to_zero_angle("z", float(display_deg[2])),
        }
        self.get_logger().info(
            f"[{display_deg[0]:.2f}, {display_deg[1]:.2f}, {display_deg[2]:.2f}] "
            f"[{zero_keys['first']}, {zero_keys['second']}, {zero_keys['third']}]"
        )

    def cleanup(self):
        if rclpy.ok():
            zero_twist = Twist()
            self.motion_pub.publish(self.generate_velocity_motion_update(zero_twist, self.frame_id))
        self.keyboard_listener.stop()


def main(args=None):
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
