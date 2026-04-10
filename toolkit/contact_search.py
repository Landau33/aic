#!/usr/bin/env python3

"""
Standalone contact-search controller for AIC cable insertion.

The controller follows a coarse-to-fine local search strategy:
1. Approach a pre-insert pose derived from the estimated port frame.
2. Descend slowly until light contact is detected.
3. Hold a small down-force by publishing position targets with feedforward force.
4. Probe rx/ry/rz with +/- angular perturbations.
5. Score each probe by insertion depth gain, lateral force, and torque magnitude.
6. Update pose toward the better direction and attempt a small insertion.
7. Back off and retry if the search stagnates.

Pause control:
- Press `r` to toggle pause/resume.
- Press `esc` to stop the script.
"""

import threading
import time
from collections import deque
from enum import Enum, auto

import numpy as np
import rclpy
from aic_control_interfaces.msg import MotionUpdate, TargetMode, TrajectoryGenerationMode
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Pose, Quaternion, Twist, Vector3, Wrench, WrenchStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener


class SearchState(Enum):
    SEEK_CONTACT = auto()
    CONTACT_HOLD = auto()
    BACKOFF_RETRY = auto()
    PAUSED = auto()


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


class KeyboardIntervention:
    def __init__(self):
        from pynput import keyboard

        self._keyboard = keyboard
        self._toggle_requested = False
        self._stop_requested = False
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._listener.start()

    def close(self):
        self._listener.stop()

    def _on_key_press(self, key):
        if key == self._keyboard.Key.esc:
            with self._lock:
                self._stop_requested = True
            return
        try:
            if hasattr(key, "char") and key.char is not None:
                char = key.char.lower()
                with self._lock:
                    if char == "r":
                        self._toggle_requested = True
        except AttributeError:
            return

    def _on_key_release(self, key):
        return

    def pop_toggle_requested(self) -> bool:
        with self._lock:
            value = self._toggle_requested
            self._toggle_requested = False
        return value

    def pop_stop_requested(self) -> bool:
        with self._lock:
            value = self._stop_requested
            self._stop_requested = False
        return value


class AICContactSearchNode(Node):
    def __init__(self):
        super().__init__("aic_contact_search")
        self.set_parameters([
            Parameter("use_sim_time", Parameter.Type.BOOL, True),
        ])

        self._declare_parameters()
        self._load_parameters()

        self.motion_pub = self.create_publisher(
            MotionUpdate,
            f"/{self.controller_namespace}/pose_commands",
            10,
        )
        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            self.processed_wrench_topic,
            self.wrench_callback,
            20,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.change_target_mode_client = self.create_client(
            ChangeTargetMode,
            f"/{self.controller_namespace}/change_target_mode",
        )

        self.current_wrench_msg = None
        self.current_force_base = np.zeros(3, dtype=np.float64)
        self.current_torque_base = np.zeros(3, dtype=np.float64)
        self.previous_torque_base = np.zeros(3, dtype=np.float64)
        self.torque_delta_base = np.zeros(3, dtype=np.float64)
        self.has_torque_history = False
        self.sample_window: deque[dict] = deque(maxlen=400)

        self.state = SearchState.CONTACT_HOLD
        self.state_started_at = time.monotonic()
        self.previous_auto_state = self.state
        self.motion_subscriber_ready = False
        self.last_wait_log_time = 0.0
        self.contact_pose = None
        self.backoff_start_pose = None
        self.stuck_detected_since = None
        self.last_stuck_warn_time = 0.0
        self.last_relief_axes: set[str] = set()

        self.keyboard = None
        if self.enable_keyboard_intervention:
            try:
                self.keyboard = KeyboardIntervention()
                self.get_logger().info(
                    "Pause control enabled. Press `r` to pause/resume, `esc` to exit."
                )
            except Exception as exc:
                self.get_logger().warning(f"Keyboard intervention unavailable: {exc}")

        self._ensure_cartesian_mode()

        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        self.get_logger().info("Contact search controller started from CONTACT_HOLD.")

    def _declare_parameters(self):
        self.declare_parameter("controller_namespace", "aic_controller")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("tcp_frame", "gripper/tcp")
        self.declare_parameter("nic_port_frame", "nic_port_rec")
        self.declare_parameter("processed_wrench_topic", "/nic_insertion/processed_wrench/filtered")
        self.declare_parameter("publish_rate", 25.0)
        self.declare_parameter("seek_velocity_z", -0.0025)
        self.declare_parameter("contact_force_threshold", 2.0)
        self.declare_parameter("hold_force_feedforward_z", 4.0)
        self.declare_parameter("probe_trigger_force_z", 10.0)
        self.declare_parameter("probe_trigger_force_xy", 10.0)
        self.declare_parameter("probe_trigger_torque", 3)
        self.declare_parameter("torque_increase_trigger", 0.12)
        self.declare_parameter("torque_increase_min_torque", 0.2)
        self.declare_parameter("relief_linear_gain", 0.0015)
        self.declare_parameter("relief_angular_gain", 0.08)
        self.declare_parameter("max_relief_linear_velocity", 0.008)
        self.declare_parameter("max_relief_angular_velocity", 0.25)
        self.declare_parameter("stuck_detection_sec", 0.5)
        self.declare_parameter("stuck_warn_interval_sec", 1.0)
        self.declare_parameter("max_force_z", 15.0)
        self.declare_parameter("max_force_xy", 12.0)
        self.declare_parameter("max_torque", 5)
        self.declare_parameter("backoff_distance_m", 0.0020)
        self.declare_parameter("backoff_duration_sec", 0.8)
        self.declare_parameter("enable_keyboard_intervention", True)

    def _load_parameters(self):
        self.controller_namespace = self.get_parameter("controller_namespace").value
        self.frame_id = self.get_parameter("frame_id").value
        self.tcp_frame = self.get_parameter("tcp_frame").value
        self.nic_port_frame = self.get_parameter("nic_port_frame").value
        self.processed_wrench_topic = self.get_parameter("processed_wrench_topic").value
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.seek_velocity_z = float(self.get_parameter("seek_velocity_z").value)
        self.contact_force_threshold = float(self.get_parameter("contact_force_threshold").value)
        self.hold_force_feedforward_z = float(self.get_parameter("hold_force_feedforward_z").value)
        self.probe_trigger_force_z = float(self.get_parameter("probe_trigger_force_z").value)
        self.probe_trigger_force_xy = float(self.get_parameter("probe_trigger_force_xy").value)
        self.probe_trigger_torque = float(self.get_parameter("probe_trigger_torque").value)
        self.torque_increase_trigger = float(self.get_parameter("torque_increase_trigger").value)
        self.torque_increase_min_torque = float(self.get_parameter("torque_increase_min_torque").value)
        self.relief_linear_gain = float(self.get_parameter("relief_linear_gain").value)
        self.relief_angular_gain = float(self.get_parameter("relief_angular_gain").value)
        self.max_relief_linear_velocity = float(self.get_parameter("max_relief_linear_velocity").value)
        self.max_relief_angular_velocity = float(self.get_parameter("max_relief_angular_velocity").value)
        self.stuck_detection_sec = float(self.get_parameter("stuck_detection_sec").value)
        self.stuck_warn_interval_sec = float(self.get_parameter("stuck_warn_interval_sec").value)
        self.max_force_z = float(self.get_parameter("max_force_z").value)
        self.max_force_xy = float(self.get_parameter("max_force_xy").value)
        self.max_torque = float(self.get_parameter("max_torque").value)
        self.backoff_distance_m = float(self.get_parameter("backoff_distance_m").value)
        self.backoff_duration_sec = float(self.get_parameter("backoff_duration_sec").value)
        self.enable_keyboard_intervention = bool(self.get_parameter("enable_keyboard_intervention").value)

    def wrench_callback(self, msg: WrenchStamped):
        self.current_wrench_msg = msg
        self.current_force_base = np.array(
            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
            dtype=np.float64,
        )
        new_torque_base = np.array(
            [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
            dtype=np.float64,
        )
        if self.has_torque_history:
            self.torque_delta_base = new_torque_base - self.previous_torque_base
        else:
            self.torque_delta_base = np.zeros(3, dtype=np.float64)
            self.has_torque_history = True
        self.current_torque_base = new_torque_base
        self.previous_torque_base = new_torque_base.copy()

        tcp_pose = self.get_current_tcp_pose(log_failure=False)
        if tcp_pose is None:
            return
        self.sample_window.append(
            {
                "time": time.monotonic(),
                "force": self.current_force_base.copy(),
                "torque": self.current_torque_base.copy(),
                "pose_z": tcp_pose.position.z,
            }
        )

    def timer_callback(self):
        if not self._ensure_motion_subscriber_ready():
            return

        if self.keyboard is not None and self.keyboard.pop_stop_requested():
            self.get_logger().info("Keyboard stop requested.")
            self.stop_motion()
            rclpy.shutdown()
            return

        self._handle_pause_control()
        if self.state == SearchState.PAUSED:
            return

        if self.state == SearchState.SEEK_CONTACT:
            self._run_seek_contact()
        elif self.state == SearchState.CONTACT_HOLD:
            self._run_contact_hold()
        elif self.state == SearchState.BACKOFF_RETRY:
            self._run_backoff_retry()
        elif self.state == SearchState.PAUSED:
            self.stop_motion()

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

    def _handle_pause_control(self):
        if self.keyboard is None:
            return
        if self.keyboard.pop_toggle_requested():
            if self.state == SearchState.PAUSED:
                self.state = self.previous_auto_state
                self.state_started_at = time.monotonic()
                self.get_logger().info(f"Automation resumed in state {self.state.name}.")
                self.stop_motion()
                return

            self.previous_auto_state = self.state
            self.state = SearchState.PAUSED
            self.state_started_at = time.monotonic()
            self.stop_motion()
            self.get_logger().warning("Automation paused. Press `r` to resume.")

    def _run_seek_contact(self):
        if self._hard_limit_triggered():
            self.get_logger().warning("Hard limit triggered during seek. Stopping.")
            self.stop_motion()
            self._transition_to(SearchState.BACKOFF_RETRY, "Seek contact hit a safety limit.")
            return

        if self.current_force_base[2] >= self.contact_force_threshold:
            self.contact_pose = self.get_current_tcp_pose()
            self.stop_motion()
            self._transition_to(SearchState.CONTACT_HOLD, "Light contact detected.")
            return

        twist = Twist()
        twist.linear.z = self.seek_velocity_z
        self.motion_pub.publish(self.generate_velocity_update(twist))

    def _run_contact_hold(self):
        if self.contact_pose is None:
            self.contact_pose = self.get_current_tcp_pose()
            if self.contact_pose is None:
                return

        if self._hard_limit_triggered():
            self.stop_motion()
            self._transition_to(
                SearchState.BACKOFF_RETRY,
                "Contact hold exceeded hard safety limit.",
            )
            return

        twist = self._build_relief_twist()
        self._check_and_warn_if_stuck(twist)
        self.motion_pub.publish(self.generate_velocity_update(twist))

    def _run_backoff_retry(self):
        current_pose = self.get_current_tcp_pose()
        if current_pose is None:
            return
        if self.backoff_start_pose is None:
            self.backoff_start_pose = self.copy_pose(current_pose)
        target_pose = self.copy_pose(self.backoff_start_pose)
        target_pose.position.z += self.backoff_distance_m
        msg = self.generate_position_update(target_pose)
        self.motion_pub.publish(msg)

        if time.monotonic() - self.state_started_at < self.backoff_duration_sec:
            return

        self.backoff_start_pose = None
        self.contact_pose = None
        self._transition_to(SearchState.SEEK_CONTACT, "Backoff complete. Re-seeking contact.")

    def _hard_limit_triggered(self) -> bool:
        force_xy = abs(self.current_force_base[0]) + abs(self.current_force_base[1])
        torque_mag = np.linalg.norm(self.current_torque_base, ord=1)
        return (
            abs(self.current_force_base[2]) > self.max_force_z
            or force_xy > self.max_force_xy
            or torque_mag > self.max_torque
        )

    def _build_relief_twist(self) -> Twist:
        twist = Twist()
        twist.linear.z = self.seek_velocity_z
        active_relief_axes = set()

        if abs(self.current_force_base[0]) >= self.probe_trigger_force_xy:
            active_relief_axes.add("Fx")
            twist.linear.x = float(
                np.clip(
                    -self.relief_linear_gain * self.current_force_base[0],
                    -self.max_relief_linear_velocity,
                    self.max_relief_linear_velocity,
                )
            )
        if abs(self.current_force_base[1]) >= self.probe_trigger_force_xy:
            active_relief_axes.add("Fy")
            twist.linear.y = float(
                np.clip(
                    -self.relief_linear_gain * self.current_force_base[1],
                    -self.max_relief_linear_velocity,
                    self.max_relief_linear_velocity,
                )
            )
        if self._torque_adjustment_triggered(0):
            active_relief_axes.add("Mx")
            twist.angular.x = float(
                np.clip(
                    -self.relief_angular_gain * self.current_torque_base[0],
                    -self.max_relief_angular_velocity,
                    self.max_relief_angular_velocity,
                )
            )
        if self._torque_adjustment_triggered(1):
            active_relief_axes.add("My")
            twist.angular.y = float(
                np.clip(
                    -self.relief_angular_gain * self.current_torque_base[1],
                    -self.max_relief_angular_velocity,
                    self.max_relief_angular_velocity,
                )
            )
        if self._torque_adjustment_triggered(2):
            active_relief_axes.add("Mz")
            twist.angular.z = float(
                np.clip(
                    -self.relief_angular_gain * self.current_torque_base[2],
                    -self.max_relief_angular_velocity,
                    self.max_relief_angular_velocity,
                )
            )

        if (
            abs(self.current_force_base[2]) >= self.probe_trigger_force_z
            or abs(twist.angular.x) > 0.0
            or abs(twist.angular.y) > 0.0
            or abs(twist.angular.z) > 0.0
        ):
            active_relief_axes.add("Fz")
            twist.linear.z = float(0.5 * self.seek_velocity_z)

        self._log_relief_trigger(active_relief_axes)
        return twist

    def _torque_adjustment_triggered(self, axis_idx: int) -> bool:
        current_abs = abs(self.current_torque_base[axis_idx])
        delta_abs = abs(self.torque_delta_base[axis_idx])
        return (
            current_abs >= self.probe_trigger_torque
            or (
                current_abs >= self.torque_increase_min_torque
                and delta_abs >= self.torque_increase_trigger
            )
        )

    def _log_relief_trigger(self, active_relief_axes: set[str]):
        new_axes = active_relief_axes - self.last_relief_axes
        if not new_axes:
            self.last_relief_axes = active_relief_axes
            return

        details = []
        for axis in sorted(new_axes):
            if axis == "Fx":
                details.append(f"Fx={self.current_force_base[0]:+.2f}")
            elif axis == "Fy":
                details.append(f"Fy={self.current_force_base[1]:+.2f}")
            elif axis == "Fz":
                details.append(f"Fz={self.current_force_base[2]:+.2f}")
            elif axis == "Mx":
                details.append(
                    f"Mx={self.current_torque_base[0]:+.2f}, dMx={self.torque_delta_base[0]:+.2f}"
                )
            elif axis == "My":
                details.append(
                    f"My={self.current_torque_base[1]:+.2f}, dMy={self.torque_delta_base[1]:+.2f}"
                )
            elif axis == "Mz":
                details.append(
                    f"Mz={self.current_torque_base[2]:+.2f}, dMz={self.torque_delta_base[2]:+.2f}"
                )

        self.get_logger().info(
            "Relief triggered on "
            f"{', '.join(sorted(new_axes))}: "
            + "; ".join(details)
        )
        self.last_relief_axes = active_relief_axes

    def _check_and_warn_if_stuck(self, twist: Twist):
        linear_saturated = (
            abs(twist.linear.x) >= 0.95 * self.max_relief_linear_velocity
            or abs(twist.linear.y) >= 0.95 * self.max_relief_linear_velocity
        )
        angular_saturated = (
            abs(twist.angular.x) >= 0.95 * self.max_relief_angular_velocity
            or abs(twist.angular.y) >= 0.95 * self.max_relief_angular_velocity
            or abs(twist.angular.z) >= 0.95 * self.max_relief_angular_velocity
        )
        correction_active = bool(self.last_relief_axes)
        likely_stuck = correction_active and (linear_saturated or angular_saturated)
        now = time.monotonic()

        if likely_stuck:
            if self.stuck_detected_since is None:
                self.stuck_detected_since = now
            elif (
                now - self.stuck_detected_since >= self.stuck_detection_sec
                and now - self.last_stuck_warn_time >= self.stuck_warn_interval_sec
            ):
                self.last_stuck_warn_time = now
                self.get_logger().warning(
                    "Stuck: relief command is saturated but force/torque is still high. "
                    f"force=[{self.current_force_base[0]:.2f}, {self.current_force_base[1]:.2f}, {self.current_force_base[2]:.2f}] "
                    f"torque=[{self.current_torque_base[0]:.2f}, {self.current_torque_base[1]:.2f}, {self.current_torque_base[2]:.2f}]"
                )
            return

        self.stuck_detected_since = None

    def get_current_tcp_pose(self, log_failure: bool = True) -> Pose | None:
        try:
            trans = self.tf_buffer.lookup_transform(
                "base_link",
                self.tcp_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
        except Exception as exc:
            if log_failure:
                self.get_logger().warning(f"Failed to get current tcp pose: {exc}")
            return None

        pose = Pose()
        pose.position.x = trans.transform.translation.x
        pose.position.y = trans.transform.translation.y
        pose.position.z = trans.transform.translation.z
        pose.orientation = Quaternion(
            x=trans.transform.rotation.x,
            y=trans.transform.rotation.y,
            z=trans.transform.rotation.z,
            w=trans.transform.rotation.w,
        )
        return pose

    @staticmethod
    def copy_pose(pose: Pose) -> Pose:
        copied_pose = Pose()
        copied_pose.position.x = pose.position.x
        copied_pose.position.y = pose.position.y
        copied_pose.position.z = pose.position.z
        copied_pose.orientation = Quaternion(
            x=pose.orientation.x,
            y=pose.orientation.y,
            z=pose.orientation.z,
            w=pose.orientation.w,
        )
        return copied_pose

    def pose_signed_relative_euler_deg(self, source_pose: Pose, target_pose: Pose) -> np.ndarray:
        source_quat = np.array(
            [
                source_pose.orientation.x,
                source_pose.orientation.y,
                source_pose.orientation.z,
                source_pose.orientation.w,
            ],
            dtype=np.float64,
        )
        target_quat = np.array(
            [
                target_pose.orientation.x,
                target_pose.orientation.y,
                target_pose.orientation.z,
                target_pose.orientation.w,
            ],
            dtype=np.float64,
        )
        relative_quat = _quat_multiply_xyzw(
            _quat_inverse_xyzw(target_quat),
            source_quat,
        )
        return _quat_xyzw_to_euler_xyz_degrees(relative_quat)

    def generate_velocity_update(self, twist: Twist) -> MotionUpdate:
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.velocity = twist
        msg.target_stiffness = np.diag([85.0, 85.0, 85.0, 85.0, 85.0, 85.0]).flatten().tolist()
        msg.target_damping = np.diag([75.0, 75.0, 75.0, 75.0, 75.0, 75.0]).flatten().tolist()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0] * 6
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg

    def stop_motion(self):
        zero_twist = Twist()
        self.motion_pub.publish(self.generate_velocity_update(zero_twist))

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

    def _transition_to(self, state: SearchState, message: str):
        self.state = state
        self.state_started_at = time.monotonic()
        angle_suffix = self._current_angle_log_suffix()
        self.get_logger().info(f"{message} -> {state.name}{angle_suffix}")

    def _current_angle_log_suffix(self) -> str:
        current_pose = self.get_current_tcp_pose(log_failure=False)
        reference_pose = self._current_reference_pose()
        if current_pose is None or reference_pose is None:
            return ""
        euler_deg = self.pose_signed_relative_euler_deg(current_pose, reference_pose)
        return f" angle_deg=[{euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}]"

    def _current_reference_pose(self) -> Pose | None:
        for pose in (self.contact_pose,):
            if pose is not None:
                return pose
        return None

    def cleanup(self):
        self.stop_motion()
        if self.keyboard is not None:
            self.keyboard.close()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AICContactSearchNode()
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
