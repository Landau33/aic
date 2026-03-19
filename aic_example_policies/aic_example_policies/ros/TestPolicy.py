#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Twist, Vector3, Wrench


@dataclass
class DeepInsertionConfig:
    wait_for_contact_trigger: bool = False
    wait_timeout_sec: float = 30.0
    control_timeout_sec: float = 12.0
    control_period_sec: float = 0.1
    contact_force_threshold_n: float = 8.0
    hole_entry_force_threshold_n: float = 4.0
    ready_dwell_sec: float = 0.3
    failure_force_threshold_n: float = 30.0
    failure_torque_threshold_nm: float = 4.0
    success_z_error_threshold_m: float = 0.003
    success_lateral_force_threshold_n: float = 6.0
    success_pitch_roll_torque_threshold_nm: float = 1.0
    max_linear_speed_xy_mps: float = 0.004
    max_linear_speed_z_mps: float = 0.006
    max_angular_speed_xy_radps: float = 0.08
    max_angular_speed_z_radps: float = 0.03
    downward_bias_mps: float = -0.0015


class _ExportedStatePolicy:
    """Execution-side hook for a state-only HIL-SERL actor.

    If `AIC_HIL_SERL_STATE_POLICY_PATH` points to an `.npz` exported actor, the
    file is expected to contain:
      - `obs_mean`, `obs_std`, `act_mean`, `act_std`
      - `w0`, `b0`, `w1`, `b1`, ... for an MLP with tanh activations

    Without an exported actor, a conservative fallback policy is used so the
    stage machine remains runnable.
    """

    def __init__(self, logger):
        self._logger = logger
        self._loaded = False
        self._obs_mean = None
        self._obs_std = None
        self._act_mean = None
        self._act_std = None
        self._layers: list[tuple[np.ndarray, np.ndarray]] = []
        policy_path = os.getenv("AIC_HIL_SERL_STATE_POLICY_PATH", "").strip()
        if policy_path:
            self._try_load_npz(policy_path)

    def _try_load_npz(self, policy_path: str) -> None:
        try:
            with np.load(policy_path) as data:
                self._obs_mean = data["obs_mean"].astype(np.float32)
                self._obs_std = np.maximum(data["obs_std"].astype(np.float32), 1e-6)
                self._act_mean = data["act_mean"].astype(np.float32)
                self._act_std = np.maximum(data["act_std"].astype(np.float32), 1e-6)

                layer_idx = 0
                while f"w{layer_idx}" in data and f"b{layer_idx}" in data:
                    self._layers.append(
                        (
                            data[f"w{layer_idx}"].astype(np.float32),
                            data[f"b{layer_idx}"].astype(np.float32),
                        )
                    )
                    layer_idx += 1

            if not self._layers:
                raise ValueError("No MLP layers found in exported actor file.")

            self._loaded = True
            self._logger.info(f"Loaded exported state-only actor from {policy_path}")
        except Exception as exc:
            self._logger.warn(
                f"Failed to load exported actor from {policy_path}: {exc}. "
                "Falling back to scripted deep-insertion control."
            )

    def reset(self) -> None:
        pass

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        if self._loaded:
            return self._sample_exported_action(state)
        return self._sample_fallback_action(state)

    def _sample_exported_action(self, state: np.ndarray) -> np.ndarray:
        x = (state.astype(np.float32) - self._obs_mean) / self._obs_std
        for idx, (weight, bias) in enumerate(self._layers):
            x = x @ weight + bias
            if idx != len(self._layers) - 1:
                x = np.tanh(x)
        action = np.tanh(x) * self._act_std + self._act_mean
        return np.asarray(action, dtype=np.float32)

    def _sample_fallback_action(self, state: np.ndarray) -> np.ndarray:
        tcp_error = state[0:6]
        tcp_vel = state[6:12]
        force = state[12:15]
        torque = state[15:18]

        action = np.zeros(6, dtype=np.float32)
        action[0] = np.clip(-12.0 * tcp_error[0] - 1.5 * tcp_vel[0], -1.0, 1.0)
        action[1] = np.clip(-12.0 * tcp_error[1] - 1.5 * tcp_vel[1], -1.0, 1.0)
        action[2] = np.clip(-8.0 * tcp_error[2] - 0.8 * tcp_vel[2] - 0.015 * force[2], -1.0, 1.0)
        action[3] = np.clip(-3.0 * tcp_error[3] - 0.3 * tcp_vel[3] - 0.05 * torque[0], -1.0, 1.0)
        action[4] = np.clip(-3.0 * tcp_error[4] - 0.3 * tcp_vel[4] - 0.05 * torque[1], -1.0, 1.0)
        action[5] = np.clip(-2.0 * tcp_error[5] - 0.2 * tcp_vel[5], -1.0, 1.0)
        return action


class TestPolicy(Policy):
    """State-only deep-insertion policy with a HIL-SERL-style actor loop."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.config = DeepInsertionConfig(
            wait_for_contact_trigger=os.getenv("AIC_DEEP_INSERT_WAIT_FOR_TRIGGER", "false").lower() in ("1", "true", "yes"),
            wait_timeout_sec=float(os.getenv("AIC_DEEP_INSERT_WAIT_TIMEOUT_SEC", "30.0")),
            control_timeout_sec=float(os.getenv("AIC_DEEP_INSERT_TIMEOUT_SEC", "12.0")),
            control_period_sec=float(os.getenv("AIC_DEEP_INSERT_PERIOD_SEC", "0.1")),
            contact_force_threshold_n=float(os.getenv("AIC_CONTACT_FORCE_THRESHOLD_N", "8.0")),
            hole_entry_force_threshold_n=float(os.getenv("AIC_HOLE_ENTRY_FORCE_THRESHOLD_N", "4.0")),
            ready_dwell_sec=float(os.getenv("AIC_DEEP_INSERT_READY_DWELL_SEC", "0.3")),
            failure_force_threshold_n=float(os.getenv("AIC_DEEP_INSERT_FAILURE_FORCE_N", "30.0")),
            failure_torque_threshold_nm=float(os.getenv("AIC_DEEP_INSERT_FAILURE_TORQUE_NM", "4.0")),
            success_z_error_threshold_m=float(os.getenv("AIC_DEEP_INSERT_SUCCESS_Z_ERR_M", "0.003")),
            success_lateral_force_threshold_n=float(os.getenv("AIC_DEEP_INSERT_SUCCESS_LATERAL_FORCE_N", "6.0")),
            success_pitch_roll_torque_threshold_nm=float(os.getenv("AIC_DEEP_INSERT_SUCCESS_TORQUE_NM", "1.0")),
            max_linear_speed_xy_mps=float(os.getenv("AIC_DEEP_INSERT_MAX_VXY_MPS", "0.004")),
            max_linear_speed_z_mps=float(os.getenv("AIC_DEEP_INSERT_MAX_VZ_MPS", "0.006")),
            max_angular_speed_xy_radps=float(os.getenv("AIC_DEEP_INSERT_MAX_WXY_RADPS", "0.08")),
            max_angular_speed_z_radps=float(os.getenv("AIC_DEEP_INSERT_MAX_WZ_RADPS", "0.03")),
            downward_bias_mps=float(os.getenv("AIC_DEEP_INSERT_DOWNWARD_BIAS_MPS", "-0.0015")),
        )
        self.actor = _ExportedStatePolicy(self.get_logger())
        self.get_logger().info("TestPolicy.__init__()")

    def _extract_tared_wrench(self, obs: Observation) -> np.ndarray:
        raw = obs.wrist_wrench.wrench
        tare = obs.controller_state.fts_tare_offset.wrench
        return np.array(
            [
                raw.force.x - tare.force.x,
                raw.force.y - tare.force.y,
                raw.force.z - tare.force.z,
                raw.torque.x - tare.torque.x,
                raw.torque.y - tare.torque.y,
                raw.torque.z - tare.torque.z,
            ],
            dtype=np.float32,
        )

    def _build_state_vector(self, obs: Observation) -> np.ndarray:
        tcp_vel = obs.controller_state.tcp_velocity
        tcp_error = np.asarray(obs.controller_state.tcp_error, dtype=np.float32)
        tared_wrench = self._extract_tared_wrench(obs)
        return np.concatenate(
            [
                tcp_error,
                np.array(
                    [
                        tcp_vel.linear.x,
                        tcp_vel.linear.y,
                        tcp_vel.linear.z,
                        tcp_vel.angular.x,
                        tcp_vel.angular.y,
                        tcp_vel.angular.z,
                    ],
                    dtype=np.float32,
                ),
                tared_wrench[:3],
                tared_wrench[3:],
            ]
        )

    def _deep_insertion_ready(self, obs: Observation, contact_seen_time: Optional[float]) -> tuple[bool, Optional[float]]:
        tared_wrench = self._extract_tared_wrench(obs)
        force_z = abs(float(tared_wrench[2]))

        if contact_seen_time is None and force_z >= self.config.contact_force_threshold_n:
            contact_seen_time = time.monotonic()

        ready = (
            contact_seen_time is not None
            and (time.monotonic() - contact_seen_time) >= self.config.ready_dwell_sec
            and force_z <= self.config.hole_entry_force_threshold_n
        )
        return ready, contact_seen_time

    def _is_success(self, obs: Observation) -> bool:
        tared_wrench = self._extract_tared_wrench(obs)
        tcp_error = np.asarray(obs.controller_state.tcp_error, dtype=np.float32)
        lateral_force = np.linalg.norm(tared_wrench[:2])
        roll_pitch_torque = np.linalg.norm(tared_wrench[3:5])
        return (
            abs(float(tcp_error[2])) <= self.config.success_z_error_threshold_m
            and lateral_force <= self.config.success_lateral_force_threshold_n
            and roll_pitch_torque <= self.config.success_pitch_roll_torque_threshold_nm
        )

    def _is_failure(self, obs: Observation) -> bool:
        tared_wrench = self._extract_tared_wrench(obs)
        force_mag = np.linalg.norm(tared_wrench[:3])
        torque_mag = np.linalg.norm(tared_wrench[3:])
        return (
            force_mag >= self.config.failure_force_threshold_n
            or torque_mag >= self.config.failure_torque_threshold_nm
        )

    def _zero_twist(self) -> Twist:
        return Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )

    def _action_to_twist(self, action: np.ndarray) -> Twist:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return Twist(
            linear=Vector3(
                x=float(action[0] * self.config.max_linear_speed_xy_mps),
                y=float(action[1] * self.config.max_linear_speed_xy_mps),
                z=float(
                    action[2] * self.config.max_linear_speed_z_mps
                    + self.config.downward_bias_mps
                ),
            ),
            angular=Vector3(
                x=float(action[3] * self.config.max_angular_speed_xy_radps),
                y=float(action[4] * self.config.max_angular_speed_xy_radps),
                z=float(action[5] * self.config.max_angular_speed_z_radps),
            ),
        )

    def _make_motion_update(self, twist: Twist, frame_id: str = "base_link") -> MotionUpdate:
        msg = MotionUpdate()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = twist
        msg.target_stiffness = np.diag(
            [200.0, 200.0, 80.0, 120.0, 120.0, 60.0]
        ).flatten()
        msg.target_damping = np.diag(
            [30.0, 30.0, 45.0, 15.0, 15.0, 10.0]
        ).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=4.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.2, 0.2, 0.3, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg

    def _stop_robot(self, move_robot: MoveRobotCallback) -> None:
        move_robot(motion_update=self._make_motion_update(self._zero_twist()))

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"TestPolicy.insert_cable() enter. Task: {task}")
        self.actor.reset()
        if self.config.wait_for_contact_trigger:
            send_feedback("Waiting for deep-insertion trigger from contact state")

            wait_start = time.monotonic()
            contact_seen_time: Optional[float] = None

            while time.monotonic() - wait_start < self.config.wait_timeout_sec:
                obs = get_observation()
                if obs is None:
                    self.sleep_for(self.config.control_period_sec)
                    continue

                ready, contact_seen_time = self._deep_insertion_ready(obs, contact_seen_time)
                if ready:
                    self.get_logger().info("Deep insertion trigger detected. Starting RL actor.")
                    break

                self.sleep_for(self.config.control_period_sec)
            else:
                self.get_logger().warn("Timed out waiting for deep-insertion trigger.")
                return False
        else:
            self.get_logger().info("External deep-insertion trigger assumed. Starting RL actor immediately.")

        send_feedback("Deep insertion RL active")
        rl_start = time.monotonic()

        while time.monotonic() - rl_start < self.config.control_timeout_sec:
            loop_start = time.monotonic()
            obs = get_observation()
            if obs is None:
                self.sleep_for(self.config.control_period_sec)
                continue

            if self._is_failure(obs):
                self.get_logger().warn("Deep insertion aborted due to force/torque safety limit.")
                self._stop_robot(move_robot)
                return False

            if self._is_success(obs):
                self.get_logger().info("Deep insertion success condition satisfied.")
                self._stop_robot(move_robot)
                return True

            state = self._build_state_vector(obs)
            action = self.actor.sample_action(state)
            move_robot(motion_update=self._make_motion_update(self._action_to_twist(action)))
            send_feedback("Deep insertion RL step")

            elapsed = time.monotonic() - loop_start
            self.sleep_for(max(0.0, self.config.control_period_sec - elapsed))

        self.get_logger().warn("Deep insertion RL timed out.")
        self._stop_robot(move_robot)
        return False
