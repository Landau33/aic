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

import time

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Twist, Vector3, Wrench
from std_msgs.msg import String

from .hil_serl.adapter import HilSerlActionAdapter, HilSerlObservationAdapter
from .hil_serl.config import HilSerlRuntimeConfig
from .hil_serl.runtime import HilSerlActorRuntime


class TestPolicy(Policy):
    """在 deep-insert 阶段执行 HIL-SERL actor 推理。"""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._config = HilSerlRuntimeConfig()
        self._observation_adapter = HilSerlObservationAdapter(self._config.observation)
        self._action_adapter = HilSerlActionAdapter(self._config.control)
        self._runtime = HilSerlActorRuntime(self._config)

        self._deep_insert = False
        self._deepinsert_event_sub = parent_node.create_subscription(
            String,
            self._config.topics.deep_insert_topic,
            self._on_deepinsert_event,
            10,
        )
        self.get_logger().info("TestPolicy.__init__()")

    def _on_deepinsert_event(self, msg: String) -> None:
        self._deep_insert = msg.data.strip().lower() == "true"
        self.get_logger().info(f"deep_insert={self._deep_insert}")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"TestPolicy.insert_cable() enter. Task: {task}")
        self._deep_insert = False
        self._observation_adapter.reset()
        self._runtime.reset()

        while not self._deep_insert:
            send_feedback("等待 motion_planning 发布 deep_insert=true")
            self.sleep_for(0.5)

        self.get_logger().info("进入 deep-insert 阶段，开始 HIL-SERL actor 推理。")
        send_feedback("deep_insert 已接管，初始化 HIL-SERL actor")

        while True: 
            time.sleep(1)
            self.get_logger().info("准备进入推理，等待中...")

        init_obs_msg = self._wait_for_observation(get_observation)
        if init_obs_msg is None:
            self.get_logger().error("初始化失败：未收到 observation。")
            self._send_zero_twist(move_robot)
            return False

        init_obs = self._observation_adapter.adapt(init_obs_msg)
        self._runtime.initialize(init_obs)
        self._runtime.reset()

        start_time = time.time()
        step_index = 0
        missing_obs_count = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > self._config.safety.max_runtime_sec:
                self.get_logger().warn("deep-insert 超时，停止 HIL-SERL actor。")
                send_feedback("deep_insert 超时，停止控制")
                self._send_zero_twist(move_robot)
                return False

            obs_msg = get_observation()
            if obs_msg is None:
                missing_obs_count += 1
                if missing_obs_count >= self._config.safety.max_consecutive_missing_obs:
                    self.get_logger().error("连续丢失 observation，停止控制。")
                    send_feedback("observation 丢失，停止控制")
                    self._send_zero_twist(move_robot)
                    return False
                self.sleep_for(self._config.control.control_period_sec)
                continue

            missing_obs_count = 0

            actor_obs = self._observation_adapter.adapt(obs_msg)
            actor_action = self._runtime.predict(actor_obs)
            command = self._action_adapter.adapt(actor_action)
            twist = self._action_adapter.to_twist(command)
            move_robot(motion_update=self._set_cartesian_twist_target(twist))

            if step_index % 10 == 0:
                send_feedback(
                    f"deep_insert 推理中 step={step_index} elapsed={elapsed:.1f}s"
                )

            step_index += 1
            self.sleep_for(self._config.control.control_period_sec)

    def _wait_for_observation(
        self, get_observation: GetObservationCallback, timeout_sec: float = 5.0
    ):
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            obs = get_observation()
            if obs is not None:
                return obs
            self.sleep_for(0.1)
        return None

    def _send_zero_twist(self, move_robot: MoveRobotCallback) -> None:
        zero_twist = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )
        move_robot(motion_update=self._set_cartesian_twist_target(zero_twist))

    def _set_cartesian_twist_target(self, twist: Twist, frame_id: str = "base_link"):
        motion_update_msg = MotionUpdate()
        motion_update_msg.velocity = twist
        motion_update_msg.header.frame_id = frame_id
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()

        motion_update_msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        motion_update_msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()

        motion_update_msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update_msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        motion_update_msg.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
        )
        return motion_update_msg
