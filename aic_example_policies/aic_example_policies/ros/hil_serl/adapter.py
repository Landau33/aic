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

"""AIC Observation 与 HIL-SERL actor 输入/输出之间的适配。"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from aic_model_interfaces.msg import Observation
from geometry_msgs.msg import Twist, Vector3

from .config import HilSerlControlConfig, HilSerlObservationConfig
from .types import CartesianVelocityCommand, HilSerlAction, ObservationDict


class HilSerlObservationAdapter:
    """把 AIC Observation 转换成 HIL-SERL actor 期望的 observation dict。

    这里优先对齐 AIC toolkit 的真实观测来源，而不是 Franka RAM 例程的双相机输入。
    当前约定的 state 顺序为：
    - tcp_pose: 7 维，xyz + quat
    - tcp_vel: 6 维
    - tcp_error: 6 维
    - joint_positions: 7 维
    - joint_velocities: 7 维
    - joint_efforts: 7 维
    - wrist_force: 3 维
    - wrist_torque: 3 维

    合计 46 维。
    图像输入使用 AIC Observation 里的三路相机：
    - left_image  -> left_camera
    - center_image -> center_camera
    - right_image -> right_camera
    """

    def __init__(self, config: HilSerlObservationConfig):
        self._config = config
        self._history: deque[ObservationDict] = deque(maxlen=config.observation_horizon)

    def reset(self) -> None:
        self._history.clear()

    def adapt(self, obs_msg: Observation) -> ObservationDict:
        obs = {"state": self._build_state(obs_msg)[None, :]}
        for image_key, source_name in zip(
            self._config.image_keys, self._config.aic_image_topics, strict=True
        ):
            obs[image_key] = self._extract_image(obs_msg, source_name)[None, ...]
        self._history.append(obs)
        return obs

    def _build_state(self, obs_msg: Observation) -> np.ndarray:
        tcp_pose = obs_msg.controller_state.tcp_pose
        tcp_vel = obs_msg.controller_state.tcp_velocity
        wrench = obs_msg.wrist_wrench.wrench
        joint_positions = self._get_joint_array(obs_msg.joint_states.position)
        joint_velocities = self._get_joint_array(obs_msg.joint_states.velocity)
        joint_efforts = self._get_joint_array(obs_msg.joint_states.effort)

        state_parts: list[np.ndarray] = []
        for key in self._config.proprio_keys:
            if key == "tcp_pose":
                state_parts.append(
                    np.asarray(
                        [
                            tcp_pose.position.x,
                            tcp_pose.position.y,
                            tcp_pose.position.z,
                            tcp_pose.orientation.x,
                            tcp_pose.orientation.y,
                            tcp_pose.orientation.z,
                            tcp_pose.orientation.w,
                        ],
                        dtype=np.float32,
                    )
                )
            elif key == "tcp_vel":
                state_parts.append(
                    np.asarray(
                        [
                            tcp_vel.linear.x,
                            tcp_vel.linear.y,
                            tcp_vel.linear.z,
                            tcp_vel.angular.x,
                            tcp_vel.angular.y,
                            tcp_vel.angular.z,
                        ],
                        dtype=np.float32,
                    )
                )
            elif key == "tcp_error":
                state_parts.append(
                    np.asarray(
                        list(obs_msg.controller_state.tcp_error[:6]),
                        dtype=np.float32,
                    )
                )
            elif key == "joint_positions":
                state_parts.append(joint_positions)
            elif key == "joint_velocities":
                state_parts.append(joint_velocities)
            elif key == "joint_efforts":
                state_parts.append(joint_efforts)
            elif key == "wrist_force":
                state_parts.append(
                    np.asarray(
                        [
                            wrench.force.x,
                            wrench.force.y,
                            wrench.force.z,
                        ],
                        dtype=np.float32,
                    )
                )
            elif key == "wrist_torque":
                state_parts.append(
                    np.asarray(
                        [
                            wrench.torque.x,
                            wrench.torque.y,
                            wrench.torque.z,
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                raise ValueError(f"Unsupported proprio key: {key}")

        return np.concatenate(state_parts, axis=0).astype(np.float32)

    @staticmethod
    def _get_joint_array(values) -> np.ndarray:
        joint_array = np.zeros((7,), dtype=np.float32)
        available = min(7, len(values))
        if available > 0:
            joint_array[:available] = np.asarray(values[:available], dtype=np.float32)
        return joint_array

    def _extract_image(self, obs_msg: Observation, source_name: str) -> np.ndarray:
        if source_name == "left":
            image_msg = obs_msg.left_image
        elif source_name == "center":
            image_msg = obs_msg.center_image
        elif source_name == "right":
            image_msg = obs_msg.right_image
        else:
            raise ValueError(f"Unknown AIC image source: {source_name}")

        img = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            image_msg.height, image_msg.width, 3
        )
        if image_msg.encoding.lower() == "rgb8":
            img = img[..., ::-1]
        img = cv2.resize(
            img,
            (self._config.image_width, self._config.image_height),
            interpolation=cv2.INTER_AREA,
        )
        return img


class HilSerlActionAdapter:
    """把 HIL-SERL actor 输出转成 AIC 控制命令。"""

    def __init__(self, config: HilSerlControlConfig):
        self._config = config

    def adapt(self, action: HilSerlAction) -> CartesianVelocityCommand:
        values = np.asarray(action.values, dtype=np.float32).reshape(-1)
        if values.shape[0] < 6:
            raise ValueError(f"HIL-SERL action dim too small: {values.shape[0]}")

        # Match the training environment's action pipeline: only scale the
        # normalized policy output before sending velocity commands.
        linear = values[:3] * self._config.action_scale_linear
        angular = values[3:6] * self._config.action_scale_angular

        return CartesianVelocityCommand(linear_xyz=linear, angular_xyz=angular)

    @staticmethod
    def to_twist(command: CartesianVelocityCommand) -> Twist:
        return Twist(
            linear=Vector3(
                x=float(command.linear_xyz[0]),
                y=float(command.linear_xyz[1]),
                z=float(command.linear_xyz[2]),
            ),
            angular=Vector3(
                x=float(command.angular_xyz[0]),
                y=float(command.angular_xyz[1]),
                z=float(command.angular_xyz[2]),
            ),
        )
