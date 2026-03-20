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

"""AIC 中运行 HIL-SERL actor 推理所需的配置。"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HilSerlTopicConfig:
    """与阶段切换和可选外部控制相关的 topic 配置。"""

    deep_insert_topic: str = "/aic/deep_insert"


@dataclass(frozen=True)
class HilSerlModelConfig:
    """HIL-SERL actor 模型相关配置。"""

    exp_name: str = "ram_insertion"
    setup_mode: str = "single-arm-fixed-gripper"
    checkpoint_path: str = "/home/young/ws_aic/hil-serl/examples/experiments/ram_insertion/first_run"
    checkpoint_step: int = 0
    seed: int = 42
    argmax: bool = False
    encoder_type: str = "resnet-pretrained"
    action_dim: int = 6
    image_keys: tuple[str, ...] = ("left_camera", "center_camera", "right_camera")


@dataclass(frozen=True)
class HilSerlObservationConfig:
    """把 AIC Observation 对齐到 HIL-SERL 输入时需要的配置。"""

    image_width: int = 128
    image_height: int = 128
    image_keys: tuple[str, ...] = ("left_camera", "center_camera", "right_camera")
    aic_image_topics: tuple[str, ...] = ("left", "center", "right")
    observation_horizon: int = 1
    proprio_keys: tuple[str, ...] = (
        "tcp_pose",
        "tcp_vel",
        "tcp_error",
        "joint_positions",
        "joint_velocities",
        "joint_efforts",
        "wrist_force",
        "wrist_torque",
    )


@dataclass(frozen=True)
class HilSerlControlConfig:
    """把 actor 输出映射成 AIC 速度控制命令时使用的参数。"""

    control_period_sec: float = 0.10
    action_scale_linear: float = 0.01
    action_scale_angular: float = 0.06
    max_linear_speed: float = 0.02
    max_angular_speed: float = 0.20
    linear_deadband: float = 1e-4
    angular_deadband: float = 1e-4


@dataclass(frozen=True)
class HilSerlSafetyConfig:
    """deep-insert 阶段的保守安全参数。"""

    max_runtime_sec: float = 30.0
    max_abs_force_z: float = 20.0
    max_abs_force_xy: float = 20.0
    max_abs_torque_xyz: float = 4.0
    max_consecutive_missing_obs: int = 10


@dataclass(frozen=True)
class HilSerlRuntimeConfig:
    """AIC 侧 HIL-SERL actor 推理的总配置。"""

    model: HilSerlModelConfig = field(default_factory=HilSerlModelConfig)
    observation: HilSerlObservationConfig = field(default_factory=HilSerlObservationConfig)
    control: HilSerlControlConfig = field(default_factory=HilSerlControlConfig)
    safety: HilSerlSafetyConfig = field(default_factory=HilSerlSafetyConfig)
    topics: HilSerlTopicConfig = field(default_factory=HilSerlTopicConfig)

    @property
    def workspace_root(self) -> Path:
        return Path("/home/young/ws_aic")

    @property
    def hil_serl_root(self) -> Path:
        return self.workspace_root / "hil-serl"
