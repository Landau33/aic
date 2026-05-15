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
import importlib.util
from pathlib import Path


def _load_actor_task_config_module():
    module_path = Path(
        "/home/yuang/ws_aic/hil-serl_aic/examples/experiments/aic_cable_insertion/config.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hil_serl_actor_task_config",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


try:
    _ACTOR_TASK_CONFIG_MODULE = _load_actor_task_config_module()
    _ACTOR_ENV_CONFIG = _ACTOR_TASK_CONFIG_MODULE.EnvConfig()
    _ACTOR_TRAIN_CONFIG = _ACTOR_TASK_CONFIG_MODULE.TrainConfig()
except Exception:
    _ACTOR_TASK_CONFIG_MODULE = None
    _ACTOR_ENV_CONFIG = None
    _ACTOR_TRAIN_CONFIG = None


@dataclass(frozen=True)
class HilSerlTopicConfig:
    """与阶段切换和可选外部控制相关的 topic 配置。"""

    deep_insert_topic: str = "/aic/deep_insert"
    observation_roi_topic: str = "observations_roi"
    left_image_roi_topic: str = "observations_roi/left_image"
    center_image_roi_topic: str = "observations_roi/center_image"
    right_image_roi_topic: str = "observations_roi/right_image"
    observation_masked_roi_topic: str = "observations_masked_roi"
    left_image_masked_roi_topic: str = "observations_masked_roi/left_image"
    center_image_masked_roi_topic: str = "observations_masked_roi/center_image"
    right_image_masked_roi_topic: str = "observations_masked_roi/right_image"


@dataclass(frozen=True)
class HilSerlModelConfig:
    """HIL-SERL actor 模型相关配置。"""

    exp_name: str = "aic_cable_insertion"
    setup_mode: str = (
        _ACTOR_TRAIN_CONFIG.setup_mode
        if _ACTOR_TRAIN_CONFIG is not None
        else "single-arm-fixed-gripper"
    )
    checkpoint_path: str = (
        "/home/yuang/ws_aic/hil-serl_aic/examples/experiments/aic_cable_insertion/checkpoints_test1"
    )
    checkpoint_step: int = 100000
    seed: int = 42
    argmax: bool = False
    encoder_type: str = (
        _ACTOR_TRAIN_CONFIG.encoder_type
        if _ACTOR_TRAIN_CONFIG is not None
        else "resnet-pretrained"
    )
    action_dim: int = 6
    image_keys: tuple[str, ...] = (
        tuple(_ACTOR_TRAIN_CONFIG.image_keys)
        if _ACTOR_TRAIN_CONFIG is not None
        else ("left_camera", "center_camera", "right_camera")
    )


@dataclass(frozen=True)
class HilSerlCameraRoiConfig:
    """单路相机 ROI 的手动裁剪参数。"""

    width: int = 800
    height: int = 800
    offset_x: float = 0.0
    offset_y: float = 0.0


@dataclass(frozen=True)
class HilSerlObservationConfig:
    """把 AIC Observation 对齐到 HIL-SERL 输入时需要的配置。"""

    image_width: int = 800
    image_height: int = 800
    roi_target_frame: str = "gripper/tcp"
    roi_target_offset_xyz: tuple[float, float, float] = (0.0, 0.015385, 0.04045)
    image_keys: tuple[str, ...] = (
        _ACTOR_ENV_CONFIG.image_keys
        if _ACTOR_ENV_CONFIG is not None
        else ("left_camera", "center_camera", "right_camera")
    )
    aic_image_topics: tuple[str, ...] = ("left", "center", "right")
    left_camera_roi: HilSerlCameraRoiConfig = field(
        default_factory=lambda: HilSerlCameraRoiConfig(
            width=300,
            height=300,
            offset_x=100.0,
            offset_y=-100.0,
        )
    )
    center_camera_roi: HilSerlCameraRoiConfig = field(
        default_factory=lambda: HilSerlCameraRoiConfig(
            width=200,
            height=400,
            offset_x=0.0,
            offset_y=-180.0,
        )
    )
    right_camera_roi: HilSerlCameraRoiConfig = field(
        default_factory=lambda: HilSerlCameraRoiConfig(
            width=300,
            height=300,
            offset_x=-100.0,
            offset_y=-100.0,
        )
    )
    observation_horizon: int = 1
    proprio_keys: tuple[str, ...] = (
        _ACTOR_ENV_CONFIG.proprio_keys
        if _ACTOR_ENV_CONFIG is not None
        else (
            "tcp_pose",
            "tcp_vel",
            "tcp_error",
            "joint_positions",
            "joint_velocities",
            "joint_efforts",
            "wrist_force",
            "wrist_torque",
        )
    )


@dataclass(frozen=True)
class HilSerlControlConfig:
    """把 actor 输出映射成 AIC 速度控制命令时使用的参数。"""

    control_period_sec: float = (
        _ACTOR_ENV_CONFIG.policy_control_period_sec
        if _ACTOR_ENV_CONFIG is not None
        else 0.10
    )
    action_scale_linear: float = (
        _ACTOR_ENV_CONFIG.action_scale_linear if _ACTOR_ENV_CONFIG is not None else 0.01
    )
    action_scale_angular: float = (
        _ACTOR_ENV_CONFIG.action_scale_angular
        if _ACTOR_ENV_CONFIG is not None
        else 0.06
    )
    control_frame_id: str = (
        _ACTOR_ENV_CONFIG.control_frame_id
        if _ACTOR_ENV_CONFIG is not None
        else "base_link"
    )
    max_linear_speed: float = 0.02
    max_angular_speed: float = 0.20
    linear_deadband: float = 1e-4
    angular_deadband: float = 1e-4


@dataclass(frozen=True)
class HilSerlSafetyConfig:
    """deep-insert 阶段的保守安全参数。"""

    max_runtime_sec: float = 600.0
    max_abs_force_z: float = 200.0
    max_abs_force_xy: float = 200.0
    max_abs_torque_xyz: float = 40.0
    max_consecutive_missing_obs: int = 10


@dataclass(frozen=True)
class HilSerlRuntimeConfig:
    """AIC 侧 HIL-SERL actor 推理的总配置。"""

    model: HilSerlModelConfig = field(default_factory=HilSerlModelConfig)
    observation: HilSerlObservationConfig = field(
        default_factory=HilSerlObservationConfig
    )
    control: HilSerlControlConfig = field(default_factory=HilSerlControlConfig)
    safety: HilSerlSafetyConfig = field(default_factory=HilSerlSafetyConfig)
    topics: HilSerlTopicConfig = field(default_factory=HilSerlTopicConfig)

    @property
    def workspace_root(self) -> Path:
        return Path("/home/yuang/ws_aic")

    @property
    def hil_serl_root(self) -> Path:
        return self.workspace_root / "hil-serl_aic"
