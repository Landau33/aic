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

"""AIC 与 HIL-SERL 推理桥接层的共享类型。"""

from dataclasses import dataclass
from typing import Any

import numpy as np


ObservationDict = dict[str, np.ndarray]


@dataclass
class HilSerlAction:
    """模型输出的原始动作。"""

    values: np.ndarray


@dataclass
class CartesianVelocityCommand:
    """下发给 AIC 控制器之前的笛卡尔速度命令。"""

    linear_xyz: np.ndarray
    angular_xyz: np.ndarray


@dataclass
class DeepInsertStatus:
    """deep-insert 控制循环中的状态快照。"""

    step_index: int
    elapsed_sec: float
    waiting_handoff: bool = False
    missing_observation_count: int = 0
    latest_feedback: str = ""
    extra: dict[str, Any] | None = None
