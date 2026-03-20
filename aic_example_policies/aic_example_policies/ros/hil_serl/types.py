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

"""AIC 与 HIL-SERL 桥接层使用的共享类型。

这个模块的目标是把以下几层边界显式定义清楚：
- 来自 AIC 的原始 ROS 消息
- 送入 SERL 模型前的观测结构
- 模型输出的原始动作
- 最终可下发给控制器的笛卡尔命令

这些类型应尽量保持轻量、清晰，并在可能时便于序列化。
它们主要用于提升集成边界的可读性，并让后续实现更容易单测。

后续可能包含的类型：
- `HilSerlObservationBatch`：已归一化或半归一化的模型输入
- `HilSerlAction`：机器人相关缩放前的模型输出
- `CartesianVelocityCommand`：线速度、角速度及附带元信息
- `DeepInsertStatus`：用于日志和反馈的运行期状态快照
"""

# TODO: 等上游 HIL-SERL task wrapper 的 observation / action 结构确认后，
# 在这里补充 dataclass 或 TypedDict。
