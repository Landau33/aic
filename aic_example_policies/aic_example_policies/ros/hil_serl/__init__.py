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

"""AIC 中 deep-insert 阶段的 HIL-SERL 部署骨架。

这个包只面向部署侧，保持尽量小，只保留把训练好的 SERL 策略接入
AIC `TestPolicy` 所需的最小组件。

建议的模块边界：
- `config.py`：运行期参数和安全参数
- `types.py`：模块间共享的轻量类型定义
- `adapter.py`：AIC Observation 与 SERL 观测/动作之间的转换
- `runtime.py`：模型加载、reset、推理与动作后处理

上游 HIL-SERL 中的训练脚本、replay buffer、learner 逻辑、机器人服务端
基础设施，不应直接复制到这里。
"""
