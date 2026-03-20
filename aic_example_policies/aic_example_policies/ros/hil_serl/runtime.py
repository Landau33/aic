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

"""在 AIC 中执行训练后 HIL-SERL 策略的运行时封装。

这个模块应只包装 deep-insert 推理阶段所需的最小上游 HIL-SERL 能力。

职责：
- 初始化所选模型类型
- 加载 checkpoint 和可能存在的归一化统计量
- 提供 `reset()`，用于 episode 或 task 边界
- 提供 `predict()`，用于单步控制推理
- 在动作交给 adapter 之前完成模型侧后处理

这个运行时最好不要依赖 ROS，这样可以独立测试。
`TestPolicy` 应该把它当成纯推理组件使用。

预期参考的上游文件：
- `hil-serl/serl_launcher/serl_launcher/agents/continuous/bc.py`
- `hil-serl/serl_launcher/serl_launcher/agents/continuous/sac.py`
- `hil-serl/serl_launcher/serl_launcher/agents/continuous/sac_hybrid_single.py`
- `hil-serl/serl_launcher/serl_launcher/networks/actor_critic_nets.py`
"""

# TODO: 后续实现运行时包装层，负责：
# - 选择正确的上游 agent 类型
# - 从 checkpoint 恢复模型参数
# - 向 `TestPolicy` 暴露稳定的推理接口
# - 对 ROS 层隐藏 JAX 或具体框架细节
