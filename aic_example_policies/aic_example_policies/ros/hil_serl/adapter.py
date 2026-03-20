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

"""AIC 与 HIL-SERL 之间的观测/动作适配层。

这个模块是迁移过程里的核心翻译层。

职责：
- 读取 `aic_model_interfaces.msg.Observation`
- 提取训练好的 HIL-SERL 策略真正需要的观测子集
- 复现原始 task wrapper 期望的 observation 布局
- 如果策略依赖时序上下文，维护短历史窗口
- 把模型输出转换成控制器可接受的笛卡尔速度命令
- 应用训练阶段约定的动作缩放规则

不应该放在这里的内容：
- ROS 控制循环本身
- 模型 checkpoint 加载
- policy 阶段切换状态机
- 依赖任务级上下文的安全停机决策

建议后续 API 形态：
- `HilSerlObservationAdapter`
- `HilSerlActionAdapter`
- 图像转换、状态向量拼接、历史窗口 reset 的辅助方法

预期参考的上游文件：
- `hil-serl/examples/experiments/ram_insertion/wrapper.py`
- `hil-serl/serl_launcher/serl_launcher/wrappers/serl_obs_wrappers.py`
"""

# TODO: 后续实现适配器类，负责：
# - 把 ROS 图像转成模型张量或数组
# - 拼接 tcp pose、速度、力觉和 joint state
# - 对齐上游 HIL-SERL 策略使用的 key 和 shape
# - 把 actor 输出转换成可映射到 Twist 的笛卡尔速度
