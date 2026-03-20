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

"""AIC 内运行 HIL-SERL 所需的配置定义。

这个模块后续建议只放小型 dataclass，用来统一管理部署期参数，
避免把常量分散在 `TestPolicy`、观测适配器和模型运行时里。

典型配置分类：
- 模型 checkpoint 路径与 checkpoint 格式
- 模型类型选择：BC / SAC / hybrid SAC
- 运行设备：CPU / CUDA
- 控制频率与观测历史长度
- 图像缩放和归一化参数
- 动作缩放、裁剪、平滑和死区
- 力/力矩安全阈值
- deep-insert 阶段进入/退出所用 topic

这个模块应保持为纯配置层，不要引入 ROS 副作用。
这里只定义配置结构和构造辅助函数。
"""

# TODO: 后续补充 dataclass，例如：
# - HilSerlModelConfig
# - HilSerlControlConfig
# - HilSerlSafetyConfig
# - HilSerlTopicConfig
# - HilSerlRuntimeConfig
