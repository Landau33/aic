HIL-SERL AIC 集成骨架

这个目录用于放置一个面向部署的桥接层，把训练好的 HIL-SERL 策略接入
AIC 的 `TestPolicy` deep-insert 阶段。

设计目标：
- 保持 `motion_planning.py` 不变
- 保留 `/aic/deep_insert` 作为阶段切换信号
- 保持 `aic_model` 作为机器人指令发布的唯一出口
- 避免把上游训练基础设施直接复制进 AIC policy 包

规划中的数据流：
1. `motion_planning.py` 发布 `deep_insert=true`
2. `TestPolicy.insert_cable()` 进入 deep-insert 控制阶段
3. `adapter.py` 将 AIC `Observation` 转成模型输入
4. `runtime.py` 执行 HIL-SERL 推理
5. `adapter.py` 将模型输出转换成笛卡尔速度命令
6. `TestPolicy` 通过 `move_robot()` 发送 `MotionUpdate`

这个目录应只保留推理期代码。
