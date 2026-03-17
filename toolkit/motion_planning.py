#!/usr/bin/env python3

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

"""
This script generates an interpolated Cartesian trajectory from the current gripper/tcp pose
to the proximity_target frame, and publishes position commands at 25 Hz using MotionUpdate
in MODE_POSITION.

新增标志位：
- use_static_target (默认 True)：
  True  → 只在开始时获取一次 proximity_target 位姿，全程插值到这个固定目标
  False → 每周期重新获取 proximity_target 的最新位姿（动态跟踪）

修改：
- 到达 proximity_target 后，暂停 1s
- 然后切换到速度模式（MODE_VELOCITY），保持 XY 不变，Z 负方向移动（向下压）
- 监测 Z 力，如果 |force.z| > 15N，停止速度发布（Twist 全 0）

Usage:
  ros2 run your_package aic_trajectory_to_proximity \
    --ros-args \
    -p duration_sec:=8.0 \
    -p use_static_target:=true \
    -p frame_id:="gripper/tcp" \
    -p z_velocity:=-0.05  # Z 负速度（m/s）
"""

import copy
import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_vector3
from geometry_msgs.msg import Pose, Twist, Wrench, WrenchStamped, Vector3, Quaternion, Vector3Stamped
from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
    TargetMode,
    ControllerState
)


class AICCartesianTrajectoryNode(Node):
    def __init__(self):
        super().__init__("aic_trajectory_to_proximity")

        # ======================== 参数 ========================
        self.declare_parameter("duration_sec", 8.0)          # 总插值时间（秒）
        self.declare_parameter("publish_rate", 25.0)         # 发布频率 Hz
        self.declare_parameter("frame_id", "base_link")    # 参考帧：gripper/tcp 或 base_link
        self.declare_parameter("controller_namespace", "aic_controller")
        self.declare_parameter("use_static_target", True)    # 是否使用静态目标位姿
        self.declare_parameter("z_velocity", -0.02)          # Z 负方向速度（m/s）
        self.declare_parameter("force_threshold", 10.0)      # Z 力阈值（N）
        # ======================== 螺旋搜索参数 ========================
        self.declare_parameter("spiral_radial_speed", 0.0015)  # 半径增加速度 (m/s)
        self.declare_parameter("spiral_omega", 3.0)            # 旋转角速度 (rad/s)
        self.declare_parameter("spiral_max_time", 20.0)        # 最大搜索时间 (s)
        self.declare_parameter("spiral_z_velocity", -0.005)    # 搜索时Z轴下压速度，保持贴合

        self.duration_sec = self.get_parameter("duration_sec").value
        self.publish_rate = self.get_parameter("publish_rate").value
        self.frame_id = self.get_parameter("frame_id").value
        self.controller_ns = self.get_parameter("controller_namespace").value
        self.use_static_target = self.get_parameter("use_static_target").value
        self.z_velocity = self.get_parameter("z_velocity").value
        self.force_threshold = self.get_parameter("force_threshold").value
        self.spiral_radial_speed = self.get_parameter("spiral_radial_speed").value
        self.spiral_omega = self.get_parameter("spiral_omega").value
        self.spiral_max_time = self.get_parameter("spiral_max_time").value
        self.spiral_z_velocity = self.get_parameter("spiral_z_velocity").value
        
        self.get_logger().info(f"Trajectory duration: {self.duration_sec} s")
        self.get_logger().info(f"Publish rate: {self.publish_rate} Hz")
        self.get_logger().info(f"Target frame mode: {self.frame_id}")
        self.get_logger().info(f"Use static target: {self.use_static_target} "
                              f"(True=固定初始目标, False=动态跟踪)")
        self.get_logger().info(f"Z velocity: {self.z_velocity} m/s")
        self.get_logger().info(f"Force threshold: {self.force_threshold} N")

        # ======================== 发布器 ========================
        self.motion_pub = self.create_publisher(
            MotionUpdate,
            f"/{self.controller_ns}/pose_commands",
            10
        )

        # 等待有订阅者
        while self.motion_pub.get_subscription_count() == 0:
            self.get_logger().info("Waiting for subscriber to pose_commands...")
            rclpy.sleep(Duration(seconds=1.0))

        # ======================== 力传感器订阅 ========================
        self.init_wrench = None
        self.current_wrench = None
        self.current_tared_wrench = None
        self.current_tare_offset_z = 0.0          # 当前 Z 方向 tare offset
        self.state_sub = self.create_subscription(
            ControllerState,
            '/aic_controller/controller_state',
            self.controller_state_callback,
            10
        )
        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            '/fts_broadcaster/wrench',
            self.wrench_callback,
            10
        )

        # ======================== TF2 ========================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ======================== 轨迹状态 ========================
        self.start_pose = None          # 开始时的 TCP pose
        self.target_pose = None         # proximity_target 的位姿（静态或动态）
        self.start_time = None          # 开始插值的时间戳
        self.trajectory_active = False  # 是否正在执行位置轨迹
        self.velocity_active = False    # 是否正在执行速度模式
        self.pause_start = None         # 暂停开始时间戳
        self.spiral_search_active = False    # 是否正在执行Spiral Search
        self.spiral_start_time = None   # 记录螺旋搜索开始的时间

        # 25 Hz 定时器
        self.timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info("Node started. Waiting for tf transforms...")

    def wrench_callback(self, msg: WrenchStamped):
        self.current_wrench = msg
        msg.wrench.force.x
    def controller_state_callback(self, msg: ControllerState):
        if self.current_wrench is None:
            return
        
        self.current_tared_wrench = copy.deepcopy(self.current_wrench)
        fts_tare_offset = msg.fts_tare_offset.wrench
        self.current_tared_wrench.wrench.force.x -= fts_tare_offset.force.x
        self.current_tared_wrench.wrench.force.y -= fts_tare_offset.force.y
        self.current_tared_wrench.wrench.force.z -= fts_tare_offset.force.z
        self.current_tared_wrench.wrench.torque.x -= fts_tare_offset.torque.x
        self.current_tared_wrench.wrench.torque.y -= fts_tare_offset.torque.y
        self.current_tared_wrench.wrench.torque.z -= fts_tare_offset.torque.z

        # self.get_logger().info(f"Tared force z: {self.current_tared_wrench.wrench.force.z} N")

    def get_current_tcp_pose(self):
        """获取当前 gripper/tcp 的位姿（相对于 base_link）"""
        try:
            trans = self.tf_buffer.lookup_transform(
                "base_link", "gripper/tcp", rclpy.time.Time(), timeout=Duration(seconds=1.0)
            )
            p = trans.transform.translation
            q = trans.transform.rotation
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            pose.position.z = p.z
            pose.orientation = q
            return pose
        except Exception as e:
            self.get_logger().warn(f"Failed to get current tcp pose: {str(e)}")
            return None

    def get_target_pose(self):
        """获取 proximity_target 的位姿（相对于 base_link）"""
        try:
            trans = self.tf_buffer.lookup_transform(
                "base_link", "proximity_target", rclpy.time.Time(), timeout=Duration(seconds=0.5)
            )
            p = trans.transform.translation
            q = trans.transform.rotation
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            pose.position.z = p.z
            pose.orientation = q
            return pose
        except Exception as e:
            self.get_logger().warn(f"Failed to get proximity_target pose: {str(e)}")
            return None

    def start_trajectory(self):
        """开始一条新轨迹"""
        self.start_pose = self.get_current_tcp_pose()
        self.target_pose = self.get_target_pose()

        if self.start_pose is None or self.target_pose is None:
            self.get_logger().warn("Cannot start trajectory: missing poses")
            return False

        self.start_time = self.get_clock().now()
        self.trajectory_active = True
        self.velocity_active = False  # 重置速度模式

        self.get_logger().info("Trajectory started:")
        self.get_logger().info(f"  Start:  {self.start_pose.position.x:.3f}, {self.start_pose.position.y:.3f}, {self.start_pose.position.z:.3f}")
        self.get_logger().info(f"  Target: {self.target_pose.position.x:.3f}, {self.target_pose.position.y:.3f}, {self.target_pose.position.z:.3f}")
        self.get_logger().info(f"  Mode: {'Static' if self.use_static_target else 'Dynamic'}")

        return True

    def interpolate_pose(self, t):
        """
        t ∈ [0,1] 的插值进度
        返回当前目标位姿（Pose）
        """
        if not self.trajectory_active:
            return None

        # 如果是动态模式，每周期重新获取目标位姿
        if not self.use_static_target:
            current_target = self.get_target_pose()
            if current_target is not None:
                self.target_pose = current_target

        # 线性插值位置
        pos = np.array([
            self.start_pose.position.x + t * (self.target_pose.position.x - self.start_pose.position.x),
            self.start_pose.position.y + t * (self.target_pose.position.y - self.start_pose.position.y),
            self.start_pose.position.z + t * (self.target_pose.position.z - self.start_pose.position.z),
        ])

        # SLERP 插值姿态
        q_start = R.from_quat([
            self.start_pose.orientation.x,
            self.start_pose.orientation.y,
            self.start_pose.orientation.z,
            self.start_pose.orientation.w
        ])
        q_target = R.from_quat([
            self.target_pose.orientation.x,
            self.target_pose.orientation.y,
            self.target_pose.orientation.z,
            self.target_pose.orientation.w
        ])
        slerp = Slerp([0, 1], R.concatenate([q_start, q_target]))
        q_interp = slerp(t)

        quat = q_interp.as_quat()  # [x,y,z,w]

        pose = Pose()
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        return pose

    def generate_position_update(self, pose,
                                 stiffness_diag=[800.0] * 6,
                                 damping_diag=[60.0] * 6,
                                 feedforward_wrench_at_tip=[0.0] * 6,
                                 wrench_feedback_gains_at_tip=[0.0] * 6
                                 ):
        """生成位置模式的 MotionUpdate 消息"""
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.pose = pose

        # 推荐的刚度与阻尼（可调）
        msg.target_stiffness = np.diag(stiffness_diag).flatten().tolist()
        msg.target_damping   = np.diag(damping_diag).flatten().tolist()

        # 无前馈力
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=feedforward_wrench_at_tip[0], y=feedforward_wrench_at_tip[1], z=feedforward_wrench_at_tip[2]),
            torque=Vector3(x=feedforward_wrench_at_tip[3], y=feedforward_wrench_at_tip[4], z=feedforward_wrench_at_tip[5])
        )
        msg.wrench_feedback_gains_at_tip = wrench_feedback_gains_at_tip

        # 位置模式
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION

        return msg

    def generate_velocity_update(self, twist,
                                 stiffness_diag=[800.0] * 6,
                                 damping_diag=[60.0] * 6,
                                 feedforward_wrench_at_tip=[0.0] * 6,
                                 wrench_feedback_gains_at_tip=[0.0] * 6
                                 ):
        """生成速度模式的 MotionUpdate 消息"""
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.velocity = twist

        # 速度模式下降低刚度（更柔顺）
        msg.target_stiffness = np.diag(stiffness_diag).flatten().tolist()
        msg.target_damping   = np.diag(damping_diag).flatten().tolist()

        # 可选前馈力（向下压）
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=feedforward_wrench_at_tip[0], y=feedforward_wrench_at_tip[1], z=feedforward_wrench_at_tip[2]),
            torque=Vector3(x=feedforward_wrench_at_tip[3], y=feedforward_wrench_at_tip[4], z=feedforward_wrench_at_tip[5])
        )
        msg.wrench_feedback_gains_at_tip = wrench_feedback_gains_at_tip

        # 速度模式
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

        return msg

    def timer_callback(self):
        # 如果还没开始轨迹，尝试启动
        if not self.trajectory_active and not self.velocity_active and not hasattr(self, 'trajectory_started'):
            if self.start_trajectory():
                self.trajectory_started = True  # 永久标记已启动
                return

        # ======================== 位置轨迹阶段 ========================
        if self.trajectory_active:
            # 计算当前进度 t ∈ [0,1]
            now = self.get_clock().now()
            elapsed = (now - self.start_time).nanoseconds / 1e9
            t = min(elapsed / self.duration_sec, 1.0)

            # 插值得到当前目标位姿
            target_pose = self.interpolate_pose(t)

            if target_pose is None:
                return

            # 发布位置指令
            msg = self.generate_position_update(target_pose)
            self.motion_pub.publish(msg)

            # 到达终点后，进入暂停阶段
            if t >= 1.0:
                self.get_logger().info("Trajectory completed! Pausing for 1s...")
                self.trajectory_active = False
                self.pause_start = self.get_clock().now()

        # ======================== 暂停阶段 ========================
        elif self.pause_start is not None:
            now = self.get_clock().now()
            pause_elapsed = (now - self.pause_start).nanoseconds / 1e9
            if pause_elapsed < 5.0:
                # 暂停中，继续发布最后位姿（保持位置）
                msg = self.generate_position_update(self.target_pose)
                self.motion_pub.publish(msg)
            else:
                self.get_logger().info("Pause complete. Entering velocity mode (Z negative)...")
                self.pause_start = None
                self.velocity_active = True

        # ======================== 速度模式阶段 ========================
        elif self.velocity_active:
            # 检查 Z 力是否超过阈值
            if abs(self.current_tare_offset_z) > self.force_threshold:
                self.get_logger().info(f"Surface detected! Z force {self.current_tare_offset_z:.2f} N > {self.force_threshold} N.")
                
                # 状态机切换：停止向下移动，启动螺旋搜索
                self.velocity_active = False
                self.spiral_search_active = True
                self.spiral_start_time = self.get_clock().now()
                self.get_logger().info("Starting Spiral Search for hole...")
            else:
                twist = Twist()
                twist.linear.z = self.z_velocity  # 只 Z 负方向

                # 发布速度指令
                msg = self.generate_velocity_update(twist,
                                                    stiffness_diag=[800.0, 800.0, 80.0, 800.0, 800.0, 800.0],
                                                    damping_diag=[75.0, 75.0, 50.0, 75.0, 75.0, 75.0])
                self.motion_pub.publish(msg)
        
        # ======================== Spiral Search阶段 ========================
        elif self.spiral_search_active:
            now = self.get_clock().now()
            t = (now - self.spiral_start_time).nanoseconds / 1e9

            # 异常处理：检查是否超时
            if t > self.spiral_max_time:
                self.get_logger().warn("Spiral search timeout! Hole not found. Stopping.")
                self.spiral_search_active = False
                msg = self.generate_velocity_update(Twist()) # 全0停止
                self.motion_pub.publish(msg)
                return

            # 成功检测：当Peg滑入孔中时，向上的支撑力会消失，Z轴阻力会急剧下降
            # 使用阈值的 30% 作为判断标准 (可以根据实际摩擦力微调)
            if t > 0.5 and abs(self.current_tare_offset_z) < 5:         # (self.force_threshold * 0.3)
                self.get_logger().info(f"Hole found! Z Force dropped to {self.current_tare_offset_z:.2f} N. Stopping spiral.")
                self.spiral_search_active = False
                
                # 停止平面运动，可以选择在这里转入深插逻辑 (Deep Insertion)
                msg = self.generate_velocity_update(Twist()) 
                self.motion_pub.publish(msg)
                return

            # 计算阿基米德螺旋线速度
            c = self.spiral_radial_speed
            w = self.spiral_omega

            vx = c * math.cos(w * t) - c * t * w * math.sin(w * t)
            vy = c * math.sin(w * t) + c * t * w * math.cos(w * t)

            twist = Twist()
            twist.linear.x = vx
            twist.linear.y = vy
            twist.linear.z = self.spiral_z_velocity  # 保持微弱向下压迫，以保证不脱离表面并在遇到孔洞时自动掉入

            # 发布速度指令：注意我们要降低 XY 刚度，增加柔顺性，防止在滑动过程中卡死
            msg = self.generate_velocity_update(twist,
                                                stiffness_diag=[200.0, 200.0, 50.0, 800.0, 800.0, 800.0],
                                                damping_diag=[30.0, 30.0, 50.0, 75.0, 75.0, 75.0],
                                                feedforward_wrench_at_tip=[0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
            self.motion_pub.publish(msg)
        
        # ======================== 通用力控处理 ==============================
        if self.current_tared_wrench is None:
            return
        if self.init_wrench is None:
            self.init_wrench = copy.deepcopy(self.current_tared_wrench)
            return
        
        try:
            force_stamped = Vector3Stamped()
            force_stamped.header = self.current_tared_wrench.header        # frame_id = ati/tool_link
            force_stamped.vector = self.current_tared_wrench.wrench.force
            force_stamped.vector.x -= self.init_wrench.wrench.force.x
            force_stamped.vector.y -= self.init_wrench.wrench.force.y
            force_stamped.vector.z -= self.init_wrench.wrench.force.z
            # 获取当前 ati/tool_link → base_link 的变换
            transform = self.tf_buffer.lookup_transform(
                "base_link", "ati/tool_link",
                rclpy.time.Time(), timeout=Duration(seconds=0.1)
            )

            # 执行向量旋转转换
            transformed = do_transform_vector3(force_stamped, transform)

            # 提取世界 Z 分量
            self.current_tare_offset_z = transformed.vector.z
            self.get_logger().info(f"Force z: {self.current_tare_offset_z} N")

        except Exception as e:
            self.get_logger().warn(f"Force transform failed: {str(e)}")
            self.current_tare_offset_z = self.current_tared_wrench.wrench.force.z - self.init_wrench.wrench.force.z

def main(args=None):
    rclpy.init(args=args)
    node = AICCartesianTrajectoryNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()