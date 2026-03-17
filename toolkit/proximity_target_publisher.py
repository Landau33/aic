#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.parameter import Parameter

class ProximityTargetPublisher(Node):
    def __init__(self):
        super().__init__('proximity_target_publisher')
        
        # ======================== 参数配置 ========================
        self.declare_parameter('tcp_frame', 'gripper/tcp')                 # 或 'gripper_link' / 'tool0'
        self.declare_parameter('nic_port_frame', 'nic_port_rec')
        self.declare_parameter('publish_frame', 'proximity_target')
        self.declare_parameter('publish_rate', 30.0)               

        self.tcp_frame = self.get_parameter('tcp_frame').value
        self.nic_port_frame = self.get_parameter('nic_port_frame').value
        self.publish_frame = self.get_parameter('publish_frame').value
        rate = self.get_parameter('publish_rate').value
        
        # ======================== 硬编码的旋转矩阵 ========================
        # current 旋转矩阵（相对于某个参考系，例如 base_link）
        self.R_current = np.array([
            [ -0.998,  0.001, -0.056],
            [ -0.052,  0.353,  0.934],
            [  0.021,  0.935, -0.353]
        ], dtype=np.float32)  # ← 这里替换成你实际的 current 旋转矩阵

        # target 旋转矩阵（你希望对齐到的目标姿态）
        self.R_target = np.array([
            [ -1.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  1.0000],
            [ 0.0000,  1.0000,  0.0000]
        ], dtype=np.float32)  # ← 这里替换成你实际的 target 旋转矩阵

        # 计算 R = R_target^{-1} * R_current
        # R_target_inv = np.linalg.inv(self.R_target)
        # self.R = R_target_inv @ self.R_current
        # R_current^{-1} ⋅ R_target
        R_current_inv = np.linalg.inv(self.R_current)
        self.R = R_current_inv @ self.R_target

        # 转换为四元数（提前算好，避免每次重复计算）
        rot = R.from_matrix(self.R)
        self.quat = rot.as_quat()  # [x, y, z, w]

        self.get_logger().info("已加载硬编码旋转矩阵，并计算出相对旋转 R")
        self.get_logger().info(f"R (matrix):\n{self.R}")
        self.get_logger().info(f"Quaternion (x,y,z,w): {self.quat}")

        # ======================== TF2 初始化 ========================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 定时发布
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

    def timer_callback(self):
        # 1. 获取 nic_port → tcp 的平移向量 T = [tx, ty, tz]
        try:
            trans = self.tf_buffer.lookup_transform(
                self.tcp_frame,           # target frame
                self.nic_port_frame,      # source frame
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
            tx = trans.transform.translation.x
            ty = trans.transform.translation.y
            tz = trans.transform.translation.z   # 虽然你只用 tx,ty，但这里先取全的
        except Exception as e:
            self.get_logger().warn(f"无法获取 {self.nic_port_frame} → {self.tcp_frame} 的变换: {str(e)}")
            return

        # 2. 构建新的 TF: tcp → proximity_target
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.tcp_frame
        t.child_frame_id = self.publish_frame

        # 先平移 [tx, ty, 0]（Z 保持不变）
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = 0.0

        # 然后应用旋转 R（使用预计算的四元数）
        t.transform.rotation.x = float(self.quat[0])
        t.transform.rotation.y = float(self.quat[1])
        t.transform.rotation.z = float(self.quat[2])
        t.transform.rotation.w = float(self.quat[3])

        # 发布
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = ProximityTargetPublisher()
    node.set_parameters([
        Parameter(
            'use_sim_time',
            Parameter.Type.BOOL,
            True
        )
    ])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()