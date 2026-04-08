import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import tf2_ros
import message_filters
from rclpy.parameter import Parameter

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation
from ultralytics import YOLO


class NICPortEstimator(Node):
    def __init__(self):
        super().__init__('nic_port_estimator')
        self.set_parameters([
            Parameter(
                'use_sim_time',
                Parameter.Type.BOOL,
                True
            )
        ])

        # 内参
        fx = 1236.63171387
        fy = 1236.63146973
        cx = 576
        cy = 512
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]], dtype=np.float32)

        self.D = np.zeros(5, dtype=np.float32)

        # 外参（camera → world）
        self.T_left_to_world = np.array([[0.500,  0.837,  0.224, -0.101],
                                         [-0.837,  0.533, -0.125,  0.056],
                                         [-0.224, -0.125,  0.967,  0.015],
                                         [0,      0,      0,      1]], dtype=np.float32)

        self.T_center_to_world = np.eye(4, dtype=np.float32)

        self.T_right_to_world = np.array([[0.500, -0.837, -0.224,  0.101],
                                          [0.837,  0.533, -0.125,  0.056],
                                          [0.224, -0.125,  0.967,  0.015],
                                          [0,      0,      0,      1]], dtype=np.float32)

        # YOLO 模型路径（请替换为你的实际路径）
        self.model = YOLO('./best.pt')

        # CvBridge
        self.bridge = CvBridge()

        # 图像发布器：发布带 OBB 标注的中心相机图像
        self.recognized_pub = self.create_publisher(Image, '/center_camera/image/recognized', 10)

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 同步订阅三个图像话题
        left_sub = message_filters.Subscriber(self, Image, '/left_camera/image')
        center_sub = message_filters.Subscriber(self, Image, '/center_camera/image')
        right_sub = message_filters.Subscriber(self, Image, '/right_camera/image')

        ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, center_sub, right_sub],
            queue_size=10,
            slop=0.1
        )
        ts.registerCallback(self.image_callback)

        self.get_logger().info("NIC Port Estimator started. Publishing recognized image to /center_camera/image/recognized")

    def detect_obb_and_select_zero(self, img, view_type):
        """检测 OBB 并根据视图类型选择 0 号点，返回角点和可视化图像"""
        results = self.model(img, verbose=False)
        if len(results) == 0 or results[0].obb is None or len(results[0].obb) == 0:
            self.get_logger().debug(f"No detection in {view_type}")
            return None, img.copy()

        obbs = results[0].obb
        centers_x = obbs.xywhr[:, 0].cpu().numpy()
        idx = np.argmin(centers_x)  # 选择最靠左的检测框
        corners = obbs[idx].xyxyxyxy[0].cpu().numpy()  # (4,2)

        # 根据视图类型决定 0 号点
        if view_type == 'left':
            zero_idx = 1      # 右上
        elif view_type == 'center':
            zero_idx = 2      # 左上（根据你之前的逻辑）
        elif view_type == 'right':
            zero_idx = 3      # 左下
        else:
            raise ValueError("Invalid view_type")

        sorted_corners = np.roll(corners, -zero_idx, axis=0)

        # 生成可视化图像（只在 center 视图真正发布时使用）
        vis_img = img.copy()
        zero_pt = tuple(map(int, sorted_corners[0]))
        cv2.circle(vis_img, zero_pt, 8, (0, 0, 255), -1)  # 红色实心圆
        cv2.putText(vis_img, "0", (zero_pt[0]+12, zero_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 可选：绘制所有四个角点（调试用）
        for i, pt in enumerate(sorted_corners):
            p = tuple(map(int, pt))
            cv2.circle(vis_img, p, 5, (0, 255, 0), 2)  # 绿色小圆
            cv2.putText(vis_img, str(i), (p[0]+8, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        return sorted_corners, vis_img

    def build_projection_matrix(self, T_cam_to_world):
        R_cam2world = T_cam_to_world[:3, :3]
        t_cam2world = T_cam_to_world[:3, 3]
        R_w2c = R_cam2world.T
        t_w2c = -R_cam2world.T @ t_cam2world
        Rt = np.hstack((R_w2c, t_w2c.reshape(3, 1)))
        P = self.K @ Rt
        return P

    def triangulate_multiview(self, corners_2d_list, P_list):
        num_views = len(P_list)
        A = []
        for i in range(num_views):
            x, y = corners_2d_list[i]
            P = P_list[i]
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
        A = np.array(A, dtype=np.float32)

        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1]
        X = X / X[3] if X[3] != 0 else X
        return X[:3]

    def image_callback(self, left_msg, center_msg, right_msg):
        stamp = center_msg.header.stamp
        try:
            left_img   = self.bridge.imgmsg_to_cv2(left_msg,   "bgr8")
            center_img = self.bridge.imgmsg_to_cv2(center_msg, "bgr8")
            right_img  = self.bridge.imgmsg_to_cv2(right_msg,  "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {str(e)}")
            return

        views = ['left', 'center', 'right']
        imgs = [left_img, center_img, right_img]
        T_list = [self.T_left_to_world, self.T_center_to_world, self.T_right_to_world]

        P_list = [self.build_projection_matrix(T) for T in T_list]

        corners_all = []
        valid_indices = []

        center_vis_img = None  # 用于发布带标注的中心图像

        for i, (view, img) in enumerate(zip(views, imgs)):
            corners, vis_img = self.detect_obb_and_select_zero(img, view)
            if corners is not None:
                corners_all.append(corners)
                valid_indices.append(i)
                if view == 'center':
                    center_vis_img = vis_img  # 保存中心相机的标注图像

        if len(corners_all) < 2:
            self.get_logger().debug("At least 2 views needed for triangulation")
            # 即使没有足够视图，也发布原始中心图像（可选）
            if center_img is not None:
                recognized_msg = self.bridge.cv2_to_imgmsg(center_img, "bgr8")
                recognized_msg.header = center_msg.header
                self.recognized_pub.publish(recognized_msg)
            return

        # =================== 计算平移部分 =====================================
        # 三角化 4 个角点
        points_3d = []
        for pt_idx in range(4):
            points_2d_this = [corners_all[j][pt_idx] for j in range(len(corners_all))]
            P_this = [P_list[valid_indices[j]] for j in range(len(corners_all))]
            world_pt = self.triangulate_multiview(points_2d_this, P_this)
            points_3d.append(world_pt)

        points_3d = np.array(points_3d)
        port_center_world = np.mean(points_3d, axis=0)

        self.get_logger().info(f"NIC Port center (world): {port_center_world}")
        
        p0, p1, p2, p3 = points_3d[0], points_3d[1], points_3d[2], points_3d[3]

        # =================== 计算旋转部分 =====================================
        # 1. 估算 X 轴 (向右) 和 Y 轴 (向下) 的粗略向量
        # X 轴 (向右): 从左往右，即从 0 指向 3，以及从 1 指向 2
        v_x_raw = ((p3 - p0) + (p2 - p1)) / 2.0
        
        # Y 轴 (向下): 从上往下，即从 0 指向 1，以及从 3 指向 2
        v_y_raw = ((p1 - p0) + (p2 - p3)) / 2.0

        # 2. 计算 Z 轴 (向外凸出)。右手定则：Y(向下) 叉乘 X(向右) = Z(向外)
        v_z = np.cross(v_x_raw, v_y_raw)
        
        # 3. 归一化并正交化，构建标准旋转矩阵
        z_axis = v_z / np.linalg.norm(v_z)
        x_axis = v_x_raw / np.linalg.norm(v_x_raw)
        y_axis = np.cross(z_axis, x_axis) # 重新计算 Y 确保三者绝对正交
        # 组装为 3x3 的旋转矩阵 (世界坐标系下网口到世界的旋转)
        R_port_to_world = np.column_stack((x_axis, y_axis, z_axis))

        # ==========================================
        # 转换到 center_camera_optical 坐标系
        # ==========================================
        T_world_to_center = np.linalg.inv(self.T_center_to_world)
        
        # 转换平移 (Translation)
        port_hom = np.append(port_center_world, 1)
        port_in_center = (T_world_to_center @ port_hom)[:3]

        # 转换旋转 (Rotation): R_port_in_center = R_world_to_center * R_port_to_world
        R_world_to_center = T_world_to_center[:3, :3]
        R_port_in_center = R_world_to_center @ R_port_to_world

        # 将旋转矩阵转换为四元数 [x, y, z, w]
        quat = SciPyRotation.from_matrix(R_port_in_center).as_quat()

        # 发布 TF
        transform = TransformStamped()
        transform.header.stamp = stamp 
        transform.header.frame_id = "center_camera/optical"
        transform.child_frame_id = "nic_port_rec"

        # 写入平移
        transform.transform.translation.x = float(port_in_center[0])
        transform.transform.translation.y = float(port_in_center[1])
        transform.transform.translation.z = float(port_in_center[2])

        # 写入旋转
        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(transform)

        # 发布带 OBB 标注的中心图像（即使三角化失败也尽量发布）
        if center_vis_img is not None:
            recognized_msg = self.bridge.cv2_to_imgmsg(center_vis_img, "bgr8")
        else:
            recognized_msg = self.bridge.cv2_to_imgmsg(center_img, "bgr8")  # fallback

        recognized_msg.header = center_msg.header
        self.recognized_pub.publish(recognized_msg)

def main(args=None):
    rclpy.init(args=args)
    node = NICPortEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
