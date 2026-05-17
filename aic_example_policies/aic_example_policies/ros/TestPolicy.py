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

import copy
import time

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Twist, Vector3, Wrench
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import TransformException

from .hil_serl.adapter import HilSerlActionAdapter, HilSerlObservationAdapter
from .hil_serl.config import (
    HilSerlCameraRoiConfig,
    HilSerlRuntimeConfig,
)
from .hil_serl.runtime import HilSerlActorRuntime
from .masked_roi import MaskedRoiPublisher


class TestPolicy(Policy):
    """在 deep-insert 阶段执行 HIL-SERL actor 推理。"""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._config = HilSerlRuntimeConfig()
        self._observation_adapter = HilSerlObservationAdapter(self._config.observation)
        self._action_adapter = HilSerlActionAdapter(self._config.control)
        self._runtime = HilSerlActorRuntime(self._config)

        self._observation_roi_pub = parent_node.create_publisher(
            Observation, self._config.topics.observation_roi_topic, 10
        )
        self._left_image_roi_pub = parent_node.create_publisher(
            Image, self._config.topics.left_image_roi_topic, 10
        )
        self._center_image_roi_pub = parent_node.create_publisher(
            Image, self._config.topics.center_image_roi_topic, 10
        )
        self._right_image_roi_pub = parent_node.create_publisher(
            Image, self._config.topics.right_image_roi_topic, 10
        )
        self._masked_roi_pub = MaskedRoiPublisher(parent_node, self._config.topics)
        self._masked_roi_pub.start()
        self._observation_sub = parent_node.create_subscription(
            Observation,
            "observations",
            self._on_observation,
            10,
        )
        self._roi_publish_count = 0
        self._roi_projection_fallback_count = 0
        self._roi_target_offset_xyz = np.asarray(
            self._config.observation.roi_target_offset_xyz,
            dtype=np.float64,
        )
        self._last_background_roi_publish_time = 0.0
        self._background_roi_publish_period_sec = 0.1
        self._last_logged_roi_params = {}
        self._declare_live_roi_parameters()

        self._deep_insert = False
        self._deepinsert_event_sub = parent_node.create_subscription(
            String,
            self._config.topics.deep_insert_topic,
            self._on_deepinsert_event,
            10,
        )
        self.get_logger().info("TestPolicy.__init__()")

    def _declare_live_roi_parameters(self) -> None:
        for camera_name in ("left", "center", "right"):
            roi = self._camera_roi_config_default(camera_name)
            self._declare_parameter_if_needed(
                f"hil_serl.roi.{camera_name}.width",
                int(roi.width),
            )
            self._declare_parameter_if_needed(
                f"hil_serl.roi.{camera_name}.height",
                int(roi.height),
            )
            self._declare_parameter_if_needed(
                f"hil_serl.roi.{camera_name}.offset_x",
                float(roi.offset_x),
            )
            self._declare_parameter_if_needed(
                f"hil_serl.roi.{camera_name}.offset_y",
                float(roi.offset_y),
            )

    def _declare_parameter_if_needed(self, name: str, value) -> None:
        if not self._parent_node.has_parameter(name):
            self._parent_node.declare_parameter(name, value)

    def _on_deepinsert_event(self, msg: String) -> None:
        self._deep_insert = msg.data.strip().lower() == "true"
        self.get_logger().info(f"deep_insert={self._deep_insert}")

    def _on_observation(self, msg: Observation) -> None:
        now = time.monotonic()
        if now - self._last_background_roi_publish_time < (
            self._background_roi_publish_period_sec
        ):
            return
        self._last_background_roi_publish_time = now
        self._publish_observation_roi(msg)

    def _publish_observation_roi(self, msg: Observation) -> Observation | None:
        """Publish a cropped Observation and return the exact message published."""
        roi_msg = Observation()
        roi_msg.controller_state = msg.controller_state
        roi_msg.wrist_wrench = msg.wrist_wrench
        roi_msg.joint_states = msg.joint_states

        try:
            left_roi = self._camera_roi_config("left")
            roi_msg.left_image, roi_msg.left_camera_info = (
                self._crop_around_tcp_projection(
                    msg.left_image,
                    msg.left_camera_info,
                    left_roi.width,
                    left_roi.height,
                    left_roi.offset_x,
                    left_roi.offset_y,
                    "left",
                )
            )
            center_roi = self._camera_roi_config("center")
            roi_msg.center_image, roi_msg.center_camera_info = (
                self._crop_around_tcp_projection(
                    msg.center_image,
                    msg.center_camera_info,
                    center_roi.width,
                    center_roi.height,
                    center_roi.offset_x,
                    center_roi.offset_y,
                    "center",
                )
            )
            right_roi = self._camera_roi_config("right")
            roi_msg.right_image, roi_msg.right_camera_info = (
                self._crop_around_tcp_projection(
                    msg.right_image,
                    msg.right_camera_info,
                    right_roi.width,
                    right_roi.height,
                    right_roi.offset_x,
                    right_roi.offset_y,
                    "right",
                )
            )
        except Exception as exc:
            self.get_logger().warn(f"Failed to build observations_roi: {exc}")
            return None

        self._observation_roi_pub.publish(roi_msg)
        self._left_image_roi_pub.publish(roi_msg.left_image)
        self._center_image_roi_pub.publish(roi_msg.center_image)
        self._right_image_roi_pub.publish(roi_msg.right_image)
        self._masked_roi_pub.publish_observation(roi_msg)
        self._roi_publish_count += 1
        if self._roi_publish_count % 20 == 1:
            self.get_logger().info(
                f"Published {self._config.topics.observation_roi_topic} "
                f"left={roi_msg.left_image.width}x{roi_msg.left_image.height}, "
                f"center={roi_msg.center_image.width}x{roi_msg.center_image.height}, "
                f"right={roi_msg.right_image.width}x{roi_msg.right_image.height}"
            )
        return roi_msg

    def _camera_roi_config_default(self, camera_name: str):
        observation_config = self._config.observation
        if camera_name == "left":
            return observation_config.left_camera_roi
        if camera_name == "center":
            return observation_config.center_camera_roi
        if camera_name == "right":
            return observation_config.right_camera_roi
        raise ValueError(f"Unknown camera ROI config: {camera_name}")

    def _camera_roi_config(self, camera_name: str) -> HilSerlCameraRoiConfig:
        default = self._camera_roi_config_default(camera_name)
        width = int(
            self._parent_node.get_parameter(f"hil_serl.roi.{camera_name}.width").value
        )
        height = int(
            self._parent_node.get_parameter(f"hil_serl.roi.{camera_name}.height").value
        )
        offset_x = float(
            self._parent_node.get_parameter(
                f"hil_serl.roi.{camera_name}.offset_x"
            ).value
        )
        offset_y = float(
            self._parent_node.get_parameter(
                f"hil_serl.roi.{camera_name}.offset_y"
            ).value
        )
        roi = HilSerlCameraRoiConfig(
            width=max(1, width),
            height=max(1, height),
            offset_x=offset_x,
            offset_y=offset_y,
        )
        logged = (roi.width, roi.height, roi.offset_x, roi.offset_y)
        if self._last_logged_roi_params.get(camera_name) != logged:
            self._last_logged_roi_params[camera_name] = logged
            self.get_logger().info(
                f"Live ROI {camera_name}: "
                f"{roi.width}x{roi.height}, "
                f"offset=({roi.offset_x:.1f}, {roi.offset_y:.1f})"
            )
        return roi

    def _crop_center(
        self,
        image_msg: Image,
        camera_info_msg,
        crop_width: int,
        crop_height: int,
    ) -> tuple[Image, object]:
        """Crop the image center and adjust camera intrinsics for the ROI."""
        return self._crop_around_pixel(
            image_msg,
            camera_info_msg,
            crop_width,
            crop_height,
            center_x=None,
            center_y=None,
        )

    def _crop_around_tcp_projection(
        self,
        image_msg: Image,
        camera_info_msg,
        crop_width: int,
        crop_height: int,
        pixel_offset_x: float,
        pixel_offset_y: float,
        camera_name: str,
    ) -> tuple[Image, object]:
        """Crop around the projected TCP offset point, falling back to image center."""
        projected_center = self._project_roi_target_to_image(camera_info_msg)
        if projected_center is None:
            self._roi_projection_fallback_count += 1
            if self._roi_projection_fallback_count % 20 == 1:
                self.get_logger().warn(
                    "TCP-offset ROI projection unavailable for "
                    f"{camera_name}; falling back to center crop"
                )
            return self._crop_around_pixel(
                image_msg,
                camera_info_msg,
                crop_width,
                crop_height,
                center_x=pixel_offset_x if pixel_offset_x != 0.0 else None,
                center_y=pixel_offset_y if pixel_offset_y != 0.0 else None,
            )

        return self._crop_around_pixel(
            image_msg,
            camera_info_msg,
            crop_width,
            crop_height,
            center_x=projected_center[0] + float(pixel_offset_x),
            center_y=projected_center[1] + float(pixel_offset_y),
        )

    def _crop_around_pixel(
        self,
        image_msg: Image,
        camera_info_msg,
        crop_width: int,
        crop_height: int,
        center_x: float | None,
        center_y: float | None,
    ) -> tuple[Image, object]:
        """Crop around a pixel location and adjust camera intrinsics for the ROI."""
        if image_msg.height <= 0 or image_msg.width <= 0 or not image_msg.data:
            raise ValueError("empty image")

        image, output_encoding = self._image_msg_to_array(image_msg)
        height, width = image.shape[:2]

        bounded_width = min(crop_width, width)
        bounded_height = min(crop_height, height)
        if center_x is None or center_y is None:
            center_x = (width - 1) * 0.5
            center_y = (height - 1) * 0.5
        center_x = float(np.clip(center_x, 0.0, max(width - 1, 0)))
        center_y = float(np.clip(center_y, 0.0, max(height - 1, 0)))
        x_start = int(round(center_x - bounded_width * 0.5))
        y_start = int(round(center_y - bounded_height * 0.5))
        x_start = min(max(x_start, 0), max(width - bounded_width, 0))
        y_start = min(max(y_start, 0), max(height - bounded_height, 0))
        x_end = x_start + bounded_width
        y_end = y_start + bounded_height

        cropped = image[y_start:y_end, x_start:x_end]
        scale_x = 1.0
        scale_y = 1.0
        if bounded_width != crop_width or bounded_height != crop_height:
            scale_x = crop_width / max(bounded_width, 1)
            scale_y = crop_height / max(bounded_height, 1)
            cropped = self._resize_image_nearest(cropped, crop_width, crop_height)
            x_start = 0
            y_start = 0

        cropped_msg = Image()
        cropped_msg.header = image_msg.header
        cropped_msg.height = crop_height
        cropped_msg.width = crop_width
        cropped_msg.encoding = output_encoding
        cropped_msg.is_bigendian = image_msg.is_bigendian
        cropped_msg.step = crop_width * 3
        cropped_msg.data = cropped.tobytes()

        camera_info = copy.deepcopy(camera_info_msg)
        camera_info.header = image_msg.header
        camera_info.width = crop_width
        camera_info.height = crop_height
        self._shift_camera_info(camera_info, x_start, y_start)
        self._scale_camera_info(camera_info, scale_x, scale_y)
        return cropped_msg, camera_info

    def _project_roi_target_to_image(
        self, camera_info_msg
    ) -> tuple[float, float] | None:
        camera_frame = getattr(getattr(camera_info_msg, "header", None), "frame_id", "")
        if not camera_frame:
            return None

        target_frame = self._config.observation.roi_target_frame
        k = list(camera_info_msg.k)
        if len(k) < 6:
            return None
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])
        if fx == 0.0 or fy == 0.0:
            return None

        camera_frames = self._camera_projection_frames(camera_frame)
        for projection_frame in camera_frames:
            try:
                transform = self._parent_node._tf_buffer.lookup_transform(
                    projection_frame,
                    target_frame,
                    Time(),
                )
            except TransformException:
                continue

            point = self._transform_offset_point(transform)
            z = float(point[2])
            if z <= 1e-6:
                continue

            u = fx * float(point[0]) / z + cx
            v = fy * float(point[1]) / z + cy
            if np.isfinite(u) and np.isfinite(v):
                return u, v
        return None

    def _transform_offset_point(self, transform) -> np.ndarray:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        offset = self._rotate_vector_by_quaternion(
            self._roi_target_offset_xyz,
            np.array(
                [rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64
            ),
        )
        return (
            np.array(
                [translation.x, translation.y, translation.z],
                dtype=np.float64,
            )
            + offset
        )

    @staticmethod
    def _rotate_vector_by_quaternion(
        vector: np.ndarray, quat_xyzw: np.ndarray
    ) -> np.ndarray:
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
        norm = np.linalg.norm(quat_xyzw)
        if norm <= 0.0:
            return np.asarray(vector, dtype=np.float64)
        x, y, z, w = quat_xyzw / norm
        q_vec = np.array([x, y, z], dtype=np.float64)
        vector = np.asarray(vector, dtype=np.float64)
        return (
            vector
            + 2.0 * w * np.cross(q_vec, vector)
            + 2.0 * np.cross(q_vec, np.cross(q_vec, vector))
        )

    @staticmethod
    def _camera_projection_frames(camera_frame: str) -> list[str]:
        """Return likely optical frames compatible with CameraInfo intrinsics."""
        frames = []
        if camera_frame.endswith("/sensor_link"):
            frames.append(camera_frame.removesuffix("/sensor_link") + "/optical")
        elif camera_frame.endswith("_sensor_link"):
            frames.append(camera_frame.removesuffix("_sensor_link") + "_optical")
        elif not camera_frame.endswith("optical"):
            frames.append(camera_frame + "/optical")
        frames.append(camera_frame)

        deduped_frames = []
        for frame in frames:
            if frame and frame not in deduped_frames:
                deduped_frames.append(frame)
        return deduped_frames

    @staticmethod
    def _image_msg_to_array(image_msg: Image) -> tuple[np.ndarray, str]:
        encoding = image_msg.encoding.lower()
        if encoding in ("rgb8", "bgr8"):
            channels = 3
            output_encoding = image_msg.encoding
        elif encoding in ("rgba8", "bgra8"):
            channels = 4
            output_encoding = "rgb8" if encoding == "rgba8" else "bgr8"
        elif encoding == "mono8":
            channels = 1
            output_encoding = "rgb8"
        else:
            channels = 3
            output_encoding = image_msg.encoding or "rgb8"

        row_bytes = image_msg.step if image_msg.step else image_msg.width * channels
        expected_bytes = image_msg.height * row_bytes
        raw = np.frombuffer(image_msg.data, dtype=np.uint8)
        if raw.size < expected_bytes:
            raise ValueError(
                f"image data too small: got {raw.size}, expected {expected_bytes}"
            )

        rows = raw[:expected_bytes].reshape(image_msg.height, row_bytes)
        image = rows[:, : image_msg.width * channels].reshape(
            image_msg.height, image_msg.width, channels
        )
        if channels == 4:
            image = image[:, :, :3]
        elif channels == 1:
            image = np.repeat(image, 3, axis=2)
        return np.ascontiguousarray(image), output_encoding

    @staticmethod
    def _resize_image_nearest(
        image: np.ndarray, target_width: int, target_height: int
    ) -> np.ndarray:
        y_indices = np.linspace(0, image.shape[0] - 1, target_height).astype(np.int64)
        x_indices = np.linspace(0, image.shape[1] - 1, target_width).astype(np.int64)
        return np.ascontiguousarray(image[y_indices][:, x_indices])

    @staticmethod
    def _shift_camera_info(camera_info, x_offset: int, y_offset: int) -> None:
        if len(camera_info.k) >= 6:
            k = list(camera_info.k)
            k[2] -= float(x_offset)
            k[5] -= float(y_offset)
            camera_info.k = k
        if len(camera_info.p) >= 7:
            p = list(camera_info.p)
            p[2] -= float(x_offset)
            p[6] -= float(y_offset)
            camera_info.p = p

    @staticmethod
    def _scale_camera_info(camera_info, scale_x: float, scale_y: float) -> None:
        if scale_x == 1.0 and scale_y == 1.0:
            return
        if len(camera_info.k) >= 6:
            k = list(camera_info.k)
            k[0] *= float(scale_x)
            k[2] *= float(scale_x)
            k[4] *= float(scale_y)
            k[5] *= float(scale_y)
            camera_info.k = k
        if len(camera_info.p) >= 7:
            p = list(camera_info.p)
            p[0] *= float(scale_x)
            p[2] *= float(scale_x)
            p[5] *= float(scale_y)
            p[6] *= float(scale_y)
            camera_info.p = p

    def _set_roi_offset_from_task(self, task: Task) -> None:
        self._roi_target_offset_xyz = np.asarray(
            self._task_gripper_offset_xyz(task),
            dtype=np.float64,
        )
        self.get_logger().info(
            "ROI target point in gripper/tcp: "
            f"x={self._roi_target_offset_xyz[0]:.6f}, "
            f"y={self._roi_target_offset_xyz[1]:.6f}, "
            f"z={self._roi_target_offset_xyz[2]:.6f}"
        )

    @staticmethod
    def _task_gripper_offset_xyz(task: Task) -> tuple[float, float, float]:
        plug_type = str(task.plug_type).strip().lower()
        target_module = str(task.target_module_name).strip().lower()

        if plug_type == "sc":
            return (0.0, 0.015385, 0.04045)
        if plug_type == "sfp" and target_module == "nic_card_mount_1":
            return (0.0, 0.015385, 0.04545)
        if plug_type == "sfp":
            return (0.0, 0.015385, 0.04245)
        return (0.0, 0.015385, 0.04045)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"TestPolicy.insert_cable() enter. Task: {task}")
        self._set_roi_offset_from_task(task)
        self._deep_insert = False
        self._observation_adapter.reset()
        self._runtime.reset()

        while not self._deep_insert:
            obs_msg = get_observation()
            if obs_msg is not None:
                self._publish_observation_roi(obs_msg)
            send_feedback("等待 motion_planning 发布 deep_insert=true")
            self.sleep_for(0.5)

        self.get_logger().info("进入 deep-insert 阶段，开始 HIL-SERL actor 推理。")
        send_feedback("deep_insert 已接管，初始化 HIL-SERL actor")

        init_obs_msg = self._wait_for_roi_observation(get_observation)
        if init_obs_msg is None:
            self.get_logger().error("初始化失败：未收到 observations_roi。")
            self._send_zero_twist(move_robot)
            return False

        init_obs = self._observation_adapter.adapt(init_obs_msg)
        self._runtime.initialize(init_obs)
        self._runtime.reset()

        start_time = time.time()
        step_index = 0
        missing_obs_count = 0

        while True:
            if not self._deep_insert:
                self.get_logger().info("deep_insert=false，暂停推理并等待重新接管。")
                self._send_zero_twist(move_robot)
                while not self._deep_insert:
                    obs_msg = get_observation()
                    if obs_msg is not None:
                        self._publish_observation_roi(obs_msg)
                    send_feedback("等待 motion_planning 发布 deep_insert=true")
                    self.sleep_for(0.5)
                self.get_logger().info(
                    "重新收到 deep_insert=true，恢复 HIL-SERL 推理。"
                )
                start_time = time.time()
                missing_obs_count = 0

            elapsed = time.time() - start_time
            if elapsed > self._config.safety.max_runtime_sec:
                self.get_logger().warn("deep-insert 超时，停止 HIL-SERL actor。")
                send_feedback("deep_insert 超时，停止控制")
                self._send_zero_twist(move_robot)
                return False

            raw_obs_msg = get_observation()
            obs_msg = (
                self._publish_observation_roi(raw_obs_msg)
                if raw_obs_msg is not None
                else None
            )
            if obs_msg is None:
                missing_obs_count += 1
                if missing_obs_count >= self._config.safety.max_consecutive_missing_obs:
                    self.get_logger().error("连续丢失 observation，停止控制。")
                    send_feedback("observation 丢失，停止控制")
                    self._send_zero_twist(move_robot)
                    return False
                self.sleep_for(self._config.control.control_period_sec)
                continue

            missing_obs_count = 0

            actor_obs = self._observation_adapter.adapt(obs_msg)
            actor_action = self._runtime.predict(actor_obs)
            command = self._action_adapter.adapt(actor_action)
            twist = self._action_adapter.to_twist(command)
            move_robot(motion_update=self._set_cartesian_twist_target(twist))

            if step_index % 10 == 0:
                send_feedback(
                    f"deep_insert 推理中 step={step_index} elapsed={elapsed:.1f}s"
                )

            step_index += 1
            self.sleep_for(self._config.control.control_period_sec)

    def _wait_for_roi_observation(
        self, get_observation: GetObservationCallback, timeout_sec: float = 5.0
    ):
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            obs = get_observation()
            if obs is not None:
                roi_obs = self._publish_observation_roi(obs)
                if roi_obs is not None:
                    return roi_obs
            self.sleep_for(0.1)
        return None

    def _send_zero_twist(self, move_robot: MoveRobotCallback) -> None:
        zero_twist = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )
        move_robot(motion_update=self._set_cartesian_twist_target(zero_twist))

    def _set_cartesian_twist_target(self, twist: Twist, frame_id: str = "base_link"):
        motion_update_msg = MotionUpdate()
        motion_update_msg.velocity = twist
        motion_update_msg.header.frame_id = frame_id
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()

        motion_update_msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        motion_update_msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()

        motion_update_msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update_msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        motion_update_msg.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
        )
        return motion_update_msg
