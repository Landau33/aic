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

"""Publish RViz-friendly mono8 views of masked ROI images."""

from __future__ import annotations

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from .masked_roi import MaskedRoiParams, image_msg_to_bgr_array


class MaskedRoiViewer(Node):
    """Publish grayscale and binary-mask views from ROI image topics."""

    def __init__(self) -> None:
        super().__init__("masked_roi_viewer")
        self.declare_parameter("input_prefix", "observations_roi")
        self.declare_parameter("output_prefix", "observations_masked_roi_vis")
        self.declare_parameter("hue_min", 90)
        self.declare_parameter("hue_max", 130)
        self.declare_parameter("saturation_min", 50)
        self.declare_parameter("value_min", 50)
        self.declare_parameter("background_value", 0)

        input_prefix = str(self.get_parameter("input_prefix").value).strip("/")
        output_prefix = str(self.get_parameter("output_prefix").value).strip("/")

        self._subs = []
        self._gray_pubs = {}
        self._mask_pubs = {}
        self._message_counts = {}
        for camera_name in ("left", "center", "right"):
            input_topic = f"{input_prefix}/{camera_name}_image"
            gray_topic = f"{output_prefix}/{camera_name}/gray"
            mask_topic = f"{output_prefix}/{camera_name}/mask"

            self._gray_pubs[camera_name] = self.create_publisher(Image, gray_topic, 10)
            self._mask_pubs[camera_name] = self.create_publisher(Image, mask_topic, 10)
            self._subs.append(
                self.create_subscription(
                    Image,
                    input_topic,
                    lambda msg, name=camera_name: self._on_image(name, msg),
                    10,
                )
            )
            self._message_counts[camera_name] = 0
            self.get_logger().info(f"{input_topic} -> {gray_topic}, {mask_topic}")

    def _on_image(self, camera_name: str, msg: Image) -> None:
        try:
            gray, mask_vis = self._build_views(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to visualize {camera_name} ROI: {exc}")
            return

        self._gray_pubs[camera_name].publish(self._mono8_msg(msg, gray))
        self._mask_pubs[camera_name].publish(self._mono8_msg(msg, mask_vis))
        self._message_counts[camera_name] += 1
        if self._message_counts[camera_name] % 30 == 1:
            self.get_logger().info(
                f"Published {camera_name} gray/mask visualization "
                f"from {msg.encoding}"
            )

    def _build_views(self, msg: Image) -> tuple[np.ndarray, np.ndarray]:
        if msg.encoding.lower() == "8uc2":
            masked_roi = self._image_msg_to_8uc2(msg)
            gray = np.ascontiguousarray(masked_roi[:, :, 0])
            mask_vis = np.ascontiguousarray(masked_roi[:, :, 1] * 255)
            return gray, mask_vis

        params = self._params()
        image_bgr = image_msg_to_bgr_array(msg)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower = np.asarray(
            [
                min(params.hue_min, params.hue_max),
                params.saturation_min,
                params.value_min,
            ],
            dtype=np.uint8,
        )
        upper = np.asarray(
            [max(params.hue_min, params.hue_max), 255, 255],
            dtype=np.uint8,
        )
        mask = cv2.inRange(hsv, lower, upper) > 0
        source_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = np.full_like(source_gray, params.background_value, dtype=np.uint8)
        gray[mask] = source_gray[mask]
        mask_vis = mask.astype(np.uint8) * 255
        return np.ascontiguousarray(gray), np.ascontiguousarray(mask_vis)

    def _params(self) -> MaskedRoiParams:
        return MaskedRoiParams(
            hue_min=self._clamp_u8_param("hue_min"),
            hue_max=self._clamp_u8_param("hue_max"),
            saturation_min=self._clamp_u8_param("saturation_min"),
            value_min=self._clamp_u8_param("value_min"),
            background_value=self._clamp_u8_param("background_value"),
        )

    def _clamp_u8_param(self, name: str) -> int:
        value = int(self.get_parameter(name).value)
        return int(np.clip(value, 0, 255))

    @staticmethod
    def _image_msg_to_8uc2(msg: Image) -> np.ndarray:
        channels = 2
        row_bytes = msg.step if msg.step else msg.width * channels
        expected_bytes = msg.height * row_bytes
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if raw.size < expected_bytes:
            raise ValueError(f"got {raw.size} bytes, expected {expected_bytes}")

        rows = raw[:expected_bytes].reshape(msg.height, row_bytes)
        image = rows[:, : msg.width * channels].reshape(
            msg.height,
            msg.width,
            channels,
        )
        return np.ascontiguousarray(image)

    @staticmethod
    def _mono8_msg(source_msg: Image, image: np.ndarray) -> Image:
        msg = Image()
        msg.header = source_msg.header
        msg.height = source_msg.height
        msg.width = source_msg.width
        msg.encoding = "mono8"
        msg.is_bigendian = source_msg.is_bigendian
        msg.step = source_msg.width
        msg.data = image.astype(np.uint8, copy=False).tobytes()
        return msg


def main() -> None:
    rclpy.init()
    node = MaskedRoiViewer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
