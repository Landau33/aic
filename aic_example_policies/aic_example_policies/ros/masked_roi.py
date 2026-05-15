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

"""Publish blue-object masked ROI images for AIC observations."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from aic_model_interfaces.msg import Observation
from sensor_msgs.msg import Image


@dataclass(frozen=True)
class MaskedRoiParams:
    """HSV threshold and fill value used to build the masked ROI image."""

    hue_min: int
    hue_max: int
    saturation_min: int
    value_min: int
    background_value: int


class MaskedRoiPublisher:
    """Build and publish two-channel masked ROI images.

    Channel 0 is grayscale from the source image inside the blue HSV mask and
    ``background_value`` outside the mask. Channel 1 is the binary blue mask
    encoded as 1 for mask pixels and 0 elsewhere.
    """

    def __init__(self, parent_node, topics) -> None:
        self._parent_node = parent_node
        self._topics = topics
        self._observation_pub = parent_node.create_publisher(
            Observation,
            topics.observation_masked_roi_topic,
            10,
        )
        self._left_image_pub = parent_node.create_publisher(
            Image,
            topics.left_image_masked_roi_topic,
            10,
        )
        self._center_image_pub = parent_node.create_publisher(
            Image,
            topics.center_image_masked_roi_topic,
            10,
        )
        self._right_image_pub = parent_node.create_publisher(
            Image,
            topics.right_image_masked_roi_topic,
            10,
        )
        self._publish_count = 0
        self._declare_parameters()

    def _declare_parameters(self) -> None:
        self._declare_parameter_if_needed("hil_serl.masked_roi.hue_min", 90)
        self._declare_parameter_if_needed("hil_serl.masked_roi.hue_max", 130)
        self._declare_parameter_if_needed("hil_serl.masked_roi.saturation_min", 50)
        self._declare_parameter_if_needed("hil_serl.masked_roi.value_min", 50)
        self._declare_parameter_if_needed("hil_serl.masked_roi.background_value", 0)

    def _declare_parameter_if_needed(self, name: str, value) -> None:
        if not self._parent_node.has_parameter(name):
            self._parent_node.declare_parameter(name, value)

    def publish_observation(self, roi_msg: Observation) -> Observation | None:
        """Publish a masked copy of an already-cropped ROI observation."""
        try:
            params = self._params()
            masked_msg = Observation()
            masked_msg.controller_state = roi_msg.controller_state
            masked_msg.wrist_wrench = roi_msg.wrist_wrench
            masked_msg.joint_states = roi_msg.joint_states
            masked_msg.left_camera_info = roi_msg.left_camera_info
            masked_msg.center_camera_info = roi_msg.center_camera_info
            masked_msg.right_camera_info = roi_msg.right_camera_info
            masked_msg.left_image = build_masked_roi_image(roi_msg.left_image, params)
            masked_msg.center_image = build_masked_roi_image(
                roi_msg.center_image,
                params,
            )
            masked_msg.right_image = build_masked_roi_image(roi_msg.right_image, params)
        except Exception as exc:
            self._parent_node.get_logger().warn(
                f"Failed to build observations_masked_roi: {exc}"
            )
            return None

        self._observation_pub.publish(masked_msg)
        self._left_image_pub.publish(masked_msg.left_image)
        self._center_image_pub.publish(masked_msg.center_image)
        self._right_image_pub.publish(masked_msg.right_image)
        self._publish_count += 1
        if self._publish_count % 20 == 1:
            self._parent_node.get_logger().info(
                f"Published {self._topics.observation_masked_roi_topic} "
                f"encoding={masked_msg.center_image.encoding}"
            )
        return masked_msg

    def _params(self) -> MaskedRoiParams:
        return MaskedRoiParams(
            hue_min=self._clamp_u8_param("hil_serl.masked_roi.hue_min"),
            hue_max=self._clamp_u8_param("hil_serl.masked_roi.hue_max"),
            saturation_min=self._clamp_u8_param("hil_serl.masked_roi.saturation_min"),
            value_min=self._clamp_u8_param("hil_serl.masked_roi.value_min"),
            background_value=self._clamp_u8_param(
                "hil_serl.masked_roi.background_value"
            ),
        )

    def _clamp_u8_param(self, name: str) -> int:
        value = int(self._parent_node.get_parameter(name).value)
        return int(np.clip(value, 0, 255))


def build_masked_roi_image(image_msg: Image, params: MaskedRoiParams) -> Image:
    """Convert one ROS image into a two-channel masked ROI image."""
    image_bgr = image_msg_to_bgr_array(image_msg)
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
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    masked_gray = np.full_like(gray, params.background_value, dtype=np.uint8)
    masked_gray[mask] = gray[mask]
    binary_mask = mask.astype(np.uint8)
    masked_roi = np.dstack((masked_gray, binary_mask))

    out_msg = Image()
    out_msg.header = image_msg.header
    out_msg.height = image_msg.height
    out_msg.width = image_msg.width
    out_msg.encoding = "8UC2"
    out_msg.is_bigendian = image_msg.is_bigendian
    out_msg.step = image_msg.width * 2
    out_msg.data = np.ascontiguousarray(masked_roi).tobytes()
    return out_msg


def image_msg_to_bgr_array(image_msg: Image) -> np.ndarray:
    """Read common ROS image encodings into a contiguous BGR uint8 array."""
    if image_msg.height <= 0 or image_msg.width <= 0 or not image_msg.data:
        raise ValueError("empty image")

    encoding = image_msg.encoding.lower()
    if encoding in ("rgb8", "bgr8"):
        channels = 3
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
    elif encoding == "mono8":
        channels = 1
    else:
        channels = 3

    row_bytes = image_msg.step if image_msg.step else image_msg.width * channels
    expected_bytes = image_msg.height * row_bytes
    raw = np.frombuffer(image_msg.data, dtype=np.uint8)
    if raw.size < expected_bytes:
        raise ValueError(
            f"image data too small: got {raw.size}, expected {expected_bytes}"
        )

    rows = raw[:expected_bytes].reshape(image_msg.height, row_bytes)
    image = rows[:, : image_msg.width * channels].reshape(
        image_msg.height,
        image_msg.width,
        channels,
    )

    if encoding == "rgb8":
        image = image[:, :, ::-1]
    elif encoding == "rgba8":
        image = image[:, :, :3][:, :, ::-1]
    elif encoding == "bgra8":
        image = image[:, :, :3]
    elif encoding == "mono8":
        image = np.repeat(image, 3, axis=2)

    return np.ascontiguousarray(image)
