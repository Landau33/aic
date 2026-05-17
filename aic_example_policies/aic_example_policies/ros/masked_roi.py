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

"""Publish blue + fixed-plug-region masked ROI images for AIC observations."""

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


@dataclass(frozen=True)
class PlugBbox:
    """Fixed plug rectangle in image pixel coordinates (inclusive-exclusive)."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def is_empty(self) -> bool:
        return self.x_max <= self.x_min or self.y_max <= self.y_min


class MaskedRoiPublisher:
    """Build and publish masked RGB ROI images plus a binary mask vis topic.

    The bgr8 masked output keeps the source pixels inside the blue-HSV mask
    unioned with a fixed plug rectangle and fills ``background_value``
    elsewhere; it feeds the policy. A mono8 0/255 mask topic under
    ``vis_prefix`` mirrors the same mask for RViz inspection.
    """

    VIS_PREFIX = "observations_masked_roi_vis"

    def __init__(self, parent_node, topics) -> None:
        self._parent_node = parent_node
        self._topics = topics
        self._started = False
        self._observation_pub = None
        self._image_pubs: dict[str, object] = {}
        self._vis_mask_pubs: dict[str, object] = {}
        self._publish_count = 0
        self._declare_parameters()

    def start(self) -> None:
        """Create all publishers. Idempotent; call once from the owner."""
        if self._started:
            return
        topics = self._topics
        self._observation_pub = self._parent_node.create_publisher(
            Observation, topics.observation_masked_roi_topic, 10
        )
        self._image_pubs = {
            "left": self._parent_node.create_publisher(
                Image, topics.left_image_masked_roi_topic, 10
            ),
            "center": self._parent_node.create_publisher(
                Image, topics.center_image_masked_roi_topic, 10
            ),
            "right": self._parent_node.create_publisher(
                Image, topics.right_image_masked_roi_topic, 10
            ),
        }
        for name in ("left", "center", "right"):
            self._vis_mask_pubs[name] = self._parent_node.create_publisher(
                Image, f"{self.VIS_PREFIX}/{name}/mask", 10
            )
        self._started = True
        self._parent_node.get_logger().info(
            f"MaskedRoiPublisher started; mask vis under {self.VIS_PREFIX}/<cam>/mask"
        )

    def _declare_parameters(self) -> None:
        self._declare_parameter_if_needed("hil_serl.masked_roi.enabled", True)
        self._declare_parameter_if_needed("hil_serl.masked_roi.hue_min", 90)
        self._declare_parameter_if_needed("hil_serl.masked_roi.hue_max", 130)
        self._declare_parameter_if_needed("hil_serl.masked_roi.saturation_min", 50)
        self._declare_parameter_if_needed("hil_serl.masked_roi.value_min", 50)
        self._declare_parameter_if_needed("hil_serl.masked_roi.background_value", 128)
        self._declare_parameter_if_needed(
            "hil_serl.masked_roi.plug_bbox.left", [110, 200, 220, 270]
        )
        self._declare_parameter_if_needed(
            "hil_serl.masked_roi.plug_bbox.center", [65, 270, 135, 400]
        )
        self._declare_parameter_if_needed(
            "hil_serl.masked_roi.plug_bbox.right", [80, 220, 190, 270]
        )

    def is_enabled(self) -> bool:
        value = self._parent_node.get_parameter("hil_serl.masked_roi.enabled").value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    def _declare_parameter_if_needed(self, name: str, value) -> None:
        if not self._parent_node.has_parameter(name):
            self._parent_node.declare_parameter(name, value)

    def publish_observation(self, roi_msg: Observation) -> Observation | None:
        """Build masks once; always publish mask vis, bgr8 only when enabled."""
        if not self._started:
            return None
        try:
            params = self._params()
            plug_bboxes = self._plug_bboxes()
            built = {
                name: _build_masked_outputs(
                    getattr(roi_msg, f"{name}_image"), params, plug_bboxes[name]
                )
                for name in ("left", "center", "right")
            }
        except Exception as exc:
            self._parent_node.get_logger().warn(
                f"Failed to build observations_masked_roi: {exc}"
            )
            return None

        for name, (_, mask_msg) in built.items():
            self._vis_mask_pubs[name].publish(mask_msg)

        if not self.is_enabled():
            return None

        masked_msg = Observation()
        masked_msg.controller_state = roi_msg.controller_state
        masked_msg.wrist_wrench = roi_msg.wrist_wrench
        masked_msg.joint_states = roi_msg.joint_states
        masked_msg.left_camera_info = roi_msg.left_camera_info
        masked_msg.center_camera_info = roi_msg.center_camera_info
        masked_msg.right_camera_info = roi_msg.right_camera_info
        masked_msg.left_image = built["left"][0]
        masked_msg.center_image = built["center"][0]
        masked_msg.right_image = built["right"][0]

        self._observation_pub.publish(masked_msg)
        self._image_pubs["left"].publish(masked_msg.left_image)
        self._image_pubs["center"].publish(masked_msg.center_image)
        self._image_pubs["right"].publish(masked_msg.right_image)
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

    def _plug_bboxes(self) -> dict[str, PlugBbox]:
        return {
            "left": self._read_plug_bbox("hil_serl.masked_roi.plug_bbox.left"),
            "center": self._read_plug_bbox("hil_serl.masked_roi.plug_bbox.center"),
            "right": self._read_plug_bbox("hil_serl.masked_roi.plug_bbox.right"),
        }

    def _read_plug_bbox(self, name: str) -> PlugBbox:
        raw = list(self._parent_node.get_parameter(name).value)
        if len(raw) != 4:
            return PlugBbox(0, 0, 0, 0)
        return PlugBbox(int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))

    def _clamp_u8_param(self, name: str) -> int:
        value = int(self._parent_node.get_parameter(name).value)
        return int(np.clip(value, 0, 255))


def build_masked_roi_image(
    image_msg: Image,
    params: MaskedRoiParams,
    plug_bbox: PlugBbox,
) -> Image:
    """Convert one ROS image into a bgr8 masked ROI image."""
    image_bgr = image_msg_to_bgr_array(image_msg)
    masked_bgr, _ = build_masked_rgb_and_mask(image_bgr, params, plug_bbox)
    return _pack_bgr8(image_msg, masked_bgr)


def _pack_bgr8(source: Image, image_bgr: np.ndarray) -> Image:
    out_msg = Image()
    out_msg.header = source.header
    out_msg.height = source.height
    out_msg.width = source.width
    out_msg.encoding = "bgr8"
    out_msg.is_bigendian = source.is_bigendian
    out_msg.step = source.width * 3
    out_msg.data = np.ascontiguousarray(image_bgr).tobytes()
    return out_msg


def _pack_mono8(source: Image, image: np.ndarray) -> Image:
    out_msg = Image()
    out_msg.header = source.header
    out_msg.height = source.height
    out_msg.width = source.width
    out_msg.encoding = "mono8"
    out_msg.is_bigendian = source.is_bigendian
    out_msg.step = source.width
    out_msg.data = image.astype(np.uint8, copy=False).tobytes()
    return out_msg


def _build_masked_outputs(
    image_msg: Image,
    params: MaskedRoiParams,
    plug_bbox: PlugBbox,
) -> tuple[Image, Image]:
    """Build the bgr8 masked image and mono8 mask vis from one source image."""
    image_bgr = image_msg_to_bgr_array(image_msg)
    masked_bgr, binary_mask = build_masked_rgb_and_mask(image_bgr, params, plug_bbox)
    masked_msg = _pack_bgr8(image_msg, masked_bgr)
    mask_vis = _pack_mono8(image_msg, binary_mask * 255)
    return masked_msg, mask_vis


def build_masked_rgb_and_mask(
    image_bgr: np.ndarray,
    params: MaskedRoiParams,
    plug_bbox: PlugBbox,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the mask to the BGR image and return (masked_bgr, 0/1 mask)."""
    mask = build_blue_mask(image_bgr, params)
    apply_plug_bbox(mask, plug_bbox)
    mask = fill_mask_polygons(mask)

    masked = np.full_like(image_bgr, params.background_value, dtype=np.uint8)
    masked[mask] = image_bgr[mask]
    binary_mask = mask.astype(np.uint8)
    return masked, binary_mask


def fill_mask_polygons(mask: np.ndarray) -> np.ndarray:
    """Fill every connected region as a solid polygon (no interior holes)."""
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.astype(bool)
    filled = np.zeros_like(mask_u8)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled > 0


def build_blue_mask(
    image_bgr: np.ndarray,
    params: MaskedRoiParams,
) -> np.ndarray:
    """Return a boolean mask of blue HSV pixels."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    blue_lower = np.asarray(
        [
            min(params.hue_min, params.hue_max),
            params.saturation_min,
            params.value_min,
        ],
        dtype=np.uint8,
    )
    blue_upper = np.asarray(
        [max(params.hue_min, params.hue_max), 255, 255],
        dtype=np.uint8,
    )
    return cv2.inRange(hsv, blue_lower, blue_upper) > 0


def apply_plug_bbox(mask: np.ndarray, plug_bbox: PlugBbox) -> None:
    """Force the fixed plug rectangle into ``mask`` in place."""
    if plug_bbox.is_empty():
        return
    h, w = mask.shape[:2]
    x_min = max(0, min(plug_bbox.x_min, w))
    x_max = max(0, min(plug_bbox.x_max, w))
    y_min = max(0, min(plug_bbox.y_min, h))
    y_max = max(0, min(plug_bbox.y_max, h))
    if x_max <= x_min or y_max <= y_min:
        return
    mask[y_min:y_max, x_min:x_max] = True


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
