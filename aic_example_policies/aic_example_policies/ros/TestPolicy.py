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

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from std_msgs.msg import String


class TestPolicy(Policy):
    """Minimal policy that blocks in the deep_insert stage."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._deepinsert_event_topic = "/aic/deep_insert"
        self._deep_insert = False
        self._deepinsert_event_sub = parent_node.create_subscription(
            String,
            self._deepinsert_event_topic,
            self._on_deepinsert_event,
            10,
        )
        self.get_logger().info("TestPolicy.__init__()")

    def _on_deepinsert_event(self, msg: String) -> None:
        self._deep_insert = msg.data.strip().lower() == "true"
        print(f"TestPolicy topic notice: deep_insert={self._deep_insert}")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"TestPolicy.insert_cable() enter. Task: {task}")
        while not self._deep_insert:
            send_feedback("waiting deep_insert == true")
            self.sleep_for(5.0)

        print("TestPolicy main entered in deep_insert stage.")
        while True:
            send_feedback("waiting rl")
            self.sleep_for(5.0)
