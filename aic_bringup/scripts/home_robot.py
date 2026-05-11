#!/usr/bin/env python3

#
#  Copyright (C) 2025 Intrinsic Innovation LLC
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

import sys
import time
import rclpy
from rclpy.executors import ExternalShutdownException

from aic_engine_interfaces.srv import ResetJoints
from aic_control_interfaces.msg import TargetMode
from aic_control_interfaces.srv import ChangeTargetMode
from controller_manager_msgs.srv import SwitchController
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint


class HomeTrajectoryNode(Node):
    def __init__(self):
        super().__init__("home_trajectory_node")
        self.get_logger().info("HomeTrajectoryNode started")

        # Declare parameters.
        self.use_aic_control = self.declare_parameter("use_aic_controller", True).value
        self.controller_namespace = self.declare_parameter(
            "controller_namespace", "aic_controller"
        ).value
        self.deep_insert_topic = self.declare_parameter(
            "deep_insert_topic", "/aic/deep_insert"
        ).value
        self.home_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        # Match the AIC engine/sample_config initial state.
        self.home_joint_positions = [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110]
        self.deep_insert_pub = self.create_publisher(String, self.deep_insert_topic, 10)
        # Create publisher if needed.
        if self.use_aic_control:
            self.switch_controller_client = self.create_client(
                SwitchController, "/controller_manager/switch_controller"
            )
            self.reset_joints_client = self.create_client(
                ResetJoints, "/scoring/reset_joints"
            )
            self.change_target_mode_client = self.create_client(
                ChangeTargetMode, f"/{self.controller_namespace}/change_target_mode"
            )
            while not self.switch_controller_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    "Waiting for /controller_manager/switch_controller..."
                )
            while not self.reset_joints_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for /scoring/reset_joints...")
                time.sleep(1.0)
            while not self.change_target_mode_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    f"Waiting for /{self.controller_namespace}/change_target_mode..."
                )
                time.sleep(1.0)

        else:
            self.action_client = ActionClient(
                self,
                FollowJointTrajectory,
                "/joint_trajectory_controller/follow_joint_trajectory",
            )
            while not self.action_client.wait_for_server(timeout_sec=1.0):
                self.get_logger().info(f"Waiting for {self.action_client._action_name}")

        # A timer that will send the trajectory only once.
        self.timer = self.create_timer(1.0, self.send_trajectory)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return
        self.get_logger().info("Home trajectory goal accepted")
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        rclpy.shutdown()

    def switch_controllers(self, activate, deactivate):
        request = SwitchController.Request()
        request.activate_controllers = list(activate)
        request.deactivate_controllers = list(deactivate)
        request.strictness = SwitchController.Request.BEST_EFFORT
        future = self.switch_controller_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result()
        return response is not None and response.ok

    def reset_deep_insert_trigger(self):
        msg = String()
        msg.data = "false"
        self.deep_insert_pub.publish(msg)
        self.get_logger().info(
            f"Published deep_insert=false to {self.deep_insert_topic}"
        )

    def set_cartesian_target_mode(self):
        request = ChangeTargetMode.Request()
        request.target_mode.mode = TargetMode.MODE_CARTESIAN
        future = self.change_target_mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result()
        return response is not None and response.success

    def send_trajectory(self):
        self.reset_deep_insert_trigger()
        if self.use_aic_control:
            if not self.switch_controllers([], [self.controller_namespace]):
                self.get_logger().error(
                    f"Failed to deactivate {self.controller_namespace}"
                )
                self.timer.cancel()
                return

            request = ResetJoints.Request()
            request.joint_names = self.home_joint_names
            request.initial_positions = self.home_joint_positions
            future = self.reset_joints_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            response = future.result()
            if response is None or not response.success:
                message = "no response" if response is None else response.message
                self.get_logger().error(f"Reset joints failed: {message}")
                self.timer.cancel()
                return

            if not self.switch_controllers([self.controller_namespace], []):
                self.get_logger().error(
                    f"Failed to reactivate {self.controller_namespace}"
                )
                self.timer.cancel()
                return
            if not self.set_cartesian_target_mode():
                self.get_logger().error(
                    f"Failed to switch {self.controller_namespace} back to Cartesian target mode"
                )
                self.timer.cancel()
                return

            self.get_logger().info("Reset robot to AIC initial joint state")
        else:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = self.home_joint_names
            home_point = JointTrajectoryPoint()
            home_point.positions = self.home_joint_positions
            home_point.time_from_start.sec = 1
            goal.trajectory.points.append(home_point)
            self.send_goal_future = self.action_client.send_goal_async(goal)
            self.send_goal_future.add_done_callback(self.goal_response_callback)

        self.timer.cancel()  # Send only once.


def main(args=None):
    try:
        with rclpy.init(args=args):
            node = HomeTrajectoryNode()
            node.send_trajectory()
            if node.use_aic_control:
                rclpy.spin_once(node, timeout_sec=0.1)
            else:
                rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main(sys.argv)
