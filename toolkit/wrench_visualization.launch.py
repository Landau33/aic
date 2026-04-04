import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    toolkit_dir = os.path.dirname(os.path.abspath(__file__))
    processor_script = os.path.join(toolkit_dir, "wrench_tf_processor_node.py")
    plot_script = os.path.join(toolkit_dir, "wrench_plot_node.py")

    target_frame_arg = DeclareLaunchArgument(
        "target_frame",
        default_value="gripper/tcp",
        description="Target frame for wrench transformation",
    )
    source_frame_arg = DeclareLaunchArgument(
        "source_frame",
        default_value="ati/tool_link",
        description="Source frame of raw wrench",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock",
    )
    enable_processor_arg = DeclareLaunchArgument(
        "enable_processor",
        default_value="true",
        description="Whether to start wrench_tf_processor_node",
    )
    enable_plot_arg = DeclareLaunchArgument(
        "enable_plot",
        default_value="true",
        description="Whether to start wrench_plot_node",
    )

    wrench_tf_processor = ExecuteProcess(
        cmd=[
            sys.executable,
            processor_script,
            "--ros-args",
            "-p",
            ["target_frame:=", LaunchConfiguration("target_frame")],
            "-p",
            ["source_frame:=", LaunchConfiguration("source_frame")],
            "-p",
            ["use_sim_time:=", LaunchConfiguration("use_sim_time")],
        ],
        name="wrench_tf_processor_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_processor")),
    )

    wrench_plot = ExecuteProcess(
        cmd=[
            sys.executable,
            plot_script,
            "--ros-args",
            "-p",
            ["use_sim_time:=", LaunchConfiguration("use_sim_time")],
        ],
        name="wrench_plot_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_plot")),
    )

    return LaunchDescription(
        [
            target_frame_arg,
            source_frame_arg,
            use_sim_time_arg,
            enable_processor_arg,
            enable_plot_arg,
            wrench_tf_processor,
            wrench_plot,
        ]
    )
