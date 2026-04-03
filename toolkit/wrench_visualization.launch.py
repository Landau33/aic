from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    target_frame_arg = DeclareLaunchArgument(
        "target_frame",
        default_value="cable_0/sfp_tip_link",
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

    wrench_tf_processor = Node(
        package="my_policy_node",
        executable="wrench_tf_processor_node",
        name="wrench_tf_processor_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_processor")),
        parameters=[
            {
                "target_frame": LaunchConfiguration("target_frame"),
                "source_frame": LaunchConfiguration("source_frame"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
        ],
    )

    wrench_plot = Node(
        package="my_policy_node",
        executable="wrench_plot_node",
        name="wrench_plot_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_plot")),
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
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
