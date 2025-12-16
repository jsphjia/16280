from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import ExecuteProcess, TimerAction
import os

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("panda", package_name="moveit_resources_panda_moveit_config")
        .to_moveit_configs()
    )

    rrt_pipeline_params = {
        "planning_pipelines": ["rrt"],
        "rrt": {
            "planning_plugin": "RrtPlannerPlugin/RrtPlannerManager",
            "request_adapters": (
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints "
                "default_planner_request_adapters/ResolveConstraintFrames "
                "default_planner_request_adapters/AddTimeOptimalParameterization"
            ),
        },
    }


    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), rrt_pipeline_params],
    )

    rviz_config = os.path.join(
        get_package_share_directory("rrt_planner"),
        "config",
        "rrt_moveit.rviz",
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[moveit_config.to_dict()],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0", "0", "0", "0", "0", "0", "world", "panda_link0"],
    )

    joint_driver = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            os.path.join(
                get_package_share_directory("moveit_resources_panda_moveit_config"),
                "config",
                "ros2_controllers.yaml",
            ),
        ],
        output="screen",
    )

    load_controllers = []
    for controller in ["panda_arm_controller", "joint_state_broadcaster"]:
        load_controllers.append(
            ExecuteProcess(
                cmd=["ros2", "control", "load_controller", "--set-state", "active", controller],
                output="screen",
            )
        )

    delayed_controller_launch = TimerAction(
        period=2.0,
        actions=load_controllers,
    )

    return LaunchDescription([
        robot_state_publisher,
        static_tf,
        joint_driver,
        delayed_controller_launch,
        move_group_node,
        rviz_node,
    ])
