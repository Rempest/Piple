import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    package_path = get_package_share_directory("piple")

    urdf_path = os.path.join(
        package_path,
        "urdf",
        "piple.urdf"
    )

    with open(urdf_path, "r") as f:
        robot_description = f.read()

    return LaunchDescription([

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{
                "robot_description": robot_description
            }],
            output="screen"
        ),

        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            output="screen"
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            output="screen"
        )

    ])