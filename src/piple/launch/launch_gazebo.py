import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, AppendEnvironmentVariable, TimerAction
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    package_path = get_package_share_directory("piple")

    # ==============================
    # Paths
    # ==============================

    world_path = os.path.join(
        package_path,
        "worlds",
        "warehouse_world.sdf"
    )

    urdf_path = os.path.join(
        package_path,
        "urdf",
        "piple.urdf"
    )

    with open(urdf_path, "r") as file:
        robot_description = file.read()

    parent_share_path = os.path.dirname(package_path)

    meshes_path = os.path.join(
        package_path,
        "meshes"
    )

    worlds_path = os.path.join(
        package_path,
        "worlds"
    )

    # ==============================
    # Gazebo environment
    # ==============================

    gz_resource_path = AppendEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=f"{parent_share_path}:{meshes_path}:{worlds_path}"
    )

    gz_plugin_path = AppendEnvironmentVariable(
        name="GZ_SIM_SYSTEM_PLUGIN_PATH",
        value="/opt/ros/jazzy/lib"
    )

    # ==============================
    # Gazebo
    # ==============================

    gazebo = ExecuteProcess(
        cmd=[
            "gz",
            "sim",
            "-r",
            world_path
        ],
        output="screen"
    )

    # ==============================
    # Camera bridge
    # ==============================

    camera_bridge = Node(
    package="ros_gz_bridge",
    executable="parameter_bridge",
    arguments=[
        "/camera@sensor_msgs/msg/Image@gz.msgs.Image",
        "/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
    ],
    output="screen"
)
    #==============================
    # LIDAR
    #==============================
        # ==============================
    # LiDAR bridge
    # ==============================

    lidar_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/piple/lidar/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan"
        ],
        output="screen"
    )

    # ==============================
    # Robot State Publisher
    # ==============================

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True
            }
        ]
    )

    # ==============================
    # Spawn Piple
    # ==============================

    spawn_robot = TimerAction(
        period=10.0,
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=[
                    "-topic",
                    "robot_description",

                    "-name",
                    "piple",

                    "-x",
                    "0.0",

                    "-y",
                    "0.0",

                    "-z",
                    "0.08",

                    "-R",
                    "0.0",

                    "-P",
                    "0.0",

                    "-Y",
                    "-0.0"
                ],
                output="screen"
            )
        ]
    )

    # ==============================
    # Controllers
    # ==============================
    # Start after Piple has been spawned.

    controllers = TimerAction(
        period=15.0,
        actions=[

            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "joint_state_broadcaster",
                    "--controller-manager",
                    "/controller_manager"
                ],
                output="screen"
            ),

            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "diff_drive_controller",
                    "--controller-manager",
                    "/controller_manager"
                ],
                output="screen"
            )

        ]
    )

    # ==============================
    # Twist adapter
    # ==============================

    twist_adapter = Node(
        package="piple",
        executable="twist_adapter",
        output="screen"
    )

    # ==============================
    # RVIZ
    # ==============================
    """
    rviz_config_path = os.path.join(package_path, "rviz", "piple.rviz")

    rviz = Node(
    package="rviz2",
    executable="rviz2",
    arguments=["-d", rviz_config_path],
    output="screen"
)
    """

    # ==============================
    # Launch
    # ==============================

    return LaunchDescription([

        gz_resource_path,
        gz_plugin_path,

        gazebo,

        robot_state_publisher,

        spawn_robot,

        controllers,

        camera_bridge,

        twist_adapter,
        #rviz

        lidar_bridge

    ])