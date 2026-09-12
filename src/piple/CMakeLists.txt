cmake_minimum_required(VERSION 3.10)

project(piple)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ament_cmake_python REQUIRED)

add_executable(base_control_node src/nodes/base_control_node.cpp)
add_executable(twist_adapter src/adapters/twist_adapter.cpp)
add_executable(auto_drive src/autonomous/drive_square.cpp)

ament_target_dependencies(
  base_control_node
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)
ament_target_dependencies(
  twist_adapter
  rclcpp
  geometry_msgs
)

ament_target_dependencies(
    auto_drive
    rclcpp
    geometry_msgs
)

install(
  TARGETS base_control_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  twist_adapter
  DESTINATION lib/${PROJECT_NAME}
)

install(
    TARGETS auto_drive
    DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
