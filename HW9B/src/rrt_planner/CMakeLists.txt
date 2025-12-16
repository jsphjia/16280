cmake_minimum_required(VERSION 3.10)
project(rrt_planner)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(random_numbers REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME} SHARED src/rrt_planner_manager.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp pluginlib moveit_core moveit_ros_planning
  moveit_ros_planning_interface random_numbers)

pluginlib_export_plugin_description_file(moveit_core plugin_description.xml)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(FILES plugin_description.xml DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
