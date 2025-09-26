cmake_minimum_required(VERSION 3.8)
project(cpp_nodes)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED)

# Add executable
add_executable(sync_node src/msg_sync_node.cpp)
add_executable(no_sync_node src/no_sync_node.cpp)

# Link libraries
ament_target_dependencies(sync_node
  rclcpp
  sensor_msgs
  nav_msgs
  message_filters
)

ament_target_dependencies(
  no_sync_node
  rclcpp
  sensor_msgs
  nav_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install target
install(TARGETS sync_node no_sync_node 
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
