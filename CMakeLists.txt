cmake_minimum_required(VERSION 3.8)
project(find_free_space)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(capella_ros_msg REQUIRED)
find_package(garage_utils_msgs REQUIRED)
find_package(garage_road_model REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

add_executable(find_parking_space
  src/main.cpp
  src/car_avoidance_point_server.cpp
  src/avoidance_search.cpp
  src/candidate_checks.cpp
  src/geometry.cpp
)
target_include_directories(find_parking_space PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(find_parking_space
  rclcpp
  rclcpp_action
  geometry_msgs
  nav2_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  capella_ros_msg
  garage_utils_msgs
  garage_road_model
)
target_link_libraries(find_parking_space garage_road_model::road_model_core ${OpenCV_LIBS})

install(TARGETS find_parking_space
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch params
  DESTINATION share/${PROJECT_NAME})

ament_package()
