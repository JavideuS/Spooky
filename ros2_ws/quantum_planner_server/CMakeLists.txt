cmake_minimum_required(VERSION 3.8)
project(quantum_planner_server)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(quantum_robotics_interface REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

set(dependencies
  rclcpp
  rclcpp_action
  quantum_robotics_interface
  grid_map_msgs
  ament_index_cpp
  nlohmann_json
)

include_directories(include)

add_library(${PROJECT_NAME} SHARED
  src/nodes/ServerNode.cpp
  src/nodes/ClientNode.cpp
)
ament_target_dependencies(${PROJECT_NAME} ${dependencies})

add_executable(quantum_server src/server_main.cpp)
ament_target_dependencies(quantum_server ${dependencies})
target_link_libraries(quantum_server ${PROJECT_NAME})

add_executable(quantum_client src/client_main.cpp)
ament_target_dependencies(quantum_client ${dependencies})
target_link_libraries(quantum_client ${PROJECT_NAME})

install(DIRECTORY maps config launch DESTINATION share/${PROJECT_NAME})

install(TARGETS
  ${PROJECT_NAME}
  quantum_server
  quantum_client
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
