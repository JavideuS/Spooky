cmake_minimum_required(VERSION 3.8)
project(quantum_robotics_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages
  "msg/QuantumPlannerParameters.msg"
  "msg/QuantumPlannerSelector.msg"
  "msg/QuantumMetadata.msg"
  "msg/QuantumPlannerStatus.msg"
  
  # Actions  
  "action/ComputeQuantumPath.action"
  
  # Services
  "srv/GetQuantumPlannerInfo.srv"
  "srv/EstimateQuantumResources.srv"

  DEPENDENCIES 
    std_msgs 
    geometry_msgs 
    nav_msgs
    builtin_interfaces

)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
