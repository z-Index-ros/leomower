cmake_minimum_required(VERSION 3.0.2)
project(leo_mower_gazebo)

find_package(catkin REQUIRED COMPONENTS
  xacro
)

catkin_package()

file(GLOB xacro_files worlds/*.world.xacro)
xacro_add_files(${xacro_files} TARGET generate_world_files)

install(
  DIRECTORY
    config
    launch
    media
    models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(
  PROGRAMS
    scripts/odom_compat_node
    scripts/zed2_imu_transform_broadcaster
    scripts/pdu_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "*.world"
)
