cmake_minimum_required(VERSION 3.0.2)
project(robotiq_2f_85_gripper_visualization)
find_package(catkin REQUIRED)
catkin_package()

install(DIRECTORY config launch meshes urdf DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})