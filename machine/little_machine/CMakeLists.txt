cmake_minimum_required(VERSION 3.22)

project(little_machine)

find_package(OpenCV REQUIRED)

include_directories(${RESOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
add_executable(Demo little_machine.cpp)

target_link_libraries(Demo ${OpenCV_LIBS})
