cmake_minimum_required(VERSION 3.24)
project(airplanes)

set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( airplanes main.cpp )
target_link_libraries( airplanes ${OpenCV_LIBS} )
