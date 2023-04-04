cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(PointCloudViewer)

find_package(PCL 1.2 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(headers ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(SOURCES
  src/main.cpp
)

set(HEADERS
  headers/main.h
)

add_executable(PointCloudViewer ${SOURCES} ${HEADERS})
target_link_libraries(PointCloudViewer ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES} ${VTK_LIBRARIES} ${PCL_FEATURES_LIBRARIES})
