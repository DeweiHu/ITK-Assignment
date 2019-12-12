cmake_minimum_required(VERSION 2.8)

project (Bayesian)

find_package (VTK REQUIRED)
include (${VTK_USE_FILE})

find_package (ITK REQUIRED)
include (${ITK_USE_FILE})

add_executable (Bayesian Bayesian.cxx)

target_link_libraries (Bayesian ${ITK_LIBRARIES} ${VTK_LIBRARIES})
