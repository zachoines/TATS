cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
SET(CMAKE_PREFIX_PATH "/usr/local/lib/python3.6/dist-packages/torch")
# SET(CMAKE_PREFIX_PATH "/home/zachoines/.local/lib/python3.6/site-packages/torch")
SET(GCC_COVERAGE_COMPILE_FLAGS "-g")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

project(test)

find_package(Torch 1.7.0 REQUIRED)
find_package(OpenCV 4.4.0 REQUIRED)
find_package(Boost 1.65.1 REQUIRED)

message(STATUS "Found Torch VERSION: ${Torch_VERSION}")
add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})
add_definitions(-DTORCH_VERSION_PATCH=${Torch_VERSION_PATCH})


include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB_RECURSE SOURCES "./examples/rc_robot_with_target_detection_and_tracking.cpp" "./src/*/*.cpp" "./src/*/*.h")
add_executable(TATS ${SOURCES})

target_link_libraries(TATS "${TORCH_LIBRARIES}")
target_link_libraries(TATS ${OpenCV_LIBS})
target_link_libraries(TATS ${BOOST_LIBRARYDIR})
target_link_libraries(TATS ${Boost_LIBRARIES} rt)
target_link_libraries(TATS librf24.so)

set_property(TARGET TATS PROPERTY CXX_STANDARD 17)
