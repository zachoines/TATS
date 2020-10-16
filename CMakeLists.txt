cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

SET(CMAKE_PREFIX_PATH "/home/zachoines/Documents/pytorch/build/lib.linux-aarch64-3.6")
SET(GCC_COVERAGE_COMPILE_FLAGS "-g")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

project(test)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost 1.65.1 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB_RECURSE SOURCES "main.cpp" "./src/*/*.cpp" "./src/*/*.h")
add_executable(TATS ${SOURCES})

target_link_libraries(TATS "${TORCH_LIBRARIES}")
target_link_libraries(TATS ${OpenCV_LIBS})
target_link_libraries(TATS ${BOOST_LIBRARYDIR})
target_link_libraries(TATS ${Boost_LIBRARIES} rt)

set_property(TARGET TATS PROPERTY CXX_STANDARD 17)
