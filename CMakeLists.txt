
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
SET(GCC_COVERAGE_COMPILE_FLAGS "-g")
SET(GCC_COVERAGE_LINK_FLAGS    "")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

project(test)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable(TATS main.cpp)

target_link_libraries(TATS "${TORCH_LIBRARIES}")
target_link_libraries(TATS ${OpenCV_LIBS})

set_property(TARGET TATS PROPERTY CXX_STANDARD 17)
