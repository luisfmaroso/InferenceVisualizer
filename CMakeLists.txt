cmake_minimum_required(VERSION 3.21)

project(InferenceVisualizer
    VERSION 0.1.0
    DESCRIPTION "Qt + OpenCV computer-vision inference viewer (learning project)"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Put binaries in build/bin so we don't dig through CMake's generator-specific
# subdirectories to find the .exe.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(Qt6 6.5 REQUIRED COMPONENTS Widgets)
qt_standard_project_setup()

add_subdirectory(src)
