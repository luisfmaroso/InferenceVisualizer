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

find_package(Qt6 6.5 REQUIRED COMPONENTS Widgets Multimedia)
qt_standard_project_setup()

# --- ONNX Runtime (Step 3) ------------------------------------------------
# Microsoft ships a prebuilt CPU package as a zip. Extract it somewhere stable
# and either keep the default path below or override it on the command line:
#   cmake -DONNXRUNTIME_DIR=D:/whatever ...
set(ONNXRUNTIME_DIR "C:/Tools/onnxruntime" CACHE PATH
    "Root of the extracted onnxruntime-win-x64-*.zip (must contain include/ and lib/)")

find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED)

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "onnxruntime_cxx_api.h not found under ${ONNXRUNTIME_DIR}/include. "
        "Check ONNXRUNTIME_DIR (see SETUP_TOMORROW.md).")
endif()

message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")

add_subdirectory(src)
