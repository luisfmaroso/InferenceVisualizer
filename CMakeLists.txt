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
# Microsoft ships prebuilt CPU packages as zip/tgz. Extract them somewhere
# stable. The defaults differ per OS; override with -DONNXRUNTIME_DIR=... on
# the configure line if you put it elsewhere.
if(WIN32)
    set(ONNXRUNTIME_DIR_DEFAULT "C:/Tools/onnxruntime"
        CACHE INTERNAL "")
else()
    # Linux / macOS: $HOME/onnxruntime (no root needed, gitignored).
    set(ONNXRUNTIME_DIR_DEFAULT "$ENV{HOME}/onnxruntime"
        CACHE INTERNAL "")
endif()
set(ONNXRUNTIME_DIR "${ONNXRUNTIME_DIR_DEFAULT}" CACHE PATH
    "Root of the extracted onnxruntime-*-x64-*.zip/tgz (must contain include/ and lib/)")

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
