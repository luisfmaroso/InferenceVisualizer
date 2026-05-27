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

# --- Qt auto-detection -------------------------------------------------------
# The Qt Installer puts Qt under ~/Qt on Linux/macOS and C:/Qt on Windows.
# We glob for the newest 6.x.x subdir so the version doesn't need to be
# hardcoded. Override the root with -DQT_ROOT=... or the compiler subdir with
# -DQT_COMPILER_SUBDIR=... if your layout differs (e.g. mingw_64 on Windows).
if(WIN32)
    set(QT_ROOT "C:/Qt" CACHE PATH "Root Qt Installer directory")
    set(QT_COMPILER_SUBDIR "mingw_64" CACHE STRING "Qt compiler subdirectory")
else()
    set(QT_ROOT "$ENV{HOME}/Qt" CACHE PATH "Root Qt Installer directory")
    set(QT_COMPILER_SUBDIR "gcc_64" CACHE STRING "Qt compiler subdirectory")
endif()

file(GLOB _qt_versions LIST_DIRECTORIES true "${QT_ROOT}/6.*")
list(SORT _qt_versions COMPARE NATURAL ORDER DESCENDING)
if(_qt_versions)
    list(GET _qt_versions 0 _qt_latest)
    list(APPEND CMAKE_PREFIX_PATH "${_qt_latest}/${QT_COMPILER_SUBDIR}")
    message(STATUS "Qt auto-detected: ${_qt_latest}/${QT_COMPILER_SUBDIR}")
endif()

find_package(Qt6 6.5 REQUIRED COMPONENTS Widgets Multimedia)
qt_standard_project_setup()

# --- ONNX Runtime (Step 3) ------------------------------------------------
# Creates an OnnxRuntime INTERFACE target so src/ links the same way on
# every platform, regardless of how the library was installed.
#
# Windows : prebuilt zip from Microsoft, extracted to C:/Tools/onnxruntime.
#           Override with -DONNXRUNTIME_DIR=<path> if you put it elsewhere.
# Linux/macOS : system package (sudo apt install libonnxruntime-dev).
#               find_package picks it up automatically via its CMake config.
if(WIN32)
    set(ONNXRUNTIME_DIR "C:/Tools/onnxruntime" CACHE PATH
        "Root of the extracted onnxruntime-*-x64-*.zip (must contain include/ and lib/)")

    find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS "${ONNXRUNTIME_DIR}/lib"
        NO_DEFAULT_PATH
        REQUIRED)

    if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR
            "onnxruntime_cxx_api.h not found under ${ONNXRUNTIME_DIR}/include. "
            "Set ONNXRUNTIME_DIR to the extracted zip root.")
    endif()

    add_library(OnnxRuntime INTERFACE)
    target_include_directories(OnnxRuntime INTERFACE "${ONNXRUNTIME_DIR}/include")
    target_link_libraries(OnnxRuntime INTERFACE "${ONNXRUNTIME_LIB}")
    message(STATUS "ONNX Runtime (tarball): ${ONNXRUNTIME_LIB}")
else()
    find_package(onnxruntime CONFIG REQUIRED)
    add_library(OnnxRuntime INTERFACE)
    target_link_libraries(OnnxRuntime INTERFACE onnxruntime::onnxruntime)
    message(STATUS "ONNX Runtime (system): ${onnxruntime_VERSION}")
endif()

add_subdirectory(src)
