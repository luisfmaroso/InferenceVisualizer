qt_add_executable(InferenceVisualizer
    WIN32
    main.cpp
)

target_sources(InferenceVisualizer PRIVATE
    app/MainWindow.h
    app/MainWindow.cpp
    app/InferenceController.h
    app/InferenceController.cpp
    media/VideoController.h
    media/VideoController.cpp
    inference/IInferenceBackend.h
    inference/OnnxRunner.h
    inference/OnnxRunner.cpp
    inference/OnnxSegmentationBackend.h
    inference/OnnxSegmentationBackend.cpp
    ui/ImageView.h
    ui/ImageView.cpp
    ui/VideoControls.h
    ui/VideoControls.cpp
)

target_include_directories(InferenceVisualizer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${ONNXRUNTIME_DIR}/include"
)

target_link_libraries(InferenceVisualizer PRIVATE
    Qt6::Widgets
    Qt6::Multimedia
    "${ONNXRUNTIME_LIB}"
)

# On Windows, copy onnxruntime.dll next to the built .exe so the loader
# finds it at launch. Without this you'd hit STATUS_DLL_NOT_FOUND
# (0xc0000135) -- same loader issue we documented in step 2 for Qt's DLLs.
#
# On Linux/macOS the equivalent (libonnxruntime.so / .dylib) is discovered
# via LD_LIBRARY_PATH / DYLD_LIBRARY_PATH, which the .vscode/launch.json
# Linux configuration sets to point at ${HOME}/onnxruntime/lib. No copy step
# needed (and copying a .so wouldn't help anyway -- the application
# directory isn't on the default Linux loader search path).
if(WIN32)
    add_custom_command(TARGET InferenceVisualizer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:InferenceVisualizer>"
        COMMENT "Copying onnxruntime.dll next to InferenceVisualizer.exe")
endif()

if(MSVC)
    target_compile_options(InferenceVisualizer PRIVATE /W4 /permissive-)
else()
    target_compile_options(InferenceVisualizer PRIVATE -Wall -Wextra -Wpedantic)
endif()
