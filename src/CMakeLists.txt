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
    inference/InferenceSettings.h
    inference/OnnxRunner.h
    inference/OnnxRunner.cpp
    inference/OnnxYoloSegBackend.h
    inference/OnnxYoloSegBackend.cpp
    inference/OnnxUnetBackend.h
    inference/OnnxUnetBackend.cpp
    ui/ImageView.h
    ui/ImageView.cpp
    ui/VideoControls.h
    ui/VideoControls.cpp
    ui/SettingsDialog.h
    ui/SettingsDialog.cpp
)

target_include_directories(InferenceVisualizer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(InferenceVisualizer PRIVATE
    Qt6::Widgets
    Qt6::Multimedia
    OnnxRuntime
)

# On Windows, copy onnxruntime.dll next to the built .exe so the loader
# finds it at launch. Without this you'd hit STATUS_DLL_NOT_FOUND
# (0xc0000135) -- same loader issue we documented in step 2 for Qt's DLLs.
#
# On Linux/macOS the system package installs libonnxruntime.so to a standard
# loader path, so no copy or LD_LIBRARY_PATH override is needed.
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
