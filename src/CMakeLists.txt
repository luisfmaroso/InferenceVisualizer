qt_add_executable(InferenceVisualizer
    WIN32
    main.cpp
)

target_sources(InferenceVisualizer PRIVATE
    app/MainWindow.h
    app/MainWindow.cpp
    ui/ImageView.h
    ui/ImageView.cpp
)

target_include_directories(InferenceVisualizer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(InferenceVisualizer PRIVATE
    Qt6::Widgets
)

if(MSVC)
    target_compile_options(InferenceVisualizer PRIVATE /W4 /permissive-)
else()
    target_compile_options(InferenceVisualizer PRIVATE -Wall -Wextra -Wpedantic)
endif()
