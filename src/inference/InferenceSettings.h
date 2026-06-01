#pragma once

#include <QColor>

#include <array>

// InferenceSettings: the small bundle of display/tuning knobs the Settings
// dialog edits and the InferenceController forwards to whichever backend is
// active. Keeping these in one POD struct (rather than as individual setters
// on IInferenceBackend) means the interface stays generic: a backend receives
// the whole struct via applySettings() and maps the fields it cares about onto
// its own internal config. A backend is free to ignore a field that doesn't
// apply to it (e.g. a future classification backend ignoring class colours).
struct InferenceSettings {
    // Alpha of the overlay composited over the original image. [0.0, 1.0].
    double overlayOpacity = 0.5;

    // YOLO: per-detection score floor. UNet: per-pixel softmax-probability
    // floor (pixels below it are left un-overlaid). [0.0, 1.0].
    float confidenceThreshold = 0.25f;

    // Per-class overlay colours. Index = class id. Two classes for now;
    // default green for class 0, red for class 1.
    std::array<QColor, 2> classColors = {
        QColor(0, 180, 0),    // class 0
        QColor(220, 20, 20),  // class 1
    };
};

// Which family of model is loaded. Selected in the Settings dialog; decides
// which concrete IInferenceBackend the InferenceController instantiates.
enum class ModelType {
    YoloSeg,  // YOLOv8 / YOLO11 instance segmentation (two outputs)
    Unet,     // UNet-style semantic segmentation (single [1,C,H,W] output)
};
