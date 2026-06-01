#pragma once

#include "inference/InferenceSettings.h"

#include <QImage>
#include <QString>

// IInferenceBackend: the abstract interface every inference backend implements.
// `InferenceController` owns one of these via std::unique_ptr and never knows
// which concrete backend it has -- segmentation, detection, classification,
// they all expose the same shape.
//
// Why an interface for one implementation today?
//   1. Step 4 (or beyond) can plug in a detection backend without changing
//      InferenceController or MainWindow.
//   2. Each backend takes responsibility for the things it alone knows how to
//      do (mask compositing, box drawing) -- the controller stays generic.
//   3. The interface is the contract: write a new backend, satisfy the four
//      virtuals, and it slots in.
class IInferenceBackend
{
public:
    virtual ~IInferenceBackend() = default;

    // Load the ONNX model at `path`. Returns true on success.
    // On failure, `errorString()` carries a human-readable reason.
    virtual bool loadModel(const QString &path) = 0;

    // True after a successful loadModel(), false otherwise.
    virtual bool isReady() const = 0;

    // The last error from loadModel() or infer(), empty if none.
    virtual QString errorString() const = 0;

    // Run inference on `input`. Returns a new QImage to display in the
    // processed pane -- usually a copy of `input` with overlays drawn on top
    // (segmentation mask, detection boxes, etc.). If isReady() is false or
    // anything goes wrong, returns a null QImage and sets errorString().
    virtual QImage infer(const QImage &input) = 0;

    // Apply the user-tunable display/inference settings. Called whenever the
    // Settings dialog changes something, and once right after construction so
    // a freshly-swapped backend starts from the current settings. Each backend
    // maps the fields it cares about onto its own internal config and may
    // ignore the rest.
    virtual void applySettings(const InferenceSettings &settings) = 0;

protected:
    IInferenceBackend() = default;

    // Non-copyable -- backends own GPU/CPU resources that don't survive copies.
    IInferenceBackend(const IInferenceBackend &) = delete;
    IInferenceBackend &operator=(const IInferenceBackend &) = delete;
};
