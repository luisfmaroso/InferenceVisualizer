#pragma once

#include <QString>

#include <memory>
#include <string>
#include <vector>

// Forward-declare the parts of ONNX Runtime's C++ API we touch in the header.
// The full <onnxruntime_cxx_api.h> pulls in a lot; we keep it isolated to the
// .cpp by holding the ORT-typed members behind a unique_ptr (PIMPL-style).
namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
}

// OnnxRunner: thin RAII wrapper around an Ort::Session.
//
// Why a separate class rather than letting each backend own its session
// directly? Because the boring parts (env lifetime, model loading, error
// translation, getting input/output names from the graph) are identical for
// every ONNX-based backend. A future detection backend will create its own
// OnnxRunner, pass the same float buffer through it, and parse the result
// differently -- so the shared mechanics live here.
//
// Supports models with a single input and any number of outputs (YOLOv8
// segmentation, for example, has two outputs: detections + mask prototypes).
class OnnxRunner
{
public:
    // A view into one output tensor. `data` points to memory owned by the
    // runner -- valid until the next run() call or until destruction.
    struct TensorView {
        const float         *data;
        std::vector<int64_t> shape;
    };

    OnnxRunner();
    ~OnnxRunner();

    OnnxRunner(const OnnxRunner &) = delete;
    OnnxRunner &operator=(const OnnxRunner &) = delete;

    // Loads the ONNX model at `path` (UTF-8). On Windows this is converted to
    // wchar_t internally, since ORT requires that on Win32.
    // Returns true on success; false sets lastError().
    bool loadModel(const QString &path);

    bool isLoaded() const;
    QString lastError() const { return m_lastError; }

    // Names of the model's inputs and outputs, as discovered after loadModel().
    // Empty until a successful load.
    const std::vector<std::string> &inputNames()  const { return m_inputNames; }
    const std::vector<std::string> &outputNames() const { return m_outputNames; }

    // Single-input run, returning all outputs.
    //   inputData   - pointer to a contiguous float buffer
    //   inputShape  - tensor shape, e.g. {1, 3, 640, 640}
    // The returned TensorViews point into memory owned by this runner; they
    // are valid until the next run() call or until the runner is destroyed.
    //
    // Returns an empty vector on failure (check lastError()).
    std::vector<TensorView> run(const float *inputData,
                                 const std::vector<int64_t> &inputShape);

private:
    // PIMPL so the ORT C++ header doesn't leak through this file.
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    QString                  m_lastError;
};
