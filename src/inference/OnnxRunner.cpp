#include "inference/OnnxRunner.h"

#include <QDebug>

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#  include <string>
#endif

// ---------------------------------------------------------------------------
// PIMPL: keep all Ort:: types out of the header so #include <onnxruntime_cxx_api.h>
// only fans out to this one .cpp.
// ---------------------------------------------------------------------------
struct OnnxRunner::Impl {
    Ort::Env                            env{ORT_LOGGING_LEVEL_WARNING,
                                            "InferenceVisualizer"};
    Ort::SessionOptions                 options;
    std::unique_ptr<Ort::Session>       session;
    Ort::AllocatorWithDefaultOptions    allocator;

    // We keep the most recent run's outputs alive here so the TensorViews
    // the backend gets back are valid until the next run() call.
    std::vector<Ort::Value>             lastOutputs;
};

OnnxRunner::OnnxRunner()
    : m_impl(std::make_unique<Impl>())
{
    // Common-sense defaults. ORT_ENABLE_EXTENDED turns on most optimisations
    // that don't change numerical output. ALL would include more aggressive
    // passes that occasionally trip on hand-crafted models.
    m_impl->options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    m_impl->options.SetIntraOpNumThreads(1);  // GUI thread today; step 4 = real threading
}

OnnxRunner::~OnnxRunner() = default;

bool OnnxRunner::isLoaded() const
{
    return m_impl && m_impl->session != nullptr;
}

bool OnnxRunner::loadModel(const QString &path)
{
    m_lastError.clear();
    m_inputNames.clear();
    m_outputNames.clear();

    try {
#ifdef _WIN32
        // ORT on Windows wants wchar_t paths.
        const std::wstring widePath = path.toStdWString();
        m_impl->session = std::make_unique<Ort::Session>(
            m_impl->env, widePath.c_str(), m_impl->options);
#else
        const std::string narrowPath = path.toStdString();
        m_impl->session = std::make_unique<Ort::Session>(
            m_impl->env, narrowPath.c_str(), m_impl->options);
#endif
    } catch (const Ort::Exception &e) {
        m_lastError = QStringLiteral("ONNX Runtime: ") + QString::fromUtf8(e.what());
        m_impl->session.reset();
        qWarning() << "[OnnxRunner] loadModel failed:" << m_lastError;
        return false;
    } catch (const std::exception &e) {
        m_lastError = QStringLiteral("Failed to load model: ") + QString::fromUtf8(e.what());
        m_impl->session.reset();
        qWarning() << "[OnnxRunner] loadModel failed:" << m_lastError;
        return false;
    }

    // Discover all input and output names from the graph. We support
    // single-input but any number of outputs.
    try {
        const size_t numInputs  = m_impl->session->GetInputCount();
        const size_t numOutputs = m_impl->session->GetOutputCount();
        if (numInputs < 1 || numOutputs < 1) {
            m_lastError = QStringLiteral(
                "Model has no inputs or outputs -- unusable.");
            m_impl->session.reset();
            return false;
        }
        if (numInputs > 1) {
            qWarning() << "[OnnxRunner] model has" << numInputs
                       << "inputs; only the first will be used.";
        }

        for (size_t i = 0; i < numInputs; ++i) {
            auto name = m_impl->session->GetInputNameAllocated(i, m_impl->allocator);
            m_inputNames.emplace_back(name.get());
        }
        for (size_t i = 0; i < numOutputs; ++i) {
            auto name = m_impl->session->GetOutputNameAllocated(i, m_impl->allocator);
            m_outputNames.emplace_back(name.get());
        }
    } catch (const Ort::Exception &e) {
        m_lastError = QStringLiteral("ORT name lookup: ") + QString::fromUtf8(e.what());
        m_impl->session.reset();
        return false;
    }

    QStringList outs;
    for (const auto &n : m_outputNames) outs << QString::fromStdString(n);
    qDebug() << "[OnnxRunner] loaded" << path
             << "input:"   << QString::fromStdString(m_inputNames.front())
             << "outputs:" << outs;
    return true;
}

std::vector<OnnxRunner::TensorView> OnnxRunner::run(
    const float *inputData,
    const std::vector<int64_t> &inputShape)
{
    m_lastError.clear();
    if (!isLoaded()) {
        m_lastError = QStringLiteral("Cannot run: no model loaded.");
        return {};
    }

    try {
        size_t numElements = 1;
        for (int64_t dim : inputShape) {
            if (dim <= 0) {
                m_lastError = QStringLiteral(
                    "Input shape contains a non-positive dimension. "
                    "If the model has a dynamic axis, give it a concrete value "
                    "before calling run().");
                return {};
            }
            numElements *= static_cast<size_t>(dim);
        }

        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            const_cast<float *>(inputData),  // ORT API takes non-const; doesn't write
            numElements,
            inputShape.data(),
            inputShape.size());

        // Build raw-pointer arrays of input/output names for Run().
        std::vector<const char *> inNames;
        std::vector<const char *> outNames;
        inNames.reserve(m_inputNames.size());
        outNames.reserve(m_outputNames.size());
        for (const auto &s : m_inputNames)  inNames.push_back(s.c_str());
        for (const auto &s : m_outputNames) outNames.push_back(s.c_str());

        m_impl->lastOutputs = m_impl->session->Run(
            Ort::RunOptions{nullptr},
            inNames.data(),  &inputTensor,           inNames.size(),
            outNames.data(), outNames.size());

        if (m_impl->lastOutputs.empty()) {
            m_lastError = QStringLiteral("Model returned no outputs.");
            return {};
        }

        std::vector<TensorView> views;
        views.reserve(m_impl->lastOutputs.size());
        for (auto &out : m_impl->lastOutputs) {
            if (!out.IsTensor()) {
                m_lastError = QStringLiteral("Non-tensor output not supported.");
                return {};
            }
            TensorView v;
            v.data  = out.GetTensorData<float>();
            v.shape = out.GetTensorTypeAndShapeInfo().GetShape();
            views.push_back(std::move(v));
        }
        return views;
    } catch (const Ort::Exception &e) {
        m_lastError = QStringLiteral("ORT run failed: ") + QString::fromUtf8(e.what());
        return {};
    } catch (const std::exception &e) {
        m_lastError = QStringLiteral("run() exception: ") + QString::fromUtf8(e.what());
        return {};
    }
}
