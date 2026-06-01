#include "app/InferenceController.h"
#include "inference/IInferenceBackend.h"
#include "inference/OnnxUnetBackend.h"
#include "inference/OnnxYoloSegBackend.h"

#include <QColor>
#include <QDebug>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QMetaObject>
#include <QSettings>

#include <utility>

InferenceController::InferenceController(QObject *parent)
    : QObject(parent)
{
    loadPersistedSettings();                  // QSettings -> m_settings / m_modelType
    m_backend = makeBackend(m_modelType);
    if (m_backend) {
        m_backend->applySettings(m_settings); // seed the fresh backend
    }
}

InferenceController::~InferenceController() = default;

std::unique_ptr<IInferenceBackend> InferenceController::makeBackend(ModelType type)
{
    switch (type) {
        case ModelType::Unet:
            return std::make_unique<OnnxUnetBackend>();
        case ModelType::YoloSeg:
        default:
            return std::make_unique<OnnxYoloSegBackend>();
    }
}

bool InferenceController::isReady() const
{
    return m_backend && m_backend->isReady();
}

void InferenceController::setEnabled(bool enabled)
{
    if (m_enabled == enabled) {
        return;
    }
    m_enabled = enabled;
    qDebug() << "[InferenceController] enabled =" << enabled;
    emit enabledChanged(enabled);
}

void InferenceController::loadModel(const QString &path)
{
    if (!m_backend) {
        emit modelLoadFailed(tr("No backend instance."));
        return;
    }
    qDebug() << "[InferenceController] loading model:" << path;
    if (m_backend->loadModel(path)) {
        m_modelPath = path;
        emit modelLoaded(path);
    } else {
        m_modelPath.clear();
        emit modelLoadFailed(m_backend->errorString());
    }
}

void InferenceController::applySettings(const InferenceSettings &settings)
{
    m_settings = settings;
    if (m_backend) {
        m_backend->applySettings(m_settings);
    }
    persistSettings();
}

void InferenceController::setModelType(ModelType type)
{
    if (type == m_modelType && m_backend) {
        return;
    }
    m_modelType = type;
    qDebug() << "[InferenceController] model type ->"
             << (type == ModelType::Unet ? "UNet" : "YOLO-seg");

    // Swap in a fresh backend of the new type and re-seed it with current
    // settings. The IInferenceBackend interface is what makes this a clean
    // one-liner -- nothing here knows the concrete class.
    m_backend = makeBackend(m_modelType);
    if (m_backend) {
        m_backend->applySettings(m_settings);
    }
    persistSettings();

    // Re-load the current model file into the new backend. The same .onnx may
    // or may not be valid for the new family -- if not, the existing
    // modelLoadFailed path surfaces the error in the UI (no crash).
    if (!m_modelPath.isEmpty()) {
        loadModel(m_modelPath);
    }
}

void InferenceController::loadPersistedSettings()
{
    QSettings s;  // org + app name set in main.cpp give this a stable location
    s.beginGroup(QStringLiteral("inference"));

    m_settings.overlayOpacity =
        s.value(QStringLiteral("overlayOpacity"), m_settings.overlayOpacity).toDouble();
    m_settings.confidenceThreshold = static_cast<float>(
        s.value(QStringLiteral("confidenceThreshold"),
                m_settings.confidenceThreshold).toDouble());

    const QColor c0 = s.value(QStringLiteral("classColor0"),
                              m_settings.classColors[0]).value<QColor>();
    const QColor c1 = s.value(QStringLiteral("classColor1"),
                              m_settings.classColors[1]).value<QColor>();
    if (c0.isValid()) m_settings.classColors[0] = c0;
    if (c1.isValid()) m_settings.classColors[1] = c1;

    const int typeInt = s.value(QStringLiteral("modelType"),
                                static_cast<int>(m_modelType)).toInt();
    m_modelType = (typeInt == static_cast<int>(ModelType::Unet))
                      ? ModelType::Unet : ModelType::YoloSeg;

    s.endGroup();
}

void InferenceController::persistSettings() const
{
    QSettings s;
    s.beginGroup(QStringLiteral("inference"));
    s.setValue(QStringLiteral("overlayOpacity"),      m_settings.overlayOpacity);
    s.setValue(QStringLiteral("confidenceThreshold"),
               static_cast<double>(m_settings.confidenceThreshold));
    s.setValue(QStringLiteral("classColor0"), m_settings.classColors[0]);
    s.setValue(QStringLiteral("classColor1"), m_settings.classColors[1]);
    s.setValue(QStringLiteral("modelType"), static_cast<int>(m_modelType));
    s.endGroup();
}

void InferenceController::processFrame(const QImage &input)
{
    // Passthrough cases: no backend, no model, or user toggled inference off.
    // In every passthrough case we still emit so the processed pane stays in
    // sync with the original pane (Step 2's dual-pane mirror behavior).
    if (!m_enabled || !isReady() || input.isNull()) {
        emit processedFrameReady(input);
        return;
    }

    // Append to FIFO -- every frame is preserved, processed in arrival order.
    m_frameQueue.enqueue(input);

    // Backpressure: when the queue grows past the high watermark, ask the
    // upstream source (the video player, via MainWindow) to pause. The
    // source will resume in runNextQueuedFrame when the queue has drained
    // back to the low watermark.
    if (!m_backpressured && m_frameQueue.size() >= m_highWatermark) {
        m_backpressured = true;
        qDebug() << "[InferenceController] backpressure ON  -- queue depth"
                 << m_frameQueue.size();
        emit wantsSourcePaused(true);
    }

    if (!m_busy) {
        m_busy = true;
        // Run the inference on the next event-loop tick. The crucial bit
        // is "next tick" -- it gives the event loop a chance to deliver
        // pending paint events (the original pane) BEFORE we block the
        // GUI thread on infer().
        QMetaObject::invokeMethod(this,
            &InferenceController::runNextQueuedFrame,
            Qt::QueuedConnection);
    }
}

void InferenceController::runNextQueuedFrame()
{
    // Defensive: state may have changed since this slot was queued.
    if (m_frameQueue.isEmpty() || !isReady() || !m_enabled) {
        m_busy = false;
        if (m_backpressured) {
            m_backpressured = false;
            qDebug() << "[InferenceController] backpressure OFF -- idle / disabled";
            emit wantsSourcePaused(false);
        }
        return;
    }

    QImage frame = m_frameQueue.dequeue();

    QElapsedTimer timer;
    timer.start();

    QImage result = m_backend->infer(frame);

    const qint64 elapsedMs = timer.elapsed();

    if (result.isNull()) {
        const QString err = m_backend->errorString();
        qWarning() << "[InferenceController] infer() failed:" << err;
        emit inferenceFailed(err);
        emit processedFrameReady(frame);   // passthrough on transient failure
    } else {
        emit processedFrameReady(result);
    }

    // Backpressure release: queue drained back to low watermark.
    if (m_backpressured && m_frameQueue.size() <= m_lowWatermark) {
        m_backpressured = false;
        qDebug() << "[InferenceController] backpressure OFF -- queue depth"
                 << m_frameQueue.size();
        emit wantsSourcePaused(false);
    }

    // Light-touch perf log -- first call, then every 10th, so steady-state
    // video doesn't drown the log. Tracks queue depth and total processed,
    // so "are we keeping up?" is answerable at a glance.
    static quint64 s_callCount = 0;
    if (s_callCount == 0 || (s_callCount % 10) == 0) {
        qDebug() << "[InferenceController] infer #" << s_callCount
                 << elapsedMs << "ms  queue depth:" << m_frameQueue.size()
                 << "  backpressure:" << m_backpressured;
    }
    ++s_callCount;

    // Either pull the next frame or go idle. Re-queue through the event
    // loop (NOT a tight while() loop) so paints and other events get a
    // chance to run between inferences.
    if (!m_frameQueue.isEmpty()) {
        QMetaObject::invokeMethod(this,
            &InferenceController::runNextQueuedFrame,
            Qt::QueuedConnection);
    } else {
        m_busy = false;
    }
}
