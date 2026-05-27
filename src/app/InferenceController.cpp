#include "app/InferenceController.h"
#include "inference/IInferenceBackend.h"

#include <QDebug>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QMetaObject>

#include <utility>

InferenceController::InferenceController(
    std::unique_ptr<IInferenceBackend> backend,
    QObject *parent)
    : QObject(parent)
    , m_backend(std::move(backend))
{
    // The backend has its own default opacity. Make sure it matches whatever
    // value our UI is going to show as the initial position.
    if (m_backend) {
        m_backend->setOverlayOpacity(0.5);
    }
}

InferenceController::~InferenceController() = default;

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

void InferenceController::setOverlayOpacity(double opacity)
{
    if (m_backend) {
        m_backend->setOverlayOpacity(opacity);
    }
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
