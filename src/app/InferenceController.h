#pragma once

#include <QImage>
#include <QObject>
#include <QQueue>
#include <QString>

#include <memory>

class IInferenceBackend;

// InferenceController: orchestrates a backend.
//
// Lives in src/app/ rather than src/inference/ because it's GUI-coupled glue
// (slots that match MainWindow's wiring) rather than inference machinery.
// It owns an IInferenceBackend via std::unique_ptr and never looks at the
// concrete type -- swapping in a detection backend would be one line in
// MainWindow's constructor.
//
// Behaviour summary:
//   - When inference is disabled OR no model is loaded: processFrame emits
//     processedFrameReady(input) unchanged. This keeps Step 2's "right pane
//     mirrors left" behavior working as a passthrough.
//   - When inference is enabled AND a model is loaded: processFrame calls
//     backend->infer() and emits the resulting overlay.
class InferenceController : public QObject
{
    Q_OBJECT

public:
    explicit InferenceController(std::unique_ptr<IInferenceBackend> backend,
                                 QObject *parent = nullptr);
    ~InferenceController() override;

    bool isEnabled() const { return m_enabled; }
    bool isReady()   const;                           // model loaded?
    QString currentModelPath() const { return m_modelPath; }

public slots:
    // Receives a frame and emits processedFrameReady -- the central slot
    // MainWindow connects VideoController::frameReady (and openImage) into.
    //
    // Process-every-frame policy (Step 3 final): the frame is appended to
    // an in-order FIFO queue. No frames are ever dropped. When the queue
    // grows past a high watermark, we emit wantsSourcePaused(true) so the
    // upstream video source can hold off producing more until we catch up;
    // when it drains, we emit wantsSourcePaused(false).
    void processFrame(const QImage &input);

    void setEnabled(bool enabled);
    void loadModel(const QString &path);
    void setOverlayOpacity(double opacity);  // forwarded to backend

private slots:
    // Internal: pulled out of processFrame so it can be invoked via
    // Qt::QueuedConnection -- this is what gives the event loop a chance to
    // run between frames so the original pane can repaint.
    void runNextQueuedFrame();

signals:
    // The result for the right pane. Always fires -- when inference is
    // disabled or the model isn't ready, it carries `input` unchanged.
    void processedFrameReady(const QImage &image);

    // Backpressure to the upstream frame source. True means "stop producing
    // frames, we're behind"; false means "resume, queue has drained".
    // MainWindow forwards this to VideoController's pause/play.
    void wantsSourcePaused(bool paused);

    // Lifecycle and UI-feedback signals.
    void enabledChanged(bool enabled);
    void modelLoaded(const QString &path);
    void modelLoadFailed(const QString &errorMessage);
    void inferenceFailed(const QString &errorMessage);

private:
    std::unique_ptr<IInferenceBackend> m_backend;
    QString                            m_modelPath;
    bool                               m_enabled = false;

    // FIFO queue: every frame is preserved, processed in arrival order.
    QQueue<QImage>                     m_frameQueue;
    bool                               m_busy = false;            // a runNextQueuedFrame is queued/running
    bool                               m_backpressured = false;   // we've told the source to pause

    // Watermarks (in frames) for the backpressure hysteresis.
    //   m_highWatermark: queue depth at which we tell the source to PAUSE.
    //   m_lowWatermark : queue depth at or below which we tell it to RESUME.
    // Hysteresis avoids pause/resume thrashing every frame.
    int                                m_highWatermark = 2;
    int                                m_lowWatermark  = 0;
};
