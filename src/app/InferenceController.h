#pragma once

#include "inference/InferenceSettings.h"

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
// concrete type -- swapping YOLO <-> UNet is a single makeBackend() call.
//
// Behaviour summary:
//   - When inference is disabled OR no model is loaded: processFrame emits
//     processedFrameReady(input) unchanged. This keeps Step 2's "right pane
//     mirrors left" behavior working as a passthrough.
//   - When inference is enabled AND a model is loaded: processFrame queues the
//     frame (FIFO) and processes in order, with watermark backpressure.
//
// Settings (opacity, confidence, class colours) and the active model type are
// loaded from QSettings at construction and re-persisted whenever they change.
class InferenceController : public QObject
{
    Q_OBJECT

public:
    explicit InferenceController(QObject *parent = nullptr);
    ~InferenceController() override;

    bool isEnabled() const { return m_enabled; }
    bool isReady()   const;                           // model loaded?
    QString currentModelPath() const { return m_modelPath; }

    // Seed values for the Settings dialog.
    InferenceSettings currentSettings()  const { return m_settings; }
    ModelType         currentModelType() const { return m_modelType; }

public slots:
    void processFrame(const QImage &input);

    void setEnabled(bool enabled);
    void loadModel(const QString &path);

    // Settings dialog -> here. applySettings forwards to the active backend and
    // persists. setModelType swaps the backend, re-applies settings, and
    // reloads the current model file (if any).
    void applySettings(const InferenceSettings &settings);
    void setModelType(ModelType type);

private slots:
    void runNextQueuedFrame();

signals:
    void processedFrameReady(const QImage &image);
    void wantsSourcePaused(bool paused);

    void enabledChanged(bool enabled);
    void modelLoaded(const QString &path);
    void modelLoadFailed(const QString &errorMessage);
    void inferenceFailed(const QString &errorMessage);

private:
    // Build a fresh backend of the given type (no model loaded yet).
    static std::unique_ptr<IInferenceBackend> makeBackend(ModelType type);

    void loadPersistedSettings();   // QSettings -> m_settings / m_modelType
    void persistSettings() const;   // m_settings / m_modelType -> QSettings

    std::unique_ptr<IInferenceBackend> m_backend;
    QString                            m_modelPath;
    bool                               m_enabled = false;

    InferenceSettings                  m_settings;
    ModelType                          m_modelType = ModelType::YoloSeg;

    // FIFO queue: every frame is preserved, processed in arrival order.
    QQueue<QImage>                     m_frameQueue;
    bool                               m_busy = false;
    bool                               m_backpressured = false;
    int                                m_highWatermark = 2;
    int                                m_lowWatermark  = 0;
};
