#pragma once

#include "inference/InferenceSettings.h"

#include <QImage>
#include <QMainWindow>
#include <QMediaPlayer>

class ImageView;
class VideoControls;
class VideoController;
class InferenceController;
class QAction;

// MainWindow: top-level window. Knows about menus, layout, and the wiring
// between the video controller, the inference controller, the two image
// views, and the video controls. It does NOT decode video frames itself,
// does NOT run inference itself, and does NOT know what kind of inference
// the backend performs.
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void openImage();
    void openVideo();
    void openModelDialog();
    void openSettingsDialog();
    void onSettingsApplied(const InferenceSettings &settings, ModelType modelType);
    void onPlaybackStateChanged(QMediaPlayer::PlaybackState state);
    void onMediaError(const QString &message);
    void onModelLoaded(const QString &path);
    void onModelLoadFailed(const QString &message);
    void onInferenceFailed(const QString &message);
    void onInferenceWantsSourcePaused(bool paused);  // backpressure from inference

private:
    void buildMenus();
    void buildCentralWidget();
    void wireController();
    void tryAutoLoadModel();   // Step 3: load model_tcc.onnx next to the .exe at startup

    // UI
    ImageView     *m_originalView;
    ImageView     *m_processedView;
    VideoControls *m_controls;

    // Engines
    VideoController     *m_controller;
    InferenceController *m_inference;

    // Inference menu actions we hold onto because we update their state
    // (e.g. disable "Run Inference" until a model is loaded).
    QAction *m_runInferenceAction = nullptr;

    // Tracks whether the video player is currently paused by us (because
    // inference has backpressure) vs. by the user. We only auto-resume
    // pauses WE initiated; the user's pauses are honoured.
    bool     m_pausedByBackpressure = false;

    // The most recent still image opened via File -> Open Image. Re-fed to the
    // inference controller after a settings change so the processed pane
    // updates live without reopening the file. Null while a video is active.
    QImage   m_lastStill;
};
