#include "app/MainWindow.h"

#include "app/InferenceController.h"
#include "media/VideoController.h"
#include "ui/ImageView.h"
#include "ui/SettingsDialog.h"
#include "ui/VideoControls.h"

#include <QAction>
#include <QApplication>
#include <QDebug>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QImage>
#include <QImageReader>
#include <QKeySequence>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QSplitter>
#include <QStatusBar>
#include <QString>
#include <QVBoxLayout>
#include <QWidget>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_originalView(new ImageView)         // parents set inside buildCentralWidget
    , m_processedView(new ImageView)
    , m_controls(new VideoControls)
    , m_controller(new VideoController(this))
    , m_inference(new InferenceController(this))  // builds its own backend from saved settings
{
    setWindowTitle(tr("InferenceVisualizer"));
    resize(1200, 720);

    buildCentralWidget();
    buildMenus();
    wireController();
    tryAutoLoadModel();   // Step 3: auto-load model_tcc.onnx if present next to the .exe

    statusBar()->showMessage(tr("Ready."));
}

void MainWindow::tryAutoLoadModel()
{
    // The model lives next to the executable -- whatever path windeployqt /
    // CMake put it in. We don't accept CLI args; this is the only path.
    const QString modelPath = QDir(QApplication::applicationDirPath())
                                  .filePath(QStringLiteral("model_tcc.onnx"));

    if (!QFileInfo::exists(modelPath)) {
        qDebug() << "[MainWindow] auto-load: model not found at" << modelPath
                 << "-- use Inference -> Load Model to pick one manually";
        statusBar()->showMessage(
            tr("No model_tcc.onnx next to the .exe. "
               "Use Inference -> Load Model to pick one."),
            8000);
        return;
    }

    qDebug() << "[MainWindow] auto-loading model:" << modelPath;
    m_inference->loadModel(modelPath);

    // If the load succeeded, m_runInferenceAction was enabled by
    // onModelLoaded(). Auto-toggle it on so the user gets inference
    // immediately on the first frame they open.
    if (m_runInferenceAction && m_runInferenceAction->isEnabled()) {
        m_runInferenceAction->setChecked(true);
    }
}

void MainWindow::buildCentralWidget()
{
    auto *splitter = new QSplitter(Qt::Horizontal);
    splitter->addWidget(m_originalView);
    splitter->addWidget(m_processedView);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 1);
    splitter->setChildrenCollapsible(false);

    auto *container = new QWidget;
    auto *layout    = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(splitter, /*stretch=*/1);
    layout->addWidget(m_controls);

    setCentralWidget(container);
}

void MainWindow::buildMenus()
{
    // --- File ----------------------------------------------------------
    auto *fileMenu = menuBar()->addMenu(tr("&File"));

    auto *openImageAction = fileMenu->addAction(tr("&Open Image..."));
    openImageAction->setShortcut(QKeySequence::Open);
    connect(openImageAction, &QAction::triggered, this, &MainWindow::openImage);

    auto *openVideoAction = fileMenu->addAction(tr("Open &Video..."));
    openVideoAction->setShortcut(QKeySequence(tr("Ctrl+Shift+O")));
    connect(openVideoAction, &QAction::triggered, this, &MainWindow::openVideo);

    fileMenu->addSeparator();

    auto *exitAction = fileMenu->addAction(tr("E&xit"));
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, qApp, &QApplication::quit);

    // --- Inference -----------------------------------------------------
    auto *inferenceMenu = menuBar()->addMenu(tr("&Inference"));

    auto *loadModelAction = inferenceMenu->addAction(tr("&Load Model..."));
    loadModelAction->setShortcut(QKeySequence(tr("Ctrl+M")));
    connect(loadModelAction, &QAction::triggered, this, &MainWindow::openModelDialog);

    m_runInferenceAction = inferenceMenu->addAction(tr("&Run Inference"));
    m_runInferenceAction->setCheckable(true);
    m_runInferenceAction->setChecked(false);
    m_runInferenceAction->setEnabled(false);  // gets enabled once a model loads
    m_runInferenceAction->setShortcut(QKeySequence(tr("Ctrl+R")));
    connect(m_runInferenceAction, &QAction::toggled,
            m_inference, &InferenceController::setEnabled);

    inferenceMenu->addSeparator();

    auto *settingsAction = inferenceMenu->addAction(tr("&Settings..."));
    settingsAction->setShortcut(QKeySequence(tr("Ctrl+,")));  // conventional Settings shortcut
    connect(settingsAction, &QAction::triggered, this, &MainWindow::openSettingsDialog);
}

void MainWindow::wireController()
{
    // Controller -> original view: every decoded frame goes here directly.
    connect(m_controller, &VideoController::frameReady,
            m_originalView, &ImageView::setImage);

    // Controller -> inference -> processed view. The inference controller
    // emits the frame unchanged when inference is disabled, so the right
    // pane keeps mirroring the left in passthrough mode.
    connect(m_controller, &VideoController::frameReady,
            m_inference,  &InferenceController::processFrame);
    connect(m_inference,  &InferenceController::processedFrameReady,
            m_processedView, &ImageView::setImage);

    // Controller -> controls: keep the slider / time label in sync.
    connect(m_controller, &VideoController::durationChanged,
            m_controls,   &VideoControls::setDuration);
    connect(m_controller, &VideoController::positionChanged,
            m_controls,   &VideoControls::setPosition);

    // Playback state -> button glyph + status bar.
    connect(m_controller, &VideoController::playbackStateChanged,
            this,         &MainWindow::onPlaybackStateChanged);

    // Errors -> dialog
    connect(m_controller, &VideoController::errorOccurred,
            this,         &MainWindow::onMediaError);

    // Controls -> controller: user intent.
    connect(m_controls, &VideoControls::playPauseClicked,
            m_controller, &VideoController::togglePlayPause);
    connect(m_controls, &VideoControls::seekRequested,
            m_controller, &VideoController::seek);

    // Inference -> UI feedback
    connect(m_inference, &InferenceController::modelLoaded,
            this,        &MainWindow::onModelLoaded);
    connect(m_inference, &InferenceController::modelLoadFailed,
            this,        &MainWindow::onModelLoadFailed);
    connect(m_inference, &InferenceController::inferenceFailed,
            this,        &MainWindow::onInferenceFailed);

    // Inference backpressure -> video pause/resume. This is what gives us
    // the "process every frame" guarantee: when inference falls behind, we
    // pause the source until the queue drains. The video plays in bursts.
    connect(m_inference, &InferenceController::wantsSourcePaused,
            this,        &MainWindow::onInferenceWantsSourcePaused);
}

void MainWindow::openImage()
{
    QStringList patterns;
    for (const QByteArray &fmt : QImageReader::supportedImageFormats()) {
        patterns << QStringLiteral("*.%1").arg(QString::fromLatin1(fmt));
    }
    const QString filter = tr("Images (%1);;All files (*)").arg(patterns.join(' '));

    const QString path = QFileDialog::getOpenFileName(
        this, tr("Open Image"), QString(), filter);
    if (path.isEmpty()) return;

    QImageReader reader(path);
    reader.setAutoTransform(true);
    const QImage image = reader.read();

    if (image.isNull()) {
        QMessageBox::warning(this, tr("Could not open image"),
            tr("Failed to load \"%1\":\n%2").arg(path, reader.errorString()));
        statusBar()->showMessage(tr("Failed to open %1").arg(path), 5000);
        return;
    }

    // Opening a still image stops any video playback and clears the controls.
    m_controller->stop();
    m_controls->setEnabledControls(false);

    m_lastStill = image;   // remembered so settings changes can re-preview it
    m_originalView->setImage(image);
    // Route the still image through inference too -- the controller emits
    // a passthrough QImage when inference is disabled.
    m_inference->processFrame(image);

    statusBar()->showMessage(
        tr("Loaded %1 (%2x%3)").arg(path).arg(image.width()).arg(image.height()),
        5000);
}

void MainWindow::openVideo()
{
    const QString filter = tr("Videos (*.mp4 *.mov *.avi *.mkv *.webm *.wmv);;All files (*)");
    const QString path = QFileDialog::getOpenFileName(
        this, tr("Open Video"), QString(), filter);
    if (path.isEmpty()) return;

    if (!m_controller->openFile(path)) {
        QMessageBox::warning(this, tr("Could not open video"),
                             tr("Empty file path."));
        return;
    }

    m_lastStill = QImage();   // a video is now the source, not a still
    m_controls->setEnabledControls(true);
    m_controller->play();
    statusBar()->showMessage(tr("Playing %1").arg(path), 5000);
}

void MainWindow::openModelDialog()
{
    const QString path = QFileDialog::getOpenFileName(
        this,
        tr("Load ONNX Model"),
        QString(),
        tr("ONNX models (*.onnx);;All files (*)"));
    if (path.isEmpty()) return;

    m_inference->loadModel(path);
}

void MainWindow::openSettingsDialog()
{
    SettingsDialog dialog(m_inference->currentSettings(),
                          m_inference->currentModelType(),
                          this);
    // Live preview: apply immediately on the dialog's Apply/OK, even before
    // it closes.
    connect(&dialog, &SettingsDialog::applied,
            this,    &MainWindow::onSettingsApplied);
    dialog.exec();   // modal; connection dies with the dialog
}

void MainWindow::onSettingsApplied(const InferenceSettings &settings,
                                   ModelType modelType)
{
    // Order matters: set the model type first (this may rebuild + reload the
    // backend), then push the latest knob values onto whatever backend is now
    // active.
    m_inference->setModelType(modelType);
    m_inference->applySettings(settings);

    // Re-run inference on the current still so the processed pane reflects the
    // change without the user reopening the file. For video, the next decoded
    // frame already picks up the new settings.
    if (!m_lastStill.isNull()) {
        m_inference->processFrame(m_lastStill);
    }
}

void MainWindow::onPlaybackStateChanged(QMediaPlayer::PlaybackState state)
{
    const bool playing = (state == QMediaPlayer::PlayingState);
    m_controls->setPlaying(playing);

    switch (state) {
        case QMediaPlayer::PlayingState:
            statusBar()->showMessage(tr("Playing"), 2000);
            break;
        case QMediaPlayer::PausedState:
            statusBar()->showMessage(tr("Paused"), 2000);
            break;
        case QMediaPlayer::StoppedState:
            statusBar()->showMessage(tr("Stopped"), 2000);
            break;
    }
}

void MainWindow::onMediaError(const QString &message)
{
    m_controls->setEnabledControls(false);
    QMessageBox::warning(this, tr("Playback error"), message);
}

void MainWindow::onModelLoaded(const QString &path)
{
    const QString shortName = QFileInfo(path).fileName();
    statusBar()->showMessage(tr("Loaded: %1").arg(shortName), 5000);

    if (m_runInferenceAction) {
        m_runInferenceAction->setEnabled(true);
    }
}

void MainWindow::onModelLoadFailed(const QString &message)
{
    QMessageBox::warning(this, tr("Could not load model"), message);
    statusBar()->showMessage(tr("Model load failed"), 5000);

    if (m_runInferenceAction) {
        m_runInferenceAction->setChecked(false);
        m_runInferenceAction->setEnabled(false);
    }
}

void MainWindow::onInferenceFailed(const QString &message)
{
    // Soft-fail: surface to status bar, don't pop a modal on every frame.
    statusBar()->showMessage(tr("Inference failed: %1").arg(message), 4000);
}

void MainWindow::onInferenceWantsSourcePaused(bool paused)
{
    // We only act if WE initiated the most recent pause. If the user paused
    // manually, we leave their pause alone -- and we don't auto-resume what
    // they explicitly stopped.
    if (paused) {
        // Inference is falling behind: pause the player IF it's playing.
        if (m_controller->playbackState() == QMediaPlayer::PlayingState) {
            m_controller->pause();
            m_pausedByBackpressure = true;
        }
    } else {
        // Inference has caught up: resume only if WE paused it.
        if (m_pausedByBackpressure) {
            m_pausedByBackpressure = false;
            m_controller->play();
        }
    }
}
