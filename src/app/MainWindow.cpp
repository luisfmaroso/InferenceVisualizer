#include "app/MainWindow.h"

#include "media/VideoController.h"
#include "ui/ImageView.h"
#include "ui/VideoControls.h"

#include <QAction>
#include <QApplication>
#include <QFileDialog>
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
{
    setWindowTitle(tr("InferenceVisualizer"));
    resize(1200, 720);

    buildCentralWidget();
    buildMenus();
    wireController();

    statusBar()->showMessage(tr("Ready."));
}

void MainWindow::buildCentralWidget()
{
    auto *splitter = new QSplitter(Qt::Horizontal);
    splitter->addWidget(m_originalView);
    splitter->addWidget(m_processedView);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 1);
    splitter->setChildrenCollapsible(false); // prevents accidental zero-width pane

    auto *container = new QWidget;
    auto *layout    = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(splitter, /*stretch=*/1);
    layout->addWidget(m_controls);

    setCentralWidget(container); // QMainWindow takes ownership
}

void MainWindow::buildMenus()
{
    auto *fileMenu = menuBar()->addMenu(tr("&File"));

    auto *openImageAction = fileMenu->addAction(tr("&Open Image..."));
    openImageAction->setShortcut(QKeySequence::Open);          // Ctrl+O
    connect(openImageAction, &QAction::triggered, this, &MainWindow::openImage);

    auto *openVideoAction = fileMenu->addAction(tr("Open &Video..."));
    openVideoAction->setShortcut(QKeySequence(tr("Ctrl+Shift+O")));
    connect(openVideoAction, &QAction::triggered, this, &MainWindow::openVideo);

    fileMenu->addSeparator();

    auto *exitAction = fileMenu->addAction(tr("E&xit"));
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, qApp, &QApplication::quit);
}

void MainWindow::wireController()
{
    // Controller -> views: every decoded frame goes to BOTH views in step 2.
    // In step 3, we'll route the right pane through the inference backend
    // instead of duplicating the frame.
    connect(m_controller, &VideoController::frameReady,
            m_originalView,  &ImageView::setImage);
    connect(m_controller, &VideoController::frameReady,
            m_processedView, &ImageView::setImage);

    // Controller -> controls: keep the slider / time label in sync.
    connect(m_controller, &VideoController::durationChanged,
            m_controls,   &VideoControls::setDuration);
    connect(m_controller, &VideoController::positionChanged,
            m_controls,   &VideoControls::setPosition);

    // Playback state drives two things: the button glyph, and our status bar.
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

    m_originalView->setImage(image);
    m_processedView->setImage(image); // mirror until step 3 fills the right pane
    statusBar()->showMessage(
        tr("Loaded %1 (%2x%3)").arg(path).arg(image.width()).arg(image.height()),
        5000);
}

void MainWindow::openVideo()
{
    // We let the user pick anything; QMediaPlayer will report the real error
    // if the format isn't supported by the active backend.
    const QString filter = tr("Videos (*.mp4 *.mov *.avi *.mkv *.webm *.wmv);;All files (*)");
    const QString path = QFileDialog::getOpenFileName(
        this, tr("Open Video"), QString(), filter);
    if (path.isEmpty()) return;

    if (!m_controller->openFile(path)) {
        QMessageBox::warning(this, tr("Could not open video"),
                             tr("Empty file path."));
        return;
    }

    m_controls->setEnabledControls(true);
    m_controller->play();
    statusBar()->showMessage(tr("Playing %1").arg(path), 5000);
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
