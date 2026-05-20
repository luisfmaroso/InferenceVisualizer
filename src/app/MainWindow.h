#pragma once

#include <QMainWindow>
#include <QMediaPlayer>

class ImageView;
class VideoControls;
class VideoController;

// MainWindow: top-level window. Knows about menus, layout, and the wiring
// between the video controller, the two image views, and the video controls.
// It does NOT decode video frames itself, and it will not run inference
// (that wires in at step 3).
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void openImage();
    void openVideo();
    void onPlaybackStateChanged(QMediaPlayer::PlaybackState state);
    void onMediaError(const QString &message);

private:
    void buildMenus();
    void buildCentralWidget();
    void wireController();

    // UI
    ImageView     *m_originalView;
    ImageView     *m_processedView;
    VideoControls *m_controls;

    // Media
    VideoController *m_controller;
};
