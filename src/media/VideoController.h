#pragma once

#include <QImage>
#include <QMediaPlayer>
#include <QObject>
#include <QString>

class QVideoSink;
class QVideoFrame;

// VideoController: owns a QMediaPlayer + QVideoSink and re-exports a small,
// stable API the rest of the app talks to.
//
// Why a controller object instead of using QMediaPlayer directly from
// MainWindow? Two reasons:
//   1. It hides Qt Multimedia behind our own signals. If we later replace the
//      backend with cv::VideoCapture or an inference pipeline that produces
//      frames, MainWindow doesn't change.
//   2. In step 3 the same frameReady() signal will fan out to the inference
//      backend in addition to the on-screen views. The controller becomes
//      the natural place to fork the pipeline.
class VideoController : public QObject
{
    Q_OBJECT

public:
    explicit VideoController(QObject *parent = nullptr);

    // Opens a video file. Returns false if the path is empty.
    // Actual decoder errors arrive later via errorOccurred().
    bool openFile(const QString &path);

    // Convenience accessors (mirror the underlying QMediaPlayer state).
    QMediaPlayer::PlaybackState playbackState() const;
    qint64 position() const;     // milliseconds
    qint64 duration() const;     // milliseconds
    bool   hasMedia() const;

public slots:
    void play();
    void pause();
    void togglePlayPause();
    void stop();
    void seek(qint64 milliseconds);

signals:
    // Emitted each time a new decoded frame is ready (GUI thread).
    void frameReady(const QImage &image);

    // Re-exports of QMediaPlayer state changes -- we forward these so that
    // anything that connects to VideoController doesn't need to know about
    // QMediaPlayer at all.
    void playbackStateChanged(QMediaPlayer::PlaybackState state);
    void positionChanged(qint64 milliseconds);
    void durationChanged(qint64 milliseconds);

    // A user-visible string error (e.g. "codec not supported").
    void errorOccurred(const QString &message);

private slots:
    void handleVideoFrame(const QVideoFrame &frame);
    void handlePlayerError(QMediaPlayer::Error error, const QString &message);
    void handleMediaStatus(QMediaPlayer::MediaStatus status);

private:
    QMediaPlayer *m_player;          // parented to this; auto-deleted
    QVideoSink   *m_sink;            // parented to this
    quint64       m_frameCount{0};   // diagnostic: counts frames we received
};
