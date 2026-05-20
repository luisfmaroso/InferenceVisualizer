#include "media/VideoController.h"

#include <QDebug>
#include <QFileInfo>
#include <QMediaMetaData>
#include <QUrl>
#include <QVideoFrame>
#include <QVideoSink>

VideoController::VideoController(QObject *parent)
    : QObject(parent)
    , m_player(new QMediaPlayer(this))
    , m_sink(new QVideoSink(this))
{
    m_player->setVideoSink(m_sink);

    // Frames arrive here whenever the decoder produces a new one.
    connect(m_sink, &QVideoSink::videoFrameChanged,
            this,   &VideoController::handleVideoFrame);

    // Re-emit player signals through our own API. Signal-to-signal connects
    // are valid Qt and create a clean "facade" layer: nothing outside this
    // file knows about QMediaPlayer.
    connect(m_player, &QMediaPlayer::playbackStateChanged,
            this,     &VideoController::playbackStateChanged);
    connect(m_player, &QMediaPlayer::positionChanged,
            this,     &VideoController::positionChanged);
    connect(m_player, &QMediaPlayer::durationChanged,
            this,     &VideoController::durationChanged);

    connect(m_player, &QMediaPlayer::errorOccurred,
            this,     &VideoController::handlePlayerError);

    // Diagnostic: media-status transitions tell us loading / buffering / invalid.
    connect(m_player, &QMediaPlayer::mediaStatusChanged,
            this,     &VideoController::handleMediaStatus);

    qDebug() << "[VideoController] active media backend ="
             << qgetenv("QT_MEDIA_BACKEND")
             << "(empty means Qt's default)";
}

bool VideoController::openFile(const QString &path)
{
    if (path.isEmpty()) {
        return false;
    }
    qDebug() << "[VideoController] openFile:" << path
             << "exists=" << QFileInfo::exists(path);
    m_player->stop();
    m_frameCount = 0;
    m_player->setSource(QUrl::fromLocalFile(path));
    return true;
}

QMediaPlayer::PlaybackState VideoController::playbackState() const
{
    return m_player->playbackState();
}

qint64 VideoController::position() const { return m_player->position(); }
qint64 VideoController::duration() const { return m_player->duration(); }
bool   VideoController::hasMedia() const { return !m_player->source().isEmpty(); }

void VideoController::play()             { m_player->play(); }
void VideoController::pause()            { m_player->pause(); }
void VideoController::stop()             { m_player->stop(); }
void VideoController::seek(qint64 ms)    { m_player->setPosition(ms); }

void VideoController::togglePlayPause()
{
    if (m_player->playbackState() == QMediaPlayer::PlayingState) {
        m_player->pause();
    } else {
        m_player->play();
    }
}

void VideoController::handleVideoFrame(const QVideoFrame &frame)
{
    if (!frame.isValid()) {
        // Invalid frames happen at start/end-of-stream; not worth logging.
        return;
    }
    // toImage() does any necessary colour conversion (YUV -> RGB) and gives
    // us a CPU-side QImage we can hand to QLabel or to inference. It's an
    // O(width*height) operation; cheap enough for typical desktop video.
    const QImage image = frame.toImage();
    if (image.isNull()) {
        // Only warn the first time per session -- if it happens once it
        // tends to happen on every frame after, and we don't want the log
        // to drown.
        if (m_frameCount == 0) {
            qWarning() << "[VideoController] frame.toImage() returned null"
                       << "pixelFormat=" << frame.pixelFormat()
                       << "size=" << frame.size();
        }
        return;
    }
    if (m_frameCount == 0) {
        qDebug() << "[VideoController] first frame received"
                 << "size=" << image.size()
                 << "format=" << image.format();
    }
    ++m_frameCount;
    emit frameReady(image);
}

void VideoController::handlePlayerError(QMediaPlayer::Error error, const QString &message)
{
    qDebug() << "[VideoController] errorOccurred: code=" << error
             << "message=" << message;
    if (error == QMediaPlayer::NoError) {
        return;
    }
    // Forward a user-friendly string. MainWindow shows it in a QMessageBox.
    emit errorOccurred(message.isEmpty() ? tr("Unknown media error") : message);
}

void VideoController::handleMediaStatus(QMediaPlayer::MediaStatus status)
{
    // Status transitions are the most informative signal Qt Multimedia gives
    // us. NoMedia -> LoadingMedia -> LoadedMedia -> BufferingMedia ->
    // BufferedMedia is the happy path. InvalidMedia means the backend failed
    // to decode the container or codec.
    const char *name = "?";
    switch (status) {
        case QMediaPlayer::NoMedia:        name = "NoMedia";        break;
        case QMediaPlayer::LoadingMedia:   name = "LoadingMedia";   break;
        case QMediaPlayer::LoadedMedia:    name = "LoadedMedia";    break;
        case QMediaPlayer::StalledMedia:   name = "StalledMedia";   break;
        case QMediaPlayer::BufferingMedia: name = "BufferingMedia"; break;
        case QMediaPlayer::BufferedMedia:  name = "BufferedMedia";  break;
        case QMediaPlayer::EndOfMedia:     name = "EndOfMedia";     break;
        case QMediaPlayer::InvalidMedia:   name = "InvalidMedia";   break;
    }
    qDebug() << "[VideoController] mediaStatus =" << name
             << "hasVideo=" << m_player->hasVideo()
             << "duration=" << m_player->duration() << "ms";

    if (status == QMediaPlayer::InvalidMedia) {
        emit errorOccurred(tr("The media backend reported InvalidMedia. "
                              "Codec or container unsupported, or the file is corrupt."));
    }
}
