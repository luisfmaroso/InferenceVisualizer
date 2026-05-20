#include "ui/VideoControls.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QStyle>
#include <QTime>

VideoControls::VideoControls(QWidget *parent)
    : QWidget(parent)
    , m_playPauseBtn(new QPushButton(this))
    , m_slider(new QSlider(Qt::Horizontal, this))
    , m_timeLabel(new QLabel(this))
{
    // Use the platform's standard play/pause icons -- no external assets.
    m_playPauseBtn->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    m_playPauseBtn->setToolTip(tr("Play / Pause (Space)"));
    m_playPauseBtn->setShortcut(Qt::Key_Space);

    m_slider->setRange(0, 0);
    m_slider->setSingleStep(1000);   // 1 second
    m_slider->setPageStep(5000);     // 5 seconds
    m_slider->setTracking(true);     // emit sliderMoved while dragging

    m_timeLabel->setText("--:-- / --:--");
    m_timeLabel->setMinimumWidth(110);

    auto *layout = new QHBoxLayout(this);
    layout->setContentsMargins(6, 4, 6, 4);
    layout->addWidget(m_playPauseBtn);
    layout->addWidget(m_slider, /*stretch=*/1);
    layout->addWidget(m_timeLabel);

    connect(m_playPauseBtn, &QPushButton::clicked,
            this,           &VideoControls::playPauseClicked);

    // Drag-vs-programmatic-update interlock: we want to know when the user
    // is actively dragging so we don't fight their finger.
    connect(m_slider, &QSlider::sliderPressed, this,
            [this]() { m_userIsDragging = true; });
    connect(m_slider, &QSlider::sliderMoved,
            this,     &VideoControls::onSliderMoved);
    connect(m_slider, &QSlider::sliderReleased,
            this,     &VideoControls::onSliderReleased);

    setEnabledControls(false);
}

void VideoControls::setEnabledControls(bool enabled)
{
    m_playPauseBtn->setEnabled(enabled);
    m_slider->setEnabled(enabled);
    if (!enabled) {
        m_slider->setValue(0);
        m_timeLabel->setText("--:-- / --:--");
    }
}

void VideoControls::setDuration(qint64 milliseconds)
{
    m_duration = milliseconds;
    m_slider->setRange(0, static_cast<int>(milliseconds));
    updateTimeLabel();
}

void VideoControls::setPosition(qint64 milliseconds)
{
    m_position = milliseconds;
    if (!m_userIsDragging) {
        // Only update the slider from outside when the user isn't holding it.
        m_slider->setValue(static_cast<int>(milliseconds));
    }
    updateTimeLabel();
}

void VideoControls::setPlaying(bool playing)
{
    m_playPauseBtn->setIcon(style()->standardIcon(
        playing ? QStyle::SP_MediaPause : QStyle::SP_MediaPlay));
}

void VideoControls::onSliderMoved(int value)
{
    // While dragging, show the prospective time. We don't seek every micro-
    // movement; the seek happens on release. (Setting QSlider::tracking(false)
    // would be the alternative; this gives smoother visual feedback.)
    m_position = value;
    updateTimeLabel();
}

void VideoControls::onSliderReleased()
{
    m_userIsDragging = false;
    emit seekRequested(static_cast<qint64>(m_slider->value()));
}

QString VideoControls::formatTime(qint64 milliseconds)
{
    if (milliseconds < 0) milliseconds = 0;
    const QTime t = QTime(0, 0).addMSecs(static_cast<int>(milliseconds));
    return milliseconds >= 3600 * 1000 ? t.toString("HH:mm:ss")
                                       : t.toString("mm:ss");
}

void VideoControls::updateTimeLabel()
{
    m_timeLabel->setText(
        QStringLiteral("%1 / %2")
            .arg(formatTime(m_position), formatTime(m_duration)));
}
