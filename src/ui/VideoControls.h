#pragma once

#include <QWidget>

class QLabel;
class QPushButton;
class QSlider;

// VideoControls: a compact widget with a play/pause button, a position slider,
// and time labels. It is deliberately unaware of QMediaPlayer or
// VideoController -- it only emits user-intent signals and consumes
// display-update slots. MainWindow wires it to the controller.
class VideoControls : public QWidget
{
    Q_OBJECT

public:
    explicit VideoControls(QWidget *parent = nullptr);

public slots:
    void setEnabledControls(bool enabled);    // grey out when no media
    void setDuration(qint64 milliseconds);
    void setPosition(qint64 milliseconds);
    void setPlaying(bool playing);            // updates the button glyph/text

signals:
    void playPauseClicked();
    void seekRequested(qint64 milliseconds);

private slots:
    void onSliderMoved(int value);
    void onSliderReleased();

private:
    static QString formatTime(qint64 milliseconds);
    void updateTimeLabel();

    QPushButton *m_playPauseBtn;
    QSlider     *m_slider;
    QLabel      *m_timeLabel;
    qint64       m_duration{0};
    qint64       m_position{0};
    bool         m_userIsDragging{false}; // suppress position-pushed-to-slider
};
