#pragma once

#include <QImage>
#include <QWidget>

class QLabel;
class QResizeEvent;

// ImageView: a self-contained widget that displays a QImage scaled to fit,
// preserving aspect ratio. It stores the *original* image and rescales from it
// on every resize, so we never compound scaling artifacts.
//
// This class deliberately knows nothing about file I/O, video, or inference --
// it just renders pixels. Steps 2-4 will feed it video frames and processed
// images through the same setImage() entry point.
class ImageView : public QWidget
{
    Q_OBJECT

public:
    explicit ImageView(QWidget *parent = nullptr);

    void setImage(const QImage &image);
    void clear();

    bool hasImage() const { return !m_original.isNull(); }

protected:
    void resizeEvent(QResizeEvent *event) override;

private:
    void updateScaledPixmap();

    QLabel *m_label;    // owned by Qt's parent-child system (this widget)
    QImage  m_original; // master copy; we rescale from this on every resize

    Q_DISABLE_COPY(ImageView)
};
