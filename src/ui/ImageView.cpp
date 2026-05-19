#include "ui/ImageView.h"

#include <QLabel>
#include <QPixmap>
#include <QResizeEvent>
#include <QVBoxLayout>

ImageView::ImageView(QWidget *parent)
    : QWidget(parent)
    , m_label(new QLabel(this)) // Qt parent-child: m_label is destroyed with us
{
    m_label->setAlignment(Qt::AlignCenter);
    m_label->setMinimumSize(1, 1);
    m_label->setText(tr("No image loaded.\nUse File -> Open Image..."));

    // A layout takes care of positioning m_label inside us, including when we
    // resize. We could also reparent m_label and call setGeometry by hand --
    // a layout is the idiomatic Qt way.
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_label);
}

void ImageView::setImage(const QImage &image)
{
    m_original = image;             // QImage uses implicit sharing -- cheap copy
    if (m_original.isNull()) {
        clear();
        return;
    }
    updateScaledPixmap();
}

void ImageView::clear()
{
    m_original = QImage();
    m_label->clear();
    m_label->setText(tr("No image loaded.\nUse File -> Open Image..."));
}

void ImageView::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    if (!m_original.isNull()) {
        updateScaledPixmap();
    }
}

void ImageView::updateScaledPixmap()
{
    // Always rescale from the original. Rescaling a scaled pixmap repeatedly
    // produces visible quality loss as the user resizes the window.
    const QSize target = m_label->size();
    if (target.isEmpty()) {
        return;
    }

    const QPixmap pixmap = QPixmap::fromImage(m_original).scaled(
        target,
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation);

    m_label->setPixmap(pixmap);
}
