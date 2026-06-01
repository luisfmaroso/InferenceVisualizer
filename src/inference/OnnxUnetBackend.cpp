#include "inference/OnnxUnetBackend.h"

#include <QDebug>
#include <QPainter>

#include <algorithm>
#include <cmath>
#include <limits>

// ===========================================================================
// CONFIG: adjust the Config defaults in OnnxUnetBackend.h to match how your
// UNet was trained -- input size and normalization especially. The class
// colours and the confidence/probability floor come from the Settings dialog
// via applySettings(); the Config values for those are pre-settings defaults.
// ===========================================================================

OnnxUnetBackend::OnnxUnetBackend()
    : m_runner(std::make_unique<OnnxRunner>())
{
}

OnnxUnetBackend::~OnnxUnetBackend() = default;

void OnnxUnetBackend::setConfig(const Config &cfg)
{
    m_cfg = cfg;
}

void OnnxUnetBackend::applySettings(const InferenceSettings &settings)
{
    m_opacity   = std::clamp(settings.overlayOpacity, 0.0, 1.0);
    m_probFloor = settings.confidenceThreshold;
    for (int i = 0; i < 2 && i < static_cast<int>(m_cfg.classes.size()); ++i) {
        m_cfg.classes[static_cast<size_t>(i)].color = settings.classColors[static_cast<size_t>(i)];
    }
}

bool OnnxUnetBackend::isReady() const
{
    return m_runner && m_runner->isLoaded();
}

bool OnnxUnetBackend::loadModel(const QString &path)
{
    m_lastError.clear();
    if (!m_runner->loadModel(path)) {
        m_lastError = m_runner->lastError();
        return false;
    }
    qDebug() << "[OnnxUnetBackend] ready. inputHW="
             << m_cfg.inputHeight << "x" << m_cfg.inputWidth
             << "classes=" << m_cfg.numClasses;
    return true;
}

// ---------------------------------------------------------------------------
// Preprocessing: QImage -> NCHW float buffer (same packing as the YOLO path)
// ---------------------------------------------------------------------------
void OnnxUnetBackend::buildInputTensor(const QImage &input)
{
    const int H = m_cfg.inputHeight;
    const int W = m_cfg.inputWidth;
    const int C = 3;

    QImage resized = input.scaled(W, H, Qt::IgnoreAspectRatio,
                                  Qt::SmoothTransformation);
    if (resized.format() != QImage::Format_RGB888) {
        resized = resized.convertToFormat(QImage::Format_RGB888);
    }

    m_inputBuffer.assign(static_cast<size_t>(C * H * W), 0.0f);

    const float invStdR = 1.0f / m_cfg.std_[0];
    const float invStdG = 1.0f / m_cfg.std_[1];
    const float invStdB = 1.0f / m_cfg.std_[2];

    for (int y = 0; y < H; ++y) {
        const uchar *line = resized.constScanLine(y);
        for (int x = 0; x < W; ++x) {
            const float r = line[x * 3 + 0] / 255.0f;
            const float g = line[x * 3 + 1] / 255.0f;
            const float b = line[x * 3 + 2] / 255.0f;
            m_inputBuffer[0 * H * W + y * W + x] = (r - m_cfg.mean[0]) * invStdR;
            m_inputBuffer[1 * H * W + y * W + x] = (g - m_cfg.mean[1]) * invStdG;
            m_inputBuffer[2 * H * W + y * W + x] = (b - m_cfg.mean[2]) * invStdB;
        }
    }
}

// ---------------------------------------------------------------------------
// Postprocessing: per-pixel argmax over the class channel.
// Output expected [1, C, H, W]. For each pixel, find the class with the
// highest logit; if its softmax probability is below m_probFloor, leave the
// pixel un-overlaid. Otherwise paint with that class's colour (if visible).
// ---------------------------------------------------------------------------
QImage OnnxUnetBackend::buildMaskImage(
    const OnnxRunner::TensorView &output,
    QSize displaySize) const
{
    // Tolerate [1, C, H, W] and [C, H, W] (some exports drop the batch dim).
    int C = m_cfg.numClasses, H = m_cfg.inputHeight, W = m_cfg.inputWidth;
    if (output.shape.size() == 4) {
        C = static_cast<int>(output.shape[1]);
        H = static_cast<int>(output.shape[2]);
        W = static_cast<int>(output.shape[3]);
    } else if (output.shape.size() == 3) {
        C = static_cast<int>(output.shape[0]);
        H = static_cast<int>(output.shape[1]);
        W = static_cast<int>(output.shape[2]);
    } else {
        qWarning() << "[OnnxUnetBackend] output has unexpected shape:"
                   << output.shape;
        return QImage();
    }

    // NCHW indexing: each class is a full H*W plane.
    auto logit = [&](int c, int y, int x) -> float {
        return output.data[(c * H + y) * W + x];
    };

    QImage mask(W, H, QImage::Format_ARGB32);
    mask.fill(Qt::transparent);

    for (int y = 0; y < H; ++y) {
        QRgb *outLine = reinterpret_cast<QRgb *>(mask.scanLine(y));
        for (int x = 0; x < W; ++x) {
            // argmax over channels
            int   bestC = 0;
            float bestL = logit(0, y, x);
            for (int c = 1; c < C; ++c) {
                const float v = logit(c, y, x);
                if (v > bestL) { bestL = v; bestC = c; }
            }

            // softmax probability of the winning class (numerically stable)
            float denom = 0.f;
            for (int c = 0; c < C; ++c) {
                denom += std::exp(logit(c, y, x) - bestL);
            }
            const float prob = 1.f / denom;   // exp(bestL - bestL) / denom
            if (prob < m_probFloor) continue;

            if (bestC >= 0 && bestC < static_cast<int>(m_cfg.classes.size())) {
                const ClassStyle &cs = m_cfg.classes[static_cast<size_t>(bestC)];
                if (cs.visible) {
                    const QColor &col = cs.color;
                    outLine[x] = qRgba(col.red(), col.green(), col.blue(), 255);
                }
            }
        }
    }

    // Upscale to display size with nearest-neighbour so class regions stay
    // crisp instead of bleeding at the boundaries.
    return mask.scaled(displaySize, Qt::IgnoreAspectRatio, Qt::FastTransformation);
}

QImage OnnxUnetBackend::compositeOverlay(
    const QImage &original,
    const QImage &mask) const
{
    QImage result = original.convertToFormat(QImage::Format_ARGB32);

    QPainter painter(&result);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
    painter.setOpacity(m_opacity);
    painter.drawImage(0, 0, mask);
    painter.end();

    return result;
}

QImage OnnxUnetBackend::infer(const QImage &input)
{
    m_lastError.clear();
    if (!isReady()) {
        m_lastError = QStringLiteral("Backend has no loaded model.");
        return QImage();
    }
    if (input.isNull()) {
        m_lastError = QStringLiteral("Input image is null.");
        return QImage();
    }

    buildInputTensor(input);

    const std::vector<int64_t> inShape = {
        1, 3, m_cfg.inputHeight, m_cfg.inputWidth
    };
    auto outputs = m_runner->run(m_inputBuffer.data(), inShape);
    if (outputs.empty()) {
        m_lastError = m_runner->lastError().isEmpty()
            ? QStringLiteral("Model returned no output.")
            : m_runner->lastError();
        return QImage();
    }

    const QImage mask = buildMaskImage(outputs[0], input.size());
    if (mask.isNull()) {
        m_lastError = QStringLiteral("Mask synthesis failed (see log).");
        return QImage();
    }

    return compositeOverlay(input, mask);
}
