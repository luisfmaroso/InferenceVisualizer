#include "inference/OnnxYoloSegBackend.h"

#include <QDebug>
#include <QFont>
#include <QFontMetrics>
#include <QPainter>
#include <QRect>

#include <algorithm>
#include <cmath>
#include <limits>

// ===========================================================================
// CONFIG: if your model differs from the defaults in OnnxYoloSegBackend.h,
// edit the Config defaults (touches one place) or call setConfig(cfg) before
// loadModel() (per-instance override).
//
// Note: confidenceThreshold and the class colours are overwritten at runtime
// by applySettings() (driven by the Settings dialog). The Config defaults for
// those are just the pre-settings starting point.
//
// Quantities that vary across models and are NOT in the UI:
//   - inputHeight / inputWidth         (training resolution, usually 640)
//   - numClasses                       (must match output0's [4+nc+nm] axis)
//   - numMaskCoeffs                    (must match output1's channel count)
//   - mean / std_                      (preprocessing -- YOLO defaults are 0/1)
//   - nmsIoUThreshold / maskThreshold
// ===========================================================================

OnnxYoloSegBackend::OnnxYoloSegBackend()
    : m_runner(std::make_unique<OnnxRunner>())
{
}

OnnxYoloSegBackend::~OnnxYoloSegBackend() = default;

void OnnxYoloSegBackend::setConfig(const Config &cfg)
{
    m_cfg = cfg;
}

void OnnxYoloSegBackend::applySettings(const InferenceSettings &settings)
{
    m_opacity = std::clamp(settings.overlayOpacity, 0.0, 1.0);
    m_cfg.confidenceThreshold = settings.confidenceThreshold;
    // Map the shared two-colour palette onto our class styles.
    for (int i = 0; i < 2 && i < static_cast<int>(m_cfg.classes.size()); ++i) {
        m_cfg.classes[static_cast<size_t>(i)].color = settings.classColors[static_cast<size_t>(i)];
    }
}

bool OnnxYoloSegBackend::isReady() const
{
    return m_runner && m_runner->isLoaded();
}

bool OnnxYoloSegBackend::loadModel(const QString &path)
{
    m_lastError.clear();
    if (!m_runner->loadModel(path)) {
        m_lastError = m_runner->lastError();
        return false;
    }

    qDebug() << "[OnnxYoloSegBackend] ready. inputHW="
             << m_cfg.inputHeight << "x" << m_cfg.inputWidth
             << "classes=" << m_cfg.numClasses
             << "maskCoeffs=" << m_cfg.numMaskCoeffs;
    return true;
}

// ---------------------------------------------------------------------------
// Preprocessing: QImage -> NCHW float buffer
// ---------------------------------------------------------------------------
void OnnxYoloSegBackend::buildInputTensor(const QImage &input)
{
    const int H = m_cfg.inputHeight;
    const int W = m_cfg.inputWidth;
    const int C = 3;

    // Stretch to model input size (no letterbox). Letterboxing preserves
    // aspect ratio and is what Ultralytics training uses; it would be more
    // accurate but adds complexity. Stretch-to-fit produces consistent
    // geometry (detection coords map back the same way) at the cost of
    // slightly degraded model accuracy on non-square inputs. Trade-off.
    QImage resized = input.scaled(W, H, Qt::IgnoreAspectRatio,
                                  Qt::SmoothTransformation);
    if (resized.format() != QImage::Format_RGB888) {
        resized = resized.convertToFormat(QImage::Format_RGB888);
    }

    m_inputBuffer.assign(static_cast<size_t>(C * H * W), 0.0f);

    // NCHW: R plane, then G plane, then B plane. PyTorch/Ultralytics export
    // convention. mean/std are the YOLO defaults (0/1) but kept parameterized
    // in case a non-Ultralytics training pipeline used different values.
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
// Postprocessing step 1: decode YOLOv8 output0 -> candidate detections, NMS
// ---------------------------------------------------------------------------
std::vector<OnnxYoloSegBackend::Detection>
OnnxYoloSegBackend::decodeAndNms(const OnnxRunner::TensorView &output0) const
{
    // Expected: [1, 4+nc+nm, A]  e.g. [1, 38, 8400] for nc=2, nm=32.
    if (output0.shape.size() != 3 || output0.shape[0] != 1) {
        qWarning() << "[OnnxYoloSegBackend] output0 has unexpected shape:"
                   << output0.shape;
        return {};
    }

    const int channels = static_cast<int>(output0.shape[1]);
    const int anchors  = static_cast<int>(output0.shape[2]);
    const int expectedChannels = 4 + m_cfg.numClasses + m_cfg.numMaskCoeffs;
    if (channels != expectedChannels) {
        qWarning() << "[OnnxYoloSegBackend] output0 channels =" << channels
                   << "but Config expects 4 + numClasses + numMaskCoeffs ="
                   << expectedChannels;
        return {};
    }

    // YOLO output0 layout: data[channel * A + anchor]. So an anchor's values
    // are NOT contiguous -- they're strided across `anchors` floats.
    auto val = [&](int channel, int anchor) -> float {
        return output0.data[channel * anchors + anchor];
    };

    const int clsOff  = 4;
    const int maskOff = 4 + m_cfg.numClasses;

    std::vector<Detection> candidates;
    candidates.reserve(64);

    for (int i = 0; i < anchors; ++i) {
        // Best class for this anchor.
        int   bestC = 0;
        float bestS = val(clsOff, i);
        for (int c = 1; c < m_cfg.numClasses; ++c) {
            const float s = val(clsOff + c, i);
            if (s > bestS) { bestS = s; bestC = c; }
        }
        if (bestS < m_cfg.confidenceThreshold) continue;

        // Decode bbox from (center_x, center_y, w, h) to (x1, y1, x2, y2).
        // Values are in pixel coordinates of the model input (e.g. 640x640).
        const float cx = val(0, i);
        const float cy = val(1, i);
        const float w  = val(2, i);
        const float h  = val(3, i);

        Detection d;
        d.x1 = cx - w * 0.5f;
        d.y1 = cy - h * 0.5f;
        d.x2 = cx + w * 0.5f;
        d.y2 = cy + h * 0.5f;
        d.score   = bestS;
        d.classId = bestC;
        d.maskCoeffs.resize(static_cast<size_t>(m_cfg.numMaskCoeffs));
        for (int k = 0; k < m_cfg.numMaskCoeffs; ++k) {
            d.maskCoeffs[k] = val(maskOff + k, i);
        }
        candidates.push_back(std::move(d));
    }

    // NMS: sort by score desc, suppress later detections of the same class
    // whose IoU with an earlier one exceeds the threshold.
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection &a, const Detection &b) { return a.score > b.score; });

    std::vector<Detection> survivors;
    survivors.reserve(candidates.size());
    std::vector<bool> suppressed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        survivors.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            if (candidates[i].classId != candidates[j].classId) continue;
            if (iou(candidates[i], candidates[j]) > m_cfg.nmsIoUThreshold) {
                suppressed[j] = true;
            }
        }
    }

    static quint64 s_logCount = 0;
    if ((s_logCount++ % 30) == 0) {
        qDebug() << "[OnnxYoloSegBackend] decoded"
                 << candidates.size() << "above-threshold,"
                 << survivors.size() << "after NMS";
    }

    return survivors;
}

float OnnxYoloSegBackend::iou(const Detection &a, const Detection &b)
{
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw  = std::max(0.f, ix2 - ix1);
    const float ih  = std::max(0.f, iy2 - iy1);
    const float inter = iw * ih;
    const float aArea = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    const float bArea = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    const float uni = aArea + bArea - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

// ---------------------------------------------------------------------------
// Postprocessing step 2: per-detection mask = sigmoid(coeffs @ prototypes)
// Paint each binarised mask onto a display-size ARGB canvas, cropped to bbox.
// ---------------------------------------------------------------------------
QImage OnnxYoloSegBackend::buildMaskImage(
    const std::vector<Detection> &dets,
    const OnnxRunner::TensorView &output1,
    QSize displaySize) const
{
    // Expected: [1, nm, ph, pw]  e.g. [1, 32, 160, 160].
    if (output1.shape.size() != 4 || output1.shape[0] != 1) {
        qWarning() << "[OnnxYoloSegBackend] output1 has unexpected shape:"
                   << output1.shape;
        return QImage();
    }
    const int nm = static_cast<int>(output1.shape[1]);
    const int ph = static_cast<int>(output1.shape[2]);
    const int pw = static_cast<int>(output1.shape[3]);
    if (nm != m_cfg.numMaskCoeffs) {
        qWarning() << "[OnnxYoloSegBackend] output1 channels =" << nm
                   << "but numMaskCoeffs =" << m_cfg.numMaskCoeffs;
        return QImage();
    }

    QImage mask(displaySize, QImage::Format_ARGB32);
    mask.fill(Qt::transparent);

    // Scale factors from model-input space (e.g. 640) to mask space (e.g. 160)
    // and display space (whatever the original image is).
    const float toMaskX = static_cast<float>(pw) / m_cfg.inputWidth;
    const float toMaskY = static_cast<float>(ph) / m_cfg.inputHeight;
    const float toDispX = static_cast<float>(displaySize.width())  / m_cfg.inputWidth;
    const float toDispY = static_cast<float>(displaySize.height()) / m_cfg.inputHeight;

    auto proto = [&](int k, int y, int x) -> float {
        return output1.data[(k * ph + y) * pw + x];
    };

    // Buffer for one instance's full-prototype-size mask -- reused per detection.
    std::vector<float> instMask(static_cast<size_t>(ph * pw));

    for (const Detection &d : dets) {
        if (d.classId < 0 || d.classId >= static_cast<int>(m_cfg.classes.size())) {
            continue;
        }
        const ClassStyle &cs = m_cfg.classes[static_cast<size_t>(d.classId)];
        if (!cs.visible) continue;

        // matmul: instMask = sigmoid( sum_k coeff[k] * proto[k, :, :] )
        for (int y = 0; y < ph; ++y) {
            for (int x = 0; x < pw; ++x) {
                float sum = 0.f;
                for (int k = 0; k < nm; ++k) {
                    sum += d.maskCoeffs[static_cast<size_t>(k)] * proto(k, y, x);
                }
                instMask[static_cast<size_t>(y * pw + x)] = 1.f / (1.f + std::exp(-sum));
            }
        }

        // bbox in mask space and display space, clipped to image bounds
        const int mx1 = std::max(0,  static_cast<int>(std::floor(d.x1 * toMaskX)));
        const int my1 = std::max(0,  static_cast<int>(std::floor(d.y1 * toMaskY)));
        const int mx2 = std::min(pw, static_cast<int>(std::ceil (d.x2 * toMaskX)));
        const int my2 = std::min(ph, static_cast<int>(std::ceil (d.y2 * toMaskY)));

        const int dx1 = std::max(0, static_cast<int>(std::floor(d.x1 * toDispX)));
        const int dy1 = std::max(0, static_cast<int>(std::floor(d.y1 * toDispY)));
        const int dx2 = std::min(displaySize.width(),
                                  static_cast<int>(std::ceil (d.x2 * toDispX)));
        const int dy2 = std::min(displaySize.height(),
                                  static_cast<int>(std::ceil (d.y2 * toDispY)));

        if (mx2 <= mx1 || my2 <= my1) continue;
        if (dx2 <= dx1 || dy2 <= dy1) continue;

        const QColor &col = cs.color;
        const QRgb classRgb = qRgba(col.red(), col.green(), col.blue(), 255);

        // For each display-space pixel in the bbox, sample the corresponding
        // mask pixel by nearest-neighbour mapping. We only paint where the
        // mask exceeds the binarisation threshold.
        const float maskW = static_cast<float>(mx2 - mx1);
        const float maskH = static_cast<float>(my2 - my1);
        const float dispW = static_cast<float>(dx2 - dx1);
        const float dispH = static_cast<float>(dy2 - dy1);

        for (int dy = dy1; dy < dy2; ++dy) {
            QRgb *outLine = reinterpret_cast<QRgb *>(mask.scanLine(dy));
            const int my = std::min(my2 - 1,
                my1 + static_cast<int>((dy - dy1) * maskH / dispH));
            for (int dx = dx1; dx < dx2; ++dx) {
                const int mx = std::min(mx2 - 1,
                    mx1 + static_cast<int>((dx - dx1) * maskW / dispW));
                if (instMask[static_cast<size_t>(my * pw + mx)] >= m_cfg.maskThreshold) {
                    outLine[dx] = classRgb;
                }
            }
        }
    }

    return mask;
}

QImage OnnxYoloSegBackend::compositeOverlay(
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

// ---------------------------------------------------------------------------
// Draw a "NN%" confidence chip at each detection. Done AFTER compositing so
// the text sits at full opacity (the mask alpha doesn't fade the numbers) and
// at display resolution (so the text is crisp, not scaled with the mask).
// ---------------------------------------------------------------------------
void OnnxYoloSegBackend::drawConfidenceChips(
    QImage &canvas,
    const std::vector<Detection> &dets,
    QSize displaySize) const
{
    const float toDispX = static_cast<float>(displaySize.width())  / m_cfg.inputWidth;
    const float toDispY = static_cast<float>(displaySize.height()) / m_cfg.inputHeight;

    QPainter painter(&canvas);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QFont font = painter.font();
    // Scale the text to the image so it's legible on big frames and not huge
    // on small ones. Clamped to a sensible minimum.
    font.setPointSizeF(std::max(9.0, displaySize.height() / 55.0));
    font.setBold(true);
    painter.setFont(font);
    const QFontMetrics fm(font);

    for (const Detection &d : dets) {
        if (d.classId < 0 || d.classId >= static_cast<int>(m_cfg.classes.size())) {
            continue;
        }
        const ClassStyle &cs = m_cfg.classes[static_cast<size_t>(d.classId)];
        if (!cs.visible) continue;

        const int pct = static_cast<int>(std::lround(d.score * 100.0f));
        const QString text = QStringLiteral("%1%").arg(pct);

        const int padX = 4;
        const int padY = 2;
        const QSize textSize = fm.size(Qt::TextSingleLine, text);
        QRect chip(0, 0, textSize.width() + 2 * padX, textSize.height() + 2 * padY);

        // Anchor at the detection's top-left in display coords, kept on-screen.
        int x = static_cast<int>(d.x1 * toDispX);
        int y = static_cast<int>(d.y1 * toDispY);
        x = std::clamp(x, 0, std::max(0, displaySize.width()  - chip.width()));
        y = std::clamp(y, 0, std::max(0, displaySize.height() - chip.height()));
        chip.moveTo(x, y);

        // Filled background in the class colour (opaque), white text on top.
        painter.fillRect(chip, QColor(cs.color.red(), cs.color.green(), cs.color.blue()));
        painter.setPen(Qt::white);
        painter.drawText(chip, Qt::AlignCenter, text);
    }

    painter.end();
}

QImage OnnxYoloSegBackend::infer(const QImage &input)
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
    if (outputs.size() < 2) {
        m_lastError = m_runner->lastError().isEmpty()
            ? QStringLiteral("Expected 2 outputs (detections + prototypes), got %1.")
                  .arg(outputs.size())
            : m_runner->lastError();
        return QImage();
    }

    const auto allDets = decodeAndNms(outputs[0]);
    if (allDets.empty()) {
        // Nothing above the confidence threshold -- not an error, just no
        // overlay. Return the original so the right pane shows it cleanly.
        return input;
    }

    // Show only the single most-confident detection: its mask and its "NN%"
    // chip. (decodeAndNms already sorts by score, so this is usually the
    // front element, but max_element is robust regardless of ordering.)
    const Detection best = *std::max_element(
        allDets.begin(), allDets.end(),
        [](const Detection &a, const Detection &b) { return a.score < b.score; });
    const std::vector<Detection> dets = { best };

    const QImage mask = buildMaskImage(dets, outputs[1], input.size());
    if (mask.isNull()) {
        m_lastError = QStringLiteral("Mask synthesis failed (see log).");
        return QImage();
    }

    QImage result = compositeOverlay(input, mask);
    drawConfidenceChips(result, dets, input.size());  // single % chip at full opacity
    return result;
}
