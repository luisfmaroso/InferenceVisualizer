#pragma once

#include "inference/IInferenceBackend.h"
#include "inference/OnnxRunner.h"

#include <QColor>
#include <QImage>
#include <QString>

#include <array>
#include <memory>
#include <vector>

// OnnxSegmentationBackend: IInferenceBackend implementation that runs a
// YOLOv8 instance-segmentation ONNX model and composites per-instance
// colored masks onto the input image.
//
// Expected model format (YOLOv8 seg, from `yolo export format=onnx` of any
// seg-task .pt or via Ultralytics):
//   Input  shape  [1, 3, H, W]   float32 (RGB, range [0, 1], no normalization)
//   Output 0      [1, 4+nc+nm, A]                per-anchor:
//                                                  4 box (cx, cy, w, h)
//                                                  nc class scores
//                                                  nm mask coefficients
//   Output 1      [1, nm, H/4, W/4]              nm mask prototypes
//
// For the corrosion model that drove Step 3:
//   H = W = 640,  nc = 2 (not-critical / critical),  nm = 32,  A = 8400.
//
// Pure semantic-segmentation models (UNet-style, output [1, C, H, W]) have
// a totally different post-processing path -- they are NOT supported by this
// file. The "load any ONNX model architecture generically" feature is on the
// future-improvements list and will eventually pick the right backend per
// model shape.
class OnnxSegmentationBackend : public IInferenceBackend
{
public:
    struct ClassStyle {
        QString name;        // displayed in legends / tooltips later
        QColor  color;       // RGB; alpha applied at composite time
        bool    visible;     // hide a class from the overlay
    };

    struct Config {
        // --- model shape ---
        int inputHeight    = 640;
        int inputWidth     = 640;
        int numClasses     = 2;
        int numMaskCoeffs  = 32;   // must match the channel count of output1

        // --- preprocessing ---
        // YOLOv8 expects raw [0, 1] floats. If your training pipeline used
        // ImageNet normalization or something else, change these to match.
        std::array<float, 3> mean = {0.0f, 0.0f, 0.0f};
        std::array<float, 3> std_ = {1.0f, 1.0f, 1.0f};

        // --- post-processing ---
        float confidenceThreshold = 0.05f;   // YOLOv8 default
        float nmsIoUThreshold     = 0.7f;    // YOLOv8 default
        float maskThreshold       = 0.5f;    // binarize sigmoid'd mask

        // --- presentation ---
        // Class index 0 = not-critical, 1 = critical. If your training
        // assigned them in the opposite order, swap the two entries below.
        std::vector<ClassStyle> classes = {
            {"not-critical", QColor(255, 215,  64), /*visible=*/ true},  // dim yellow
            {"critical",     QColor(220,  20,  20), /*visible=*/ true},  // bold red
        };
    };

    OnnxSegmentationBackend();
    ~OnnxSegmentationBackend() override;

    void setConfig(const Config &cfg);
    const Config &config() const { return m_cfg; }

    // IInferenceBackend
    bool    loadModel(const QString &path) override;
    bool    isReady() const override;
    QString errorString() const override { return m_lastError; }
    QImage  infer(const QImage &input) override;
    void    setOverlayOpacity(double opacity) override;

private:
    // One detected instance, in model-input coordinates (e.g. 0..640).
    struct Detection {
        float x1, y1, x2, y2;            // xyxy
        float score;                     // best class score
        int   classId;
        std::vector<float> maskCoeffs;   // length == Config::numMaskCoeffs
    };

    void                   buildInputTensor(const QImage &input);
    std::vector<Detection> decodeAndNms(const OnnxRunner::TensorView &output0) const;
    QImage                 buildMaskImage(const std::vector<Detection> &dets,
                                          const OnnxRunner::TensorView &output1,
                                          QSize displaySize) const;
    QImage                 compositeOverlay(const QImage &original,
                                            const QImage &mask) const;

    static float           iou(const Detection &a, const Detection &b);

    Config                       m_cfg;
    std::unique_ptr<OnnxRunner>  m_runner;
    std::vector<float>           m_inputBuffer;   // reused across frames
    QString                      m_lastError;
    double                       m_opacity = 0.5;
};
