#pragma once

#include "inference/IInferenceBackend.h"
#include "inference/OnnxRunner.h"

#include <QColor>
#include <QImage>
#include <QString>

#include <array>
#include <memory>
#include <vector>

// OnnxUnetBackend: IInferenceBackend implementation for UNet-style semantic
// segmentation models -- the kind hand-exported from PyTorch via
// torch.onnx.export.
//
// Expected model format:
//   Input  shape  [1, 3, H, W]   float32
//   Output shape  [1, C, H, W]   float32 per-pixel class logits
//
// Postprocessing is per-pixel argmax over the C channel axis -- no anchors,
// no NMS, no mask coefficients (that's the YOLO path, in OnnxYoloSegBackend).
// The confidenceThreshold from the Settings dialog is interpreted here as a
// softmax-probability floor: a pixel whose winning class probability is below
// it is left un-overlaid, so the slider stays meaningful for UNet too.
class OnnxUnetBackend : public IInferenceBackend
{
public:
    struct ClassStyle {
        QString name;
        QColor  color;
        bool    visible;
    };

    struct Config {
        // --- model shape ---
        // CONFIG: set these to match how YOUR UNet was trained/exported.
        int inputHeight = 512;
        int inputWidth  = 512;
        int numClasses  = 2;

        // --- preprocessing ---
        // CONFIG: many UNet pipelines use ImageNet normalization; the default
        // here is raw [0, 1] (mean 0 / std 1). Adjust if your training differed.
        std::array<float, 3> mean = {0.0f, 0.0f, 0.0f};
        std::array<float, 3> std_ = {1.0f, 1.0f, 1.0f};

        // --- presentation ---
        // Colours overwritten by applySettings(); these defaults match the
        // InferenceSettings defaults (green / red). Class 0 is treated as
        // "background" by default (not drawn) so a 2-class fg/bg UNet shows
        // only the foreground -- flip `visible` if your class 0 is meaningful.
        std::vector<ClassStyle> classes = {
            {"class 0", QColor(0, 180, 0),   /*visible=*/ false},  // background by default
            {"class 1", QColor(220, 20, 20), /*visible=*/ true},
        };
    };

    OnnxUnetBackend();
    ~OnnxUnetBackend() override;

    void setConfig(const Config &cfg);
    const Config &config() const { return m_cfg; }

    // IInferenceBackend
    bool    loadModel(const QString &path) override;
    bool    isReady() const override;
    QString errorString() const override { return m_lastError; }
    QImage  infer(const QImage &input) override;
    void    applySettings(const InferenceSettings &settings) override;

private:
    void   buildInputTensor(const QImage &input);
    QImage buildMaskImage(const OnnxRunner::TensorView &output,
                          QSize displaySize) const;
    QImage compositeOverlay(const QImage &original, const QImage &mask) const;

    Config                       m_cfg;
    std::unique_ptr<OnnxRunner>  m_runner;
    std::vector<float>           m_inputBuffer;   // reused across frames
    QString                      m_lastError;
    double                       m_opacity = 0.5;
    float                        m_probFloor = 0.25f;  // from confidenceThreshold
};
