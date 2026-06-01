#pragma once

#include "inference/InferenceSettings.h"

#include <QDialog>

class QComboBox;
class QSlider;
class QLabel;
class QDoubleSpinBox;
class QPushButton;

// SettingsDialog: a modal QDialog for editing inference settings -- model type,
// mask opacity, confidence threshold, and the two class colours.
//
// It is deliberately decoupled from InferenceController: it's constructed with
// the current values, edits a local copy, and reports back via the applied()
// signal (on Apply/OK) plus the settings()/modelType() getters. MainWindow
// owns the wiring between this dialog and the controller.
class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    SettingsDialog(const InferenceSettings &settings,
                   ModelType modelType,
                   QWidget *parent = nullptr);

    InferenceSettings settings() const  { return m_settings; }
    ModelType         modelType() const { return m_modelType; }

signals:
    // Emitted on Apply and on OK -- lets MainWindow live-preview changes.
    void applied(const InferenceSettings &settings, ModelType modelType);

private slots:
    void pickColor(int classIndex);     // open QColorDialog for class 0 or 1
    void onOpacitySliderChanged(int percent);
    void applyChanges();                // pull widgets -> m_settings, emit applied()
    void restoreDefaults();

private:
    void   pullFromWidgets();           // widgets -> m_settings / m_modelType
    void   pushToWidgets();             // m_settings / m_modelType -> widgets
    void   updateColorSwatch(int classIndex);

    InferenceSettings m_settings;
    ModelType         m_modelType;

    QComboBox      *m_modelTypeCombo  = nullptr;
    QSlider        *m_opacitySlider   = nullptr;
    QLabel         *m_opacityLabel    = nullptr;
    QDoubleSpinBox *m_confidenceSpin  = nullptr;
    QPushButton    *m_colorButtons[2] = {nullptr, nullptr};
};
