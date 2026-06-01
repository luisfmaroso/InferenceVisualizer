#include "ui/SettingsDialog.h"

#include <QColorDialog>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>

namespace {
// Paint a button's background with `color` so it reads as a colour swatch.
void styleSwatch(QPushButton *btn, const QColor &color)
{
    // A stylesheet is the simplest way to fill a push button with a solid
    // colour and keep a thin border so light colours stay visible.
    btn->setStyleSheet(QStringLiteral(
        "QPushButton { background-color: %1; border: 1px solid #888; "
        "min-width: 48px; min-height: 22px; }")
        .arg(color.name()));
}
} // namespace

SettingsDialog::SettingsDialog(const InferenceSettings &settings,
                               ModelType modelType,
                               QWidget *parent)
    : QDialog(parent)
    , m_settings(settings)
    , m_modelType(modelType)
{
    setWindowTitle(tr("Inference Settings"));
    setModal(true);

    // --- model type -----------------------------------------------------
    m_modelTypeCombo = new QComboBox(this);
    m_modelTypeCombo->addItem(tr("YOLOv8 / YOLO11 Segmentation"),
                              static_cast<int>(ModelType::YoloSeg));
    m_modelTypeCombo->addItem(tr("UNet Segmentation"),
                              static_cast<int>(ModelType::Unet));

    // --- mask opacity (slider 0..100 + live % label) --------------------
    m_opacitySlider = new QSlider(Qt::Horizontal, this);
    m_opacitySlider->setRange(0, 100);
    m_opacityLabel = new QLabel(this);
    m_opacityLabel->setMinimumWidth(40);
    auto *opacityRow = new QWidget(this);
    auto *opacityLayout = new QHBoxLayout(opacityRow);
    opacityLayout->setContentsMargins(0, 0, 0, 0);
    opacityLayout->addWidget(m_opacitySlider, /*stretch=*/1);
    opacityLayout->addWidget(m_opacityLabel);
    connect(m_opacitySlider, &QSlider::valueChanged,
            this, &SettingsDialog::onOpacitySliderChanged);

    // --- confidence threshold -------------------------------------------
    m_confidenceSpin = new QDoubleSpinBox(this);
    m_confidenceSpin->setRange(0.00, 1.00);
    m_confidenceSpin->setSingleStep(0.01);   // user-requested step
    m_confidenceSpin->setDecimals(2);

    // --- class colour swatches ------------------------------------------
    for (int i = 0; i < 2; ++i) {
        m_colorButtons[i] = new QPushButton(this);
        connect(m_colorButtons[i], &QPushButton::clicked, this,
                [this, i]() { pickColor(i); });
    }

    // --- form layout -----------------------------------------------------
    auto *form = new QFormLayout;
    form->addRow(tr("Model type:"),          m_modelTypeCombo);
    form->addRow(tr("Mask opacity:"),        opacityRow);
    form->addRow(tr("Confidence threshold:"), m_confidenceSpin);
    form->addRow(tr("Class 0 colour:"),      m_colorButtons[0]);
    form->addRow(tr("Class 1 colour:"),      m_colorButtons[1]);

    // --- buttons ---------------------------------------------------------
    auto *buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel |
        QDialogButtonBox::Apply | QDialogButtonBox::RestoreDefaults,
        this);

    connect(buttons->button(QDialogButtonBox::Apply), &QPushButton::clicked,
            this, &SettingsDialog::applyChanges);
    connect(buttons->button(QDialogButtonBox::RestoreDefaults), &QPushButton::clicked,
            this, &SettingsDialog::restoreDefaults);
    connect(buttons, &QDialogButtonBox::accepted, this, [this]() {
        applyChanges();   // OK = apply + close
        accept();
    });
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    auto *root = new QVBoxLayout(this);
    root->addLayout(form);
    root->addWidget(buttons);

    pushToWidgets();
}

void SettingsDialog::onOpacitySliderChanged(int percent)
{
    m_opacityLabel->setText(QStringLiteral("%1%").arg(percent));
}

void SettingsDialog::pickColor(int classIndex)
{
    if (classIndex < 0 || classIndex > 1) return;
    const QColor chosen = QColorDialog::getColor(
        m_settings.classColors[static_cast<size_t>(classIndex)],
        this,
        tr("Choose colour for class %1").arg(classIndex));
    if (chosen.isValid()) {
        m_settings.classColors[static_cast<size_t>(classIndex)] = chosen;
        updateColorSwatch(classIndex);
    }
}

void SettingsDialog::updateColorSwatch(int classIndex)
{
    styleSwatch(m_colorButtons[classIndex],
                m_settings.classColors[static_cast<size_t>(classIndex)]);
}

void SettingsDialog::pushToWidgets()
{
    const int comboIdx = m_modelTypeCombo->findData(static_cast<int>(m_modelType));
    m_modelTypeCombo->setCurrentIndex(comboIdx >= 0 ? comboIdx : 0);

    const int pct = static_cast<int>(m_settings.overlayOpacity * 100.0 + 0.5);
    m_opacitySlider->setValue(pct);
    onOpacitySliderChanged(pct);   // refresh the % label

    m_confidenceSpin->setValue(static_cast<double>(m_settings.confidenceThreshold));

    updateColorSwatch(0);
    updateColorSwatch(1);
}

void SettingsDialog::pullFromWidgets()
{
    m_modelType = static_cast<ModelType>(
        m_modelTypeCombo->currentData().toInt());
    m_settings.overlayOpacity = m_opacitySlider->value() / 100.0;
    m_settings.confidenceThreshold =
        static_cast<float>(m_confidenceSpin->value());
    // class colours are already stored in m_settings by pickColor()
}

void SettingsDialog::applyChanges()
{
    pullFromWidgets();
    emit applied(m_settings, m_modelType);
}

void SettingsDialog::restoreDefaults()
{
    m_settings  = InferenceSettings{};        // struct defaults (green/red, 0.5, 0.25)
    m_modelType = ModelType::YoloSeg;
    pushToWidgets();
}
