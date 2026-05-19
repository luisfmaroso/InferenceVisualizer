#include "app/MainWindow.h"
#include "ui/ImageView.h"

#include <QAction>
#include <QApplication>
#include <QFileDialog>
#include <QImage>
#include <QImageReader>
#include <QKeySequence>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QStatusBar>
#include <QString>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_view(new ImageView(this))
{
    setWindowTitle(tr("InferenceVisualizer"));
    resize(1000, 700);

    setCentralWidget(m_view); // QMainWindow takes ownership

    buildMenus();
    statusBar()->showMessage(tr("Ready."));
}

void MainWindow::buildMenus()
{
    auto *fileMenu = menuBar()->addMenu(tr("&File"));

    auto *openAction = fileMenu->addAction(tr("&Open Image..."));
    openAction->setShortcut(QKeySequence::Open); // Ctrl+O on Windows/Linux
    connect(openAction, &QAction::triggered, this, &MainWindow::openImage);

    fileMenu->addSeparator();

    auto *exitAction = fileMenu->addAction(tr("E&xit"));
    exitAction->setShortcut(QKeySequence::Quit); // Ctrl+Q
    // qApp is a macro for QApplication::instance(); ::quit posts a quit event
    // to the event loop, which causes exec() in main() to return cleanly.
    connect(exitAction, &QAction::triggered, qApp, &QApplication::quit);
}

void MainWindow::openImage()
{
    // Build the filter string from Qt's list of supported formats so we
    // automatically pick up any plugins (e.g. .webp via the imageformats plugin).
    QStringList patterns;
    for (const QByteArray &fmt : QImageReader::supportedImageFormats()) {
        patterns << QStringLiteral("*.%1").arg(QString::fromLatin1(fmt));
    }
    const QString filter = tr("Images (%1);;All files (*)").arg(patterns.join(' '));

    const QString path = QFileDialog::getOpenFileName(
        this,
        tr("Open Image"),
        QString(),  // start in the last-used dir (Qt default)
        filter);

    if (path.isEmpty()) {
        return; // user cancelled -- not an error
    }

    QImageReader reader(path);
    reader.setAutoTransform(true); // honour EXIF orientation
    const QImage image = reader.read();

    if (image.isNull()) {
        QMessageBox::warning(
            this,
            tr("Could not open image"),
            tr("Failed to load \"%1\":\n%2").arg(path, reader.errorString()));
        statusBar()->showMessage(tr("Failed to open %1").arg(path), 5000);
        return;
    }

    m_view->setImage(image);
    statusBar()->showMessage(
        tr("Loaded %1 (%2x%3)").arg(path).arg(image.width()).arg(image.height()),
        5000);
}
