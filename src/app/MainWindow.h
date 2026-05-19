#pragma once

#include <QMainWindow>

class ImageView;

// MainWindow: the top-level application window. Knows about menus and file
// dialogs; does NOT know how to render an image (that's ImageView's job) and
// does NOT know about inference (will be added in step 3 via a controller).
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void openImage();

private:
    void buildMenus();

    ImageView *m_view; // owned via Qt parent-child (set as central widget)
};
