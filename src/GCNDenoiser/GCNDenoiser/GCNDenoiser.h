#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_GCNDenoiser.h"

class GCNDenoiser : public QMainWindow
{
    Q_OBJECT

public:
    GCNDenoiser(QWidget *parent = Q_NULLPTR);

private:
    Ui::GCNDenoiserClass ui;
};
