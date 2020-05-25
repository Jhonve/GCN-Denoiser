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

private:
	void bindSlotsAndSignals();

signals:
	void signalGenNoise(double noise_level, QString noise_type);
	void signalDenoise(int gcns, int normal_iterations);

public slots:
	void slotGenNoise();
	void slotDenoise();
};
