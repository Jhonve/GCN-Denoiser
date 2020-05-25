#include "GCNDenoiser.h"

GCNDenoiser::GCNDenoiser(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

	bindSlotsAndSignals();
}

void GCNDenoiser::bindSlotsAndSignals()
{
	connect(ui.button_load_noise, SIGNAL(clicked()), ui.mesh_viewer, SLOT(slotLoadNoise()));
	connect(ui.button_load_gt, SIGNAL(clicked()), ui.mesh_viewer, SLOT(slotLoadGT()));

	connect(ui.button_gen_noise, SIGNAL(clicked()), this, SLOT(slotGenNoise()));
	connect(this, SIGNAL(signalGenNoise(double, QString)), ui.mesh_viewer, SLOT(slotGenNoise(double, QString)));

	connect(ui.button_delete, SIGNAL(clicked()), ui.mesh_viewer, SLOT(slotDelete()));

	connect(ui.button_denoise, SIGNAL(clicked()), this, SLOT(slotDenoise()));
	connect(this, SIGNAL(signalDenoise(int, int)), ui.mesh_viewer, SLOT(slotDenoise(int, int)));
}

void GCNDenoiser::slotGenNoise()
{
	emit signalGenNoise(ui.box_level->value(), ui.box_type->currentText());
}

void GCNDenoiser::slotDenoise()
{
	emit signalDenoise(ui.box_gcns->value(), ui.box_iterations->value());
}