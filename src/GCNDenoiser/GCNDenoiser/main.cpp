#include "GCNDenoiser.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    GCNDenoiser w;
    w.show();
    return a.exec();
}
