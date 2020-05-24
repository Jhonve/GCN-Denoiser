#pragma once

#include "QtWidgets/QOpenGLWidget.h"
#include "QtGui/qopenglshaderprogram.h"
#include "qdebug.h"
#include "qopenglfunctions.h"

class Shader {
public:
	Shader(const QString& vertex_source_path, const QString& gragment_source_path);
	~Shader();

	QOpenGLShaderProgram shader_program;

	void use();
};
