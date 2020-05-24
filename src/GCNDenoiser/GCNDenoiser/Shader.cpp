#include "Shader.h"

Shader::Shader(const QString& vertex_shader_path, const QString& fragment_shader_path) {
	QOpenGLShader vertex_shader(QOpenGLShader::Vertex);
	bool success = vertex_shader.compileSourceFile(vertex_shader_path);
	if (!success) {
		qDebug() << "ERROR, SHADER VERTEX COMPILE FAILED" << endl;
		qDebug() << vertex_shader.log() << endl;
		return;
	}

	QOpenGLShader fragment_shader(QOpenGLShader::Fragment);
	success = fragment_shader.compileSourceFile(fragment_shader_path);
	if (!success) {
		qDebug() << "ERROR, SHADER FRAGMENT COMPILE FAILED" << endl;
		qDebug() << fragment_shader.log() << endl;
		return;
	}

	shader_program.addShader(&vertex_shader);
	shader_program.addShader(&fragment_shader);
	success = shader_program.link();
	if (!success) {
		qDebug() << "ERROR, SHADER PROGRAM LINKING FAILED" << endl;
		qDebug() << shader_program.log() << endl;
	}
}

Shader::~Shader() {
}

void Shader::use() {
	shader_program.bind();
}