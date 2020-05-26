#include <torch/script.h>
#include <torch/torch.h>

#include "MeshViewer.h"
#include <QmouseEvent>
#include <qcoreapplication.h>
#include <qdir.h>

#include "FlannKDTree.h"

#include "Noise.h"
#include "MeshNormalFiltering.h"

#include <omp.h>
#include <cstdio>
#include <ctime>

#include <QFileDialog>

GLuint VBOObj, VAOObj;

bool MeshViewer::m_transparent = false;

MeshViewer::MeshViewer(QWidget* parent)
	:QOpenGLWidget(parent),
	m_xRot(0),
	m_yRot(0),
	m_zRot(0)
{
	if (m_transparent)
	{
		QSurfaceFormat fmt = format();
		fmt.setAlphaBufferSize(8);
		setFormat(fmt);
	}

	setFocusPolicy(Qt::StrongFocus);

	m_data_manager = new DataManager();
	m_current_file_name = "./example_noisy.obj";
	meshInitializedGet(m_current_file_name.c_str(), false);
	meshInitializedGet("./example.obj", true);
	int start = m_current_file_name.find_last_of('/');
	int end = m_current_file_name.find_last_of('.');
	m_model_name = m_current_file_name.substr(start + 1, end - start - 1);
	m_current_noise_level = -1.0;
	m_current_noise_type = -1;
}

MeshViewer::~MeshViewer()
{
	if (m_data_manager != NULL)
	{
		delete m_data_manager;
	}

	cleanup();
}

void MeshViewer::meshInitializedGet(const char* file_name, bool is_original)
{
	std::cout << "Loading... mesh dir: " << file_name << std::endl;

	if (!is_original)
	{
		OpenMesh::IO::read_mesh(m_noised_tri_mesh, file_name);
		
		int new_num_faces = m_noised_tri_mesh.n_faces();
		if (m_num_faces == 0)
		{
			m_num_faces = new_num_faces;
		}
		else
		{
			if (new_num_faces != m_num_faces)
			{
				std::cout << "Not Match!" << std::endl;
				return;
			}
		}
		std::cout << "\tNumber of faces: " << m_num_faces << std::endl;

		m_is_have_noise = true;

		m_is_noise = true;
		m_is_gt = false;
		m_is_denoised = false;

		m_current_file_name = file_name;
		int start = m_current_file_name.find_last_of('/');
		int end = m_current_file_name.find_last_of('.');
		m_model_name = m_current_file_name.substr(start + 1, end - start - 1);

		m_data_manager->setNoisyMesh(m_noised_tri_mesh);
		m_data_manager->setDenoisedMesh(m_noised_tri_mesh);

		m_noised_tri_mesh.request_face_normals();
		m_noised_tri_mesh.request_vertex_normals();
		m_noised_tri_mesh.update_normals();

		m_center = OpenMesh::Vec3d(0., 0., 0.);
		for (TriMesh::VertexIter v_it = m_noised_tri_mesh.vertices_begin(); v_it != m_noised_tri_mesh.vertices_end(); v_it++)
		{
			OpenMesh::Vec3d center_position_temp = m_noised_tri_mesh.point(*v_it);
			m_center += center_position_temp;
		}
		m_center /= int(m_noised_tri_mesh.n_vertices());

		m_max = 0;
		for (TriMesh::VertexIter v_it = m_noised_tri_mesh.vertices_begin(); v_it != m_noised_tri_mesh.vertices_end(); v_it++)
		{
			OpenMesh::Vec3d position_temp = m_noised_tri_mesh.point(*v_it) - m_center;
			for (int i = 0; i < 3; i++)
			{
				double temp_max = abs(position_temp[i]);
				if (temp_max > m_max)
				{
					m_max = temp_max;
				}
			}
		}

		m_noised_mesh_visual_data = new GLfloat[3 * 9 * m_num_faces];
		m_denoised_mesh_visual_data = new GLfloat[3 * 9 * m_num_faces];

		for (TriMesh::VertexIter v_it = m_noised_tri_mesh.vertices_begin(); v_it != m_noised_tri_mesh.vertices_end(); v_it++)
		{
			m_noised_tri_mesh.point(*v_it) -= m_center;
			m_noised_tri_mesh.point(*v_it) /= m_max / 1.0;
			OpenMesh::Vec3d center_position_temp = m_noised_tri_mesh.point(*v_it);
			m_vertices_kd_tree.push_back(glm::vec3(center_position_temp[0], center_position_temp[1], center_position_temp[2]));
		}

		m_flann_kd_tree = shen::Geometry::EasyFlann(m_vertices_kd_tree);

		FaceNeighborType face_neighbor_type = kVertexBased;

		getAllFaceNeighbor(m_noised_tri_mesh, m_all_face_neighbor, face_neighbor_type, false);
		getFaceArea(m_noised_tri_mesh, m_face_area);
		getFaceCentroid(m_noised_tri_mesh, m_face_centroid);

		meshVisualization(m_noised_mesh_visual_data, m_noised_tri_mesh);
		meshVisualization(m_denoised_mesh_visual_data, m_noised_tri_mesh);
	}
	else
	{
		OpenMesh::IO::read_mesh(m_gt_tri_mesh, file_name);

		int new_num_faces = m_gt_tri_mesh.n_faces();
		if (m_num_faces == 0)
		{
			m_num_faces = new_num_faces;
		}
		else
		{
			if (new_num_faces != m_num_faces)
			{
				std::cout << "Not Match!" << std::endl;
				return;
			}
		}
		std::cout << "\tNumber of faces: " << m_num_faces << std::endl;

		m_current_file_name = file_name;
		int start = m_current_file_name.find_last_of('/');
		int end = m_current_file_name.find_last_of('.');
		m_model_name = m_current_file_name.substr(start + 1, end - start - 1);

		m_is_have_gt = true;

		m_is_gt = true;
		m_is_noise = false;
		m_is_denoised = false;

		m_data_manager->setOriginalMesh(m_gt_tri_mesh);

		m_gt_tri_mesh.request_face_normals();
		m_gt_tri_mesh.request_vertex_normals();
		m_gt_tri_mesh.update_normals();

		m_center = OpenMesh::Vec3d(0., 0., 0.);
		for (TriMesh::VertexIter v_it = m_gt_tri_mesh.vertices_begin(); v_it != m_gt_tri_mesh.vertices_end(); v_it++)
		{
			OpenMesh::Vec3d center_position_temp = m_gt_tri_mesh.point(*v_it);
			m_center += center_position_temp;
		}
		m_center /= int(m_gt_tri_mesh.n_vertices());

		m_max = 0;
		for (TriMesh::VertexIter v_it = m_gt_tri_mesh.vertices_begin(); v_it != m_gt_tri_mesh.vertices_end(); v_it++)
		{
			OpenMesh::Vec3d position_temp = m_gt_tri_mesh.point(*v_it) - m_center;
			for (int i = 0; i < 3; i++)
			{
				double temp_max = abs(position_temp[i]);
				if (temp_max > m_max)
				{
					m_max = temp_max;
				}
			}
		}

		m_gt_mesh_visual_data = new GLfloat[3 * 9 * m_num_faces];

		

		for (TriMesh::VertexIter v_it = m_gt_tri_mesh.vertices_begin(); v_it != m_gt_tri_mesh.vertices_end(); v_it++)
		{
			m_gt_tri_mesh.point(*v_it) -= m_center;
			m_gt_tri_mesh.point(*v_it) /= m_max / 1.0;
		}

		m_is_have_gt = true;

		meshVisualization(m_gt_mesh_visual_data, m_gt_tri_mesh);
	}
}

void MeshViewer::meshVisualization(GLfloat* m_mesh_visual_data, TriMesh& tri_mesh)
{
	int m_i = 0;
	for (TriMesh::FaceIter f_it = tri_mesh.faces_begin(); f_it != tri_mesh.faces_end(); f_it++)
	{
		OpenMesh::Vec3d normal = tri_mesh.normal(*f_it);

		int v_i = 0;
		for (TriMesh::FaceVertexIter fv_it = tri_mesh.fv_iter(*f_it); fv_it.is_valid(); fv_it++)
		{
			OpenMesh::Vec3d position = tri_mesh.point(*fv_it);

			// for position
			for (int i = 0; i < 3; i++)
			{
				m_mesh_visual_data[m_i * 3 * 9 + v_i * 9 + i] = position[i];
			}

			// for normal
			for (int i = 3; i < 6; i++)
			{
				m_mesh_visual_data[m_i * 3 * 9 + v_i * 9 + i] = normal[i - 3];
				m_mesh_visual_data[m_i * 3 * 9 + v_i * 9 + i + 3] = (normal[i - 3] + 1.) / 2.; // for normal map
			}
			v_i++;
		}
		m_i++;
	}
	m_is_reload = true;
	update();
}

QSize MeshViewer::minimumSizeHint() const
{
	return QSize(256, 256);
}

QSize MeshViewer::sizeHint() const
{
	return QSize(512, 512);
}

static void qNormalizeAngle(int &angle)
{
	while (angle < 0)
	{
		angle += 360 * 16;
	}
	while (angle > 360 * 16)
	{
		angle -= 360 * 16;
	}
}

void MeshViewer::setXRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != m_xRot)
	{
		m_xRot = angle;
		emit xRotationChanged(angle);
		update();
	}
}

void MeshViewer::setYRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != m_yRot)
	{
		m_yRot = angle;
		emit yRotationChanged(angle);
		update();
	}
}

void MeshViewer::setZRotation(int angle)
{
	qNormalizeAngle(angle);
	if (angle != m_zRot)
	{
		m_zRot = angle;
		emit zRotationChanged(angle);
		update();
	}
}

void MeshViewer::cleanup()
{
	if (&m_shader->shader_program == nullptr)
	{
		return;
	}
	makeCurrent();
	delete &m_shader->shader_program;
	doneCurrent();
}

void MeshViewer::initializeGL()
{
	// shader part
	m_core = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_4_5_Core>();
	m_shader = new Shader("ShaderFiles/vertex_shader_source.vert", "ShaderFiles/fragment_shader_source.frag");

	m_core->glGenVertexArrays(1, &VAOObj);
	m_core->glGenBuffers(1, &VBOObj);
	m_core->glBindVertexArray(VAOObj);
	m_core->glBindBuffer(GL_ARRAY_BUFFER, VBOObj);
	m_core->glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * (m_num_faces * 3 * 9), m_noised_mesh_visual_data, GL_STATIC_DRAW);
	m_core->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (void*)0);
	m_core->glEnableVertexAttribArray(0);
	m_core->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	m_core->glEnableVertexAttribArray(1);
	m_core->glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
	m_core->glEnableVertexAttribArray(2);

	m_core->glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	// m_core->glClearColor(0.7f, 0.7f, 0.7f, 1.0f);

	// set camera
	m_camera.setToIdentity();
	m_camera.translate(0, 0, -3);

	m_a_light_pos = QVector3D(0.0, 3.0, 6.0);
	m_b_light_pos = QVector3D(0.0, 3.0, -6.0);
	m_view_pos = QVector3D(0.0, 0.0, -1.0);

	m_is_reload = false;
	m_is_gt = false;
}

void MeshViewer::resizeGL(int width, int height)
{
	m_proj.setToIdentity();
	m_proj.perspective(45.0f, GLfloat(width) / height, 0.01f, 100.0f);
}

void MeshViewer::paintGL()
{
	m_core->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	m_core->glEnable(GL_DEPTH_TEST);
	// m_core->glEnable(GL_CULL_FACE);

	if (m_is_reload)
	{
		if (m_is_gt)
		{
			m_core->glGenVertexArrays(1, &VAOObj);
			m_core->glGenBuffers(1, &VBOObj);
			m_core->glBindVertexArray(VAOObj);
			m_core->glBindBuffer(GL_ARRAY_BUFFER, VBOObj);
			m_core->glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * (m_num_faces * 3 * 9), m_gt_mesh_visual_data, GL_STATIC_DRAW);
			m_core->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (void*)0);
			m_core->glEnableVertexAttribArray(0);
			m_core->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
			m_core->glEnableVertexAttribArray(1);
			m_core->glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
			m_core->glEnableVertexAttribArray(2);
		}
		else if (m_is_denoised)
		{
			m_core->glGenVertexArrays(1, &VAOObj);
			m_core->glGenBuffers(1, &VBOObj);
			m_core->glBindVertexArray(VAOObj);
			m_core->glBindBuffer(GL_ARRAY_BUFFER, VBOObj);
			m_core->glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * (m_num_faces * 3 * 9), m_denoised_mesh_visual_data, GL_STATIC_DRAW);
			m_core->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (void*)0);
			m_core->glEnableVertexAttribArray(0);
			m_core->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
			m_core->glEnableVertexAttribArray(1);
			m_core->glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
			m_core->glEnableVertexAttribArray(2);
		}
		else if (m_is_noise)
		{
			m_core->glGenVertexArrays(1, &VAOObj);
			m_core->glGenBuffers(1, &VBOObj);
			m_core->glBindVertexArray(VAOObj);
			m_core->glBindBuffer(GL_ARRAY_BUFFER, VBOObj);
			m_core->glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * (m_num_faces * 3 * 9), m_noised_mesh_visual_data, GL_STATIC_DRAW);
			m_core->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (void*)0);
			m_core->glEnableVertexAttribArray(0);
			m_core->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
			m_core->glEnableVertexAttribArray(1);
			m_core->glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
			m_core->glEnableVertexAttribArray(2);
		}

		m_is_reload = false;
	}

	m_world.setToIdentity();
	m_world.rotate(m_xRot / 16.0f, 1, 0, 0);
	m_world.rotate(m_yRot / 16.0f, 0, 1, 0);
	m_world.rotate(m_zRot / 16.0f, 0, 0, 1);

	m_shader->use();
	m_core->glBindVertexArray(VAOObj);
	m_projMatrixLoc = m_shader->shader_program.uniformLocation("proj_mat");
	m_modelMatrixLoc = m_shader->shader_program.uniformLocation("model_mat");
	m_viewMatrixLoc = m_shader->shader_program.uniformLocation("view_mat");
	m_aLightPosLoc = m_shader->shader_program.uniformLocation("a_light_pos");
	m_bLightPosLoc = m_shader->shader_program.uniformLocation("b_light_pos");
	m_viewPosLoc = m_shader->shader_program.uniformLocation("view_pos");

	m_shader->shader_program.setUniformValue(m_projMatrixLoc, m_proj);
	m_shader->shader_program.setUniformValue(m_modelMatrixLoc, m_world);
	m_shader->shader_program.setUniformValue(m_viewMatrixLoc, m_camera);
	m_shader->shader_program.setUniformValue(m_aLightPosLoc, m_a_light_pos);
	m_shader->shader_program.setUniformValue(m_bLightPosLoc, m_b_light_pos);
	m_shader->shader_program.setUniformValue(m_viewPosLoc, m_view_pos);
	m_core->glDrawArrays(GL_TRIANGLES, 0, m_num_faces * 3);

	m_shader->shader_program.release();
}

void MeshViewer::keyPressEvent(QKeyEvent *event)
{
	switch (event->key())
	{
	case Qt::Key_Escape:
		close();
	case Qt::Key_N:
		if (!m_is_have_noise)
		{
			std::cout << "No noise model, load one first..." << std::endl;
			break;
		}

		m_is_reload = true;

		m_is_noise = true;
		m_is_gt = false;
		m_is_denoised = false;

		update();
		break;
	case Qt::Key_D:
		if (!m_is_have_noise)
		{
			std::cout << "No noise model, load one first..." << std::endl;
			break;
		}

		m_is_reload = true;

		m_is_denoised = true;
		m_is_noise = false;
		m_is_gt = false;

		update();
		break;
	case Qt::Key_G:
		if (!m_is_have_gt)
		{
			std::cout << "No gt model, load one first..." << std::endl;
			break;
		}

		m_is_reload = true;

		m_is_gt = true;
		m_is_noise = false;
		m_is_denoised = false;

		update();
		break;
	default:
		break;
	}
}

void MeshViewer::mousePressEvent(QMouseEvent *event)
{
	m_lastPos = event->pos();
}

void MeshViewer::mouseMoveEvent(QMouseEvent *event)
{
	int dx = event->x() - m_lastPos.x();
	int dy = event->y() - m_lastPos.y();

	if (event->buttons() & Qt::LeftButton)
	{
		setXRotation(m_xRot + 8 * dy);
		setYRotation(m_yRot + 8 * dx);
	}
	else if (event->buttons() & Qt::RightButton)
	{
		setXRotation(m_xRot + 8 * dy);
		setZRotation(m_zRot + 8 * dx);
	}
	else if (event->buttons() & Qt::MidButton)
	{
		m_camera.translate((GLfloat)dx / 600, -1 * (GLfloat)dy / 600, 0);
		update();
	}
	m_lastPos = event->pos();
}

void MeshViewer::wheelEvent(QWheelEvent *event)
{
	if (event->delta() > 0)
	{
		m_camera.translate(0, 0, 0.05f);
		update();
	}
	else
	{
		m_camera.translate(0, 0, -0.05f);
		update();
	}
}

void MeshViewer::getFaceNeighbor(TriMesh &mesh, TriMesh::FaceHandle fh, FaceNeighborType face_neighbor_type, std::vector<TriMesh::FaceHandle> &face_neighbor)
{
	face_neighbor.clear();
	if (face_neighbor_type == kEdgeBased)
	{
		for (TriMesh::FaceFaceIter ff_it = mesh.ff_iter(fh); ff_it.is_valid(); ff_it++)
			face_neighbor.push_back(*ff_it);
	}
	else if (face_neighbor_type == kVertexBased)
	{
		std::set<int> neighbor_face_index; neighbor_face_index.clear();

		for (TriMesh::FaceVertexIter fv_it = mesh.fv_begin(fh); fv_it.is_valid(); fv_it++)
		{
			for (TriMesh::VertexFaceIter vf_it = mesh.vf_iter(*fv_it); vf_it.is_valid(); vf_it++)
			{
				if ((*vf_it) != fh)
					neighbor_face_index.insert(vf_it->idx());
			}
		}

		for (std::set<int>::iterator iter = neighbor_face_index.begin(); iter != neighbor_face_index.end(); ++iter)
		{
			face_neighbor.push_back(TriMesh::FaceHandle(*iter));
		}
	}
}

void MeshViewer::getAllFaceNeighbor(TriMesh &mesh, std::vector<std::vector<TriMesh::FaceHandle> > &all_face_neighbor, FaceNeighborType face_neighbor_type, bool include_central_face)
{
	all_face_neighbor.resize(mesh.n_faces());
	for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
	{
		std::vector<TriMesh::FaceHandle> face_neighbor;
		getFaceNeighbor(mesh, *f_it, face_neighbor_type, face_neighbor);
		if (include_central_face) face_neighbor.push_back(*f_it);
		all_face_neighbor[f_it->idx()] = face_neighbor;
	}
}

void MeshViewer::getFaceArea(TriMesh &mesh, std::vector<double> &area)
{
	area.resize(mesh.n_faces());

	for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
	{
		std::vector<TriMesh::Point> point;
		point.resize(3); int index = 0;
		for (TriMesh::FaceVertexIter fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); fv_it++)
		{
			point[index] = mesh.point(*fv_it);
			index++;
		}
		TriMesh::Point edge1 = point[1] - point[0];
		TriMesh::Point edge2 = point[1] - point[2];
		double S = 0.5 * (edge1 % edge2).length();
		area[(*f_it).idx()] = S;
	}
}

void MeshViewer::getFaceCentroid(TriMesh &mesh, std::vector<TriMesh::Point> &centroid)
{
	centroid.resize(mesh.n_faces(), TriMesh::Point(0.0, 0.0, 0.0));
	for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
	{
		TriMesh::Point pt = mesh.calc_face_centroid(*f_it);
		centroid[(*f_it).idx()] = pt;
	}
}

void MeshViewer::slotLoadNoise()
{
	if (m_is_have_noise)
	{
		std::cout << "Existing noisy model, delete first." << std::endl;
		return;
	}

	QString q_noise_mesh_file = QFileDialog::getOpenFileName(this, tr("Open File"), "", tr("OBJ, OFF(*.obj *.off)"));
	
	if (q_noise_mesh_file == "")
	{
		return;
	}

	std::string noise_model_file = q_noise_mesh_file.toStdString();

	meshInitializedGet(noise_model_file.c_str(), false);
}

void MeshViewer::slotLoadGT()
{
	if (m_is_have_gt)
	{
		std::cout << "Existing gt model, delete first." << std::endl;
		return;
	}

	QString q_gt_mesh_file = QFileDialog::getOpenFileName(this, tr("Open File"), "", tr("OBJ, OFF(*.obj *.off)"));

	if (q_gt_mesh_file == "" || m_is_have_gt)
	{
		return;
	}

	std::string gt_model_file = q_gt_mesh_file.toStdString();

	meshInitializedGet(gt_model_file.c_str(), true);
}

void MeshViewer::slotDelete()
{
	if (!m_is_have_gt && !m_is_have_noise)
	{
		return;
	}

	if (m_is_have_noise)
	{
		m_noised_tri_mesh.clean();
		m_vertices_kd_tree.clear();
		m_all_face_neighbor.clear();
		m_face_area.clear();
		m_face_centroid.clear();

		delete m_noised_mesh_visual_data;
		delete m_denoised_mesh_visual_data;

		m_is_have_noise = false;
	}

	if (m_is_have_gt)
	{
		m_gt_tri_mesh.clean();
		delete m_gt_mesh_visual_data;

		m_is_have_gt = false;
	}

	delete m_data_manager;
	m_data_manager = new DataManager();
	
	m_num_faces = 0;
	m_is_reload = true;
	update();
}

void MeshViewer::slotGenNoise(double noise_level, QString noise_type)
{
	if (!m_is_have_gt)
	{
		std::cout << "Load GT first!" << std::endl;
		return;
	}

	if (m_is_have_noise)
	{
		m_noised_tri_mesh.clean();
		m_vertices_kd_tree.clear();
		m_all_face_neighbor.clear();
		m_face_area.clear();
		m_face_centroid.clear();

		delete m_noised_mesh_visual_data;
		delete m_denoised_mesh_visual_data;
	}

	m_is_have_noise = true;
	m_data_manager->MeshToOriginalMesh();

	Noise* noise;
	if (noise_type == "Gaussian")
	{
		noise = new Noise(m_data_manager, noise_level, 0);
	}
	else
	{
		noise = new Noise(m_data_manager, noise_level, 1);
	}
	noise->addNoise();

	m_noised_tri_mesh = m_data_manager->getNoisyMesh();

	m_noised_tri_mesh.request_face_normals();
	m_noised_tri_mesh.request_vertex_normals();
	m_noised_tri_mesh.update_normals();

	m_noised_mesh_visual_data = new GLfloat[3 * 9 * m_num_faces];
	m_denoised_mesh_visual_data = new GLfloat[3 * 9 * m_num_faces];

	m_center = OpenMesh::Vec3d(0., 0., 0.);
	for (TriMesh::VertexIter v_it = m_noised_tri_mesh.vertices_begin(); v_it != m_noised_tri_mesh.vertices_end(); v_it++)
	{
		OpenMesh::Vec3d center_position_temp = m_noised_tri_mesh.point(*v_it);
		m_center += center_position_temp;
	}
	m_center /= int(m_noised_tri_mesh.n_vertices());

	m_max = 0;
	for (TriMesh::VertexIter v_it = m_noised_tri_mesh.vertices_begin(); v_it != m_noised_tri_mesh.vertices_end(); v_it++)
	{
		OpenMesh::Vec3d position_temp = m_noised_tri_mesh.point(*v_it) - m_center;
		for (int i = 0; i < 3; i++)
		{
			double temp_max = abs(position_temp[i]);
			if (temp_max > m_max)
			{
				m_max = temp_max;
			}
		}
	}

	for (TriMesh::VertexIter v_it = m_noised_tri_mesh.vertices_begin(); v_it != m_noised_tri_mesh.vertices_end(); v_it++)
	{
		m_noised_tri_mesh.point(*v_it) -= m_center;
		m_noised_tri_mesh.point(*v_it) /= m_max / 1.0;
		OpenMesh::Vec3d center_position_temp = m_noised_tri_mesh.point(*v_it);
		m_vertices_kd_tree.push_back(glm::vec3(center_position_temp[0], center_position_temp[1], center_position_temp[2]));
	}

	m_flann_kd_tree = shen::Geometry::EasyFlann(m_vertices_kd_tree);

	FaceNeighborType face_neighbor_type = kVertexBased;

	getAllFaceNeighbor(m_noised_tri_mesh, m_all_face_neighbor, face_neighbor_type, false);
	getFaceArea(m_noised_tri_mesh, m_face_area);
	getFaceCentroid(m_noised_tri_mesh, m_face_centroid);

	m_is_noise = true;
	m_is_denoised = false;
	m_is_gt = false;

	meshVisualization(m_noised_mesh_visual_data, m_noised_tri_mesh);
	meshVisualization(m_denoised_mesh_visual_data, m_noised_tri_mesh);
}

void MeshViewer::slotDenoise(int gcns, int normal_iterations)
{
	if (!m_is_have_noise)
	{
		std::cout << "No noisy model, load one first..." << std::endl;
		return;
	}
	if (!m_is_have_gt)
	{
		std::cout << "No gt model, load one first for calculating error..." << std::endl;
		return;
	}

	torch::jit::script::Module GCN_1;
	torch::jit::script::Module GCN_2;

	try
	{
		GCN_1 = torch::jit::load("./NetworkModel/script_model_1.pt");
		GCN_2 = torch::jit::load("./NetworkModel/script_model_2.pt");
	}
	catch (const c10::Error& e)
	{
		std::cerr << "Error of loading pre-trained models!" << std::endl;
		return;
	}
	GCN_1.to(at::kCUDA);
	GCN_1.eval();
	GCN_2.to(at::kCUDA);
	GCN_2.eval();
	std::cout << "Pre-trained models load success!" << std::endl;

	// pre-trained parameters
	int p_num_ring = 2;
	int p_radius_region = 16; // 16-->64
	int p_num_neighbors = 64; // 64-->256
	int p_features = 17;
	int batch_size = 720; // 720-->160

	int num_batches = m_num_faces / batch_size;
	int num_last_batch = m_num_faces - (num_batches * batch_size);

	std::vector<TriMesh::Normal> predicted_normal(m_num_faces);

	std::cout << "Start denoising..." << std::endl;
	TriMesh::FaceIter all_f_it = m_noised_tri_mesh.faces_begin();
	TriMesh::FaceIter all_gt_f_it = m_gt_tri_mesh.faces_begin();
	for (int i_batch = 0; i_batch < num_batches; i_batch++)
	{
		std::cout << "\tBatch idx: " << i_batch << "/" << num_batches;
		printf("\33[2k\r");
		std::vector<at::Tensor> input_tensors(batch_size);
		std::vector<glm::dmat3x3> trans_mats(batch_size);
		std::vector<bool> is_valid(batch_size);
		std::vector<TriMesh::Normal> noisy_normal(batch_size);

		int start_face_idx = i_batch * batch_size;
		int end_face_idx = (i_batch + 1) * batch_size;

#pragma omp parallel for
		for (int i_face = start_face_idx; i_face < end_face_idx; i_face++)
		{
			TriMesh::FaceIter f_it = all_f_it;
			TriMesh::FaceIter gt_f_it = all_gt_f_it;

			for (int i = start_face_idx; i < i_face; i++)
			{
				f_it++;
				gt_f_it++;
			}

			OpenMesh::Vec3d gt_normal_om = m_gt_tri_mesh.normal(*gt_f_it);
			glm::dvec3 gt_normal = glm::dvec3(gt_normal_om[0], gt_normal_om[1], gt_normal_om[2]);
			PatchData* patch_data = new PatchData(m_noised_tri_mesh, m_all_face_neighbor, m_face_area, m_face_centroid, f_it, m_flann_kd_tree, gt_normal, p_num_ring, p_radius_region);
			at::Tensor temp_input_tensor;
			if (patch_data->m_patch_num_faces == 1 || patch_data->m_aligned_patch_num_faces <= 1)
			{
				// std::cout << "false" << std::endl;
				is_valid[i_face - start_face_idx] = false;
				noisy_normal[i_face - start_face_idx] = m_noised_tri_mesh.normal(*f_it);
				temp_input_tensor = torch::zeros({ 1, 20, p_num_neighbors });
				temp_input_tensor = temp_input_tensor.toType(c10::ScalarType::Float);
			}
			else
			{
				is_valid[i_face - start_face_idx] = true;
				noisy_normal[i_face - start_face_idx] = m_noised_tri_mesh.normal(*f_it);

				int num_patch_faces = patch_data->m_aligned_patch_num_faces;
				trans_mats[i_face - start_face_idx] = patch_data->m_matrix;
				temp_input_tensor = torch::from_blob(patch_data->m_temp_network_input, { 1, num_patch_faces, 17 + 3 }, c10::ScalarType::Double);

				temp_input_tensor = temp_input_tensor.toType(c10::ScalarType::Float);

				if (num_patch_faces >= p_num_neighbors)
				{
					temp_input_tensor = temp_input_tensor.slice(1, 0, p_num_neighbors);
				}
				else
				{
					at::Tensor temp_tensor_cat = torch::zeros({ 1, p_num_neighbors - num_patch_faces, 20 });
					temp_input_tensor = torch::cat({ temp_input_tensor, temp_tensor_cat }, 1);
				}

				temp_input_tensor = temp_input_tensor.permute({ 0, 2, 1 });
			}

			input_tensors[i_face - start_face_idx] = temp_input_tensor;
			delete patch_data;
		}

		std::vector<at::Tensor> batch_tensors_vec;
		for (int i_s = 0; i_s < batch_size; i_s++)
		{
			batch_tensors_vec.push_back(input_tensors[i_s]);
		}
		at::Tensor batch_tensors = torch::cat(batch_tensors_vec, 0);

		std::vector<torch::jit::IValue> jit_inputs;
		jit_inputs.push_back(batch_tensors.to(at::kCUDA));
		at::Tensor output = GCN_1.forward(jit_inputs).toTensor();
		output = output.cpu();
		auto out_reader = output.accessor<float, 2>();

#pragma omp parallel for
		for (int i_s = 0; i_s < batch_size; i_s++)
		{
			if (is_valid[i_s] == true)
			{
				float x = out_reader[i_s][0];
				float y = out_reader[i_s][1];
				float z = out_reader[i_s][2];
				glm::dvec3 temp_res((double)x, (double)y, (double)z);
				temp_res = glm::normalize(temp_res);
				temp_res = trans_mats[i_s] * temp_res;
				TriMesh::Normal temp_normal(temp_res[0], temp_res[1], temp_res[2]);
				predicted_normal[i_batch * batch_size + i_s] = temp_normal;
			}
			else
			{
				predicted_normal[i_batch * batch_size + i_s] = noisy_normal[i_s];
			}
		}

		batch_tensors_vec.clear();
		jit_inputs.clear();

		input_tensors.clear();
		trans_mats.clear();
		is_valid.clear();
		noisy_normal.clear();

		for (int i = 0; i < batch_size; i++)
		{
			all_f_it++;
			all_gt_f_it++;
		}
	}

	if (num_last_batch > 0)
	{
		std::cout << "\tBatch idx: " << num_batches << "/" << num_batches;
		printf("\33[2k\r");
		std::vector<at::Tensor> input_tensors(num_last_batch);
		std::vector<glm::dmat3x3> trans_mats(num_last_batch);
		std::vector<bool> is_valid(num_last_batch);
		std::vector<TriMesh::Normal> noisy_normal(num_last_batch);

		int start_face_idx = num_batches * batch_size;
		int end_face_idx = num_batches * batch_size + num_last_batch;

#pragma omp parallel for
		for (int i_face = start_face_idx; i_face < end_face_idx; i_face++)
		{
			TriMesh::FaceIter f_it = m_noised_tri_mesh.faces_begin();
			TriMesh::FaceIter gt_f_it = m_gt_tri_mesh.faces_begin();

			for (int i = 0; i < i_face; i++)
			{
				f_it++;
				gt_f_it++;
			}

			OpenMesh::Vec3d gt_normal_om = m_gt_tri_mesh.normal(*gt_f_it);
			glm::dvec3 gt_normal = glm::dvec3(gt_normal_om[0], gt_normal_om[1], gt_normal_om[2]);
			PatchData* patch_data = new PatchData(m_noised_tri_mesh, m_all_face_neighbor, m_face_area, m_face_centroid, f_it, m_flann_kd_tree, gt_normal, p_num_ring, p_radius_region);
			at::Tensor temp_input_tensor;
			if (patch_data->m_patch_num_faces == 1 || patch_data->m_aligned_patch_num_faces <= 1)
			{
				is_valid[i_face - start_face_idx] = false;
				noisy_normal[i_face - start_face_idx] = m_noised_tri_mesh.normal(*f_it);
				temp_input_tensor = torch::zeros({ 1, 20, p_num_neighbors });
				temp_input_tensor = temp_input_tensor.toType(c10::ScalarType::Float);
			}
			else
			{
				is_valid[i_face - start_face_idx] = true;
				noisy_normal[i_face - start_face_idx] = m_noised_tri_mesh.normal(*f_it);

				int num_patch_faces = patch_data->m_aligned_patch_num_faces;
				trans_mats[i_face - start_face_idx] = patch_data->m_matrix;
				temp_input_tensor = torch::from_blob(patch_data->m_temp_network_input, { 1, num_patch_faces, 17 + 3 }, c10::ScalarType::Double);

				temp_input_tensor = temp_input_tensor.toType(c10::ScalarType::Float);

				if (num_patch_faces >= p_num_neighbors)
				{
					temp_input_tensor = temp_input_tensor.slice(1, 0, p_num_neighbors);
				}
				else
				{
					at::Tensor temp_tensor_cat = torch::zeros({ 1, p_num_neighbors - num_patch_faces, 20 });
					temp_input_tensor = torch::cat({ temp_input_tensor, temp_tensor_cat }, 1);
				}

				temp_input_tensor = temp_input_tensor.permute({ 0, 2, 1 });
			}

			input_tensors[i_face - start_face_idx] = temp_input_tensor;
			delete patch_data;
		}

		std::vector<at::Tensor> batch_tensors_vec;
		for (int i_s = 0; i_s < num_last_batch; i_s++)
		{
			batch_tensors_vec.push_back(input_tensors[i_s]);
		}
		at::Tensor batch_tensors = torch::cat(batch_tensors_vec, 0);

		std::vector<torch::jit::IValue> jit_inputs;
		jit_inputs.push_back(batch_tensors.to(at::kCUDA));

		at::Tensor output = GCN_1.forward(jit_inputs).toTensor();
		output = output.cpu();
		auto out_reader = output.accessor<float, 2>();

#pragma omp parallel for
		for (int i_s = 0; i_s < num_last_batch; i_s++)
		{
			if (is_valid[i_s] == true)
			{
				float x = out_reader[i_s][0];
				float y = out_reader[i_s][1];
				float z = out_reader[i_s][2];
				glm::dvec3 temp_res((double)x, (double)y, (double)z);
				temp_res = glm::normalize(temp_res);
				temp_res = trans_mats[i_s] * temp_res;
				TriMesh::Normal temp_normal(temp_res[0], temp_res[1], temp_res[2]);
				predicted_normal[num_batches * batch_size + i_s] = temp_normal;
			}
			else
			{
				predicted_normal[num_batches * batch_size + i_s] = noisy_normal[i_s];
			}
		}

		batch_tensors_vec.clear();
		jit_inputs.clear();

		input_tensors.clear();
		trans_mats.clear();
		is_valid.clear();
		noisy_normal.clear();
	}

	std::cout << std::endl;
	std::cout << "\tNetwork finish!" << std::endl;
	std::cout << "\tPredicted Normal Generation Finish!" << std::endl;

	/*----------------------------------------------------------*/

	std::vector<TriMesh::Normal> gt_normals;
	for (TriMesh::FaceIter f_it = m_gt_tri_mesh.faces_begin(); f_it != m_gt_tri_mesh.faces_end(); f_it++)
	{
		gt_normals.push_back(m_gt_tri_mesh.normal(f_it));
	}

	MeshNormalFiltering *mesh_denoising = new MeshNormalFiltering(m_data_manager);

	if (gcns == 1)
	{
		mesh_denoising->denoiseWithPredictedNormal(predicted_normal, normal_iterations);
	}
	else
	{
		mesh_denoising->denoiseWithPredictedNormal(predicted_normal, 1);
	}

	std::cout << "\tFirst Denoising Finished! Start calculate Error..." << std::endl;
	double error = mesh_denoising->getError();
	std::cout << "First Finish! Ea: " << error << std::endl;
	// double ea = mesh_denoising->getErrorMSAE();
	// std::cout << "First Finish! MSE: " << error << " MSAE: " << ea << std::endl;
	// double dv = mesh_denoising->getErrorDv();
	// std::cout << "First Finish! MSE: " << error << " MSAE: " << ea << " Dv: " << dv << std::endl;

	std::string res_name = "./Results/Denoised_";
	std::string tail = "_1.obj";
	res_name = res_name + m_model_name + tail;
	m_data_manager->ExportMeshToFile(res_name);
	std::cout << "Result Save Success!" << std::endl;

	predicted_normal.clear();

	if (gcns > 1)
	{
		std::cout << "Start second denoising..." << std::endl;
		TriMesh temp_denoised_mesh = m_data_manager->getDenoisedMesh();

		std::vector<TriMesh::Normal> predicted_normal_exten(m_num_faces);
		std::vector<double> face_area;
		std::vector<TriMesh::Point> face_centroid;

		std::vector<glm::vec3> vertices_kd_tree;
		for (TriMesh::VertexIter v_it = temp_denoised_mesh.vertices_begin(); v_it != temp_denoised_mesh.vertices_end(); v_it++)
		{
			temp_denoised_mesh.point(*v_it) -= m_center;
			temp_denoised_mesh.point(*v_it) /= m_max / 1.0;
			OpenMesh::Vec3d center_position_temp = temp_denoised_mesh.point(*v_it);
			vertices_kd_tree.push_back(glm::vec3(center_position_temp[0], center_position_temp[1], center_position_temp[2]));
		}

		shen::Geometry::EasyFlann flann_kd_tree = shen::Geometry::EasyFlann(vertices_kd_tree);
		getFaceArea(temp_denoised_mesh, face_area);
		getFaceCentroid(temp_denoised_mesh, face_centroid);

		TriMesh::FaceIter all_f_it_fine = m_noised_tri_mesh.faces_begin();
		TriMesh::FaceIter all_gt_f_it_fine = m_gt_tri_mesh.faces_begin();
		for (int i_batch = 0; i_batch < num_batches; i_batch++)
		{
			std::cout << "\tBatch idx: " << i_batch << "/" << num_batches;
			printf("\33[2k\r");
			std::vector<at::Tensor> input_tensors(batch_size);
			std::vector<glm::dmat3x3> trans_mats(batch_size);
			std::vector<bool> is_valid(batch_size);
			std::vector<TriMesh::Normal> noisy_normal(batch_size);

			int start_face_idx = i_batch * batch_size;
			int end_face_idx = (i_batch + 1) * batch_size;

#pragma omp parallel for
			for (int i_face = start_face_idx; i_face < end_face_idx; i_face++)
			{
				TriMesh::FaceIter f_it = all_f_it_fine;
				TriMesh::FaceIter gt_f_it = all_gt_f_it_fine;

				for (int i = start_face_idx; i < i_face; i++)
				{
					f_it++;
					gt_f_it++;
				}

				OpenMesh::Vec3d gt_normal_om = m_gt_tri_mesh.normal(*gt_f_it);
				glm::dvec3 gt_normal = glm::dvec3(gt_normal_om[0], gt_normal_om[1], gt_normal_om[2]);
				PatchData* patch_data = new PatchData(temp_denoised_mesh, m_all_face_neighbor, face_area, face_centroid, f_it, flann_kd_tree, gt_normal, p_num_ring, p_radius_region);
				at::Tensor temp_input_tensor;
				if (patch_data->m_patch_num_faces == 1 || patch_data->m_aligned_patch_num_faces <= 1)
				{
					// std::cout << "false" << std::endl;
					is_valid[i_face - start_face_idx] = false;
					noisy_normal[i_face - start_face_idx] = m_noised_tri_mesh.normal(*f_it);
					temp_input_tensor = torch::zeros({ 1, 20, p_num_neighbors });
					temp_input_tensor = temp_input_tensor.toType(c10::ScalarType::Float);
				}
				else
				{
					is_valid[i_face - start_face_idx] = true;
					noisy_normal[i_face - start_face_idx] = m_noised_tri_mesh.normal(*f_it);

					int num_patch_faces = patch_data->m_aligned_patch_num_faces;
					trans_mats[i_face - start_face_idx] = patch_data->m_matrix;
					temp_input_tensor = torch::from_blob(patch_data->m_temp_network_input, { 1, num_patch_faces, 17 + 3 }, c10::ScalarType::Double);

					temp_input_tensor = temp_input_tensor.toType(c10::ScalarType::Float);

					if (num_patch_faces >= p_num_neighbors)
					{
						temp_input_tensor = temp_input_tensor.slice(1, 0, p_num_neighbors);
					}
					else
					{
						at::Tensor temp_tensor_cat = torch::zeros({ 1, p_num_neighbors - num_patch_faces, 20 });
						temp_input_tensor = torch::cat({ temp_input_tensor, temp_tensor_cat }, 1);
					}

					temp_input_tensor = temp_input_tensor.permute({ 0, 2, 1 });
				}

				input_tensors[i_face - start_face_idx] = temp_input_tensor;
				delete patch_data;
			}

			std::vector<at::Tensor> batch_tensors_vec;
			for (int i_s = 0; i_s < batch_size; i_s++)
			{
				batch_tensors_vec.push_back(input_tensors[i_s]);
			}
			at::Tensor batch_tensors = torch::cat(batch_tensors_vec, 0);

			std::vector<torch::jit::IValue> jit_inputs;
			jit_inputs.push_back(batch_tensors.to(at::kCUDA));
			at::Tensor output = GCN_2.forward(jit_inputs).toTensor();
			output = output.cpu();
			auto out_reader = output.accessor<float, 2>();

#pragma omp parallel for
			for (int i_s = 0; i_s < batch_size; i_s++)
			{
				if (is_valid[i_s] == true)
				{
					float x = out_reader[i_s][0];
					float y = out_reader[i_s][1];
					float z = out_reader[i_s][2];
					glm::dvec3 temp_res((double)x, (double)y, (double)z);
					temp_res = glm::normalize(temp_res);
					temp_res = trans_mats[i_s] * temp_res;
					TriMesh::Normal temp_normal(temp_res[0], temp_res[1], temp_res[2]);
					predicted_normal_exten[i_batch * batch_size + i_s] = temp_normal;
				}
				else
				{
					predicted_normal_exten[i_batch * batch_size + i_s] = noisy_normal[i_s];
				}
			}

			batch_tensors_vec.clear();
			jit_inputs.clear();

			input_tensors.clear();
			trans_mats.clear();
			is_valid.clear();
			noisy_normal.clear();

			for (int i = 0; i < batch_size; i++)
			{
				all_f_it_fine++;
				all_gt_f_it_fine++;
			}
		}

		if (num_last_batch > 0)
		{
			std::cout << "\tBatch idx: " << num_batches << "/" << num_batches;
			printf("\33[2k\r");
			std::vector<at::Tensor> input_tensors(num_last_batch);
			std::vector<glm::dmat3x3> trans_mats(num_last_batch);
			std::vector<bool> is_valid(num_last_batch);
			std::vector<TriMesh::Normal> noisy_normal(num_last_batch);

			int start_face_idx = num_batches * batch_size;
			int end_face_idx = num_batches * batch_size + num_last_batch;

#pragma omp parallel for
			for (int i_face = start_face_idx; i_face < end_face_idx; i_face++)
			{
				TriMesh::FaceIter f_it = all_f_it_fine;
				TriMesh::FaceIter gt_f_it = all_gt_f_it_fine;

				for (int i = start_face_idx; i < i_face; i++)
				{
					f_it++;
					gt_f_it++;
				}

				OpenMesh::Vec3d gt_normal_om = m_gt_tri_mesh.normal(*gt_f_it);
				glm::dvec3 gt_normal = glm::dvec3(gt_normal_om[0], gt_normal_om[1], gt_normal_om[2]);
				PatchData* patch_data = new PatchData(temp_denoised_mesh, m_all_face_neighbor, face_area, face_centroid, f_it, flann_kd_tree, gt_normal, p_num_ring, p_radius_region);
				at::Tensor temp_input_tensor;
				if (patch_data->m_patch_num_faces == 1 || patch_data->m_aligned_patch_num_faces <= 1)
				{
					is_valid[i_face - start_face_idx] = false;
					noisy_normal[i_face - start_face_idx] = m_noised_tri_mesh.normal(*f_it);
					temp_input_tensor = torch::zeros({ 1, 20, p_num_neighbors });
					temp_input_tensor = temp_input_tensor.toType(c10::ScalarType::Float);
				}
				else
				{
					is_valid[i_face - start_face_idx] = true;
					noisy_normal[i_face - start_face_idx] = m_noised_tri_mesh.normal(*f_it);

					int num_patch_faces = patch_data->m_aligned_patch_num_faces;
					trans_mats[i_face - start_face_idx] = patch_data->m_matrix;
					temp_input_tensor = torch::from_blob(patch_data->m_temp_network_input, { 1, num_patch_faces, 17 + 3 }, c10::ScalarType::Double);

					temp_input_tensor = temp_input_tensor.toType(c10::ScalarType::Float);

					if (num_patch_faces >= p_num_neighbors)
					{
						temp_input_tensor = temp_input_tensor.slice(1, 0, p_num_neighbors);
					}
					else
					{
						at::Tensor temp_tensor_cat = torch::zeros({ 1, p_num_neighbors - num_patch_faces, 20 });
						temp_input_tensor = torch::cat({ temp_input_tensor, temp_tensor_cat }, 1);
					}

					temp_input_tensor = temp_input_tensor.permute({ 0, 2, 1 });
				}

				input_tensors[i_face - start_face_idx] = temp_input_tensor;
				delete patch_data;
			}

			std::vector<at::Tensor> batch_tensors_vec;
			for (int i_s = 0; i_s < num_last_batch; i_s++)
			{
				batch_tensors_vec.push_back(input_tensors[i_s]);
			}
			at::Tensor batch_tensors = torch::cat(batch_tensors_vec, 0);

			std::vector<torch::jit::IValue> jit_inputs;
			jit_inputs.push_back(batch_tensors.to(at::kCUDA));

			at::Tensor output = GCN_2.forward(jit_inputs).toTensor();
			output = output.cpu();
			auto out_reader = output.accessor<float, 2>();

#pragma omp parallel for
			for (int i_s = 0; i_s < num_last_batch; i_s++)
			{
				if (is_valid[i_s] == true)
				{
					float x = out_reader[i_s][0];
					float y = out_reader[i_s][1];
					float z = out_reader[i_s][2];
					glm::dvec3 temp_res((double)x, (double)y, (double)z);
					temp_res = glm::normalize(temp_res);
					temp_res = trans_mats[i_s] * temp_res;
					TriMesh::Normal temp_normal(temp_res[0], temp_res[1], temp_res[2]);
					predicted_normal_exten[num_batches * batch_size + i_s] = temp_normal;
				}
				else
				{
					predicted_normal_exten[num_batches * batch_size + i_s] = noisy_normal[i_s];
				}
			}

			batch_tensors_vec.clear();
			jit_inputs.clear();

			input_tensors.clear();
			trans_mats.clear();
			is_valid.clear();
			noisy_normal.clear();
		}

		std::cout << std::endl;
		std::cout << "\tNetwork finish!" << std::endl;
		std::cout << "\tPredicted Normal Generation Finish!" << std::endl;

		mesh_denoising->denoiseWithPredictedNormal(predicted_normal_exten, normal_iterations);

		std::cout << "\tSecond Denoising Finished! Start calculate Error..." << std::endl;
		double error = mesh_denoising->getError();
		std::cout << "Second Finish! Ea: " << error << std::endl;
		// double ea = mesh_denoising->getErrorMSAE();
		// std::cout << "Second Finish! MSE: " << error << " MSAE: " << ea << std::endl;
		// double dv = mesh_denoising->getErrorDv();
		// std::cout << "Second Finish! MSE: " << error << " MSAE: " << ea << " Dv: " << dv << std::endl;

		res_name = "./Results/Denoised_";
		tail = "_2.obj";
		res_name = res_name + m_model_name + tail;
		m_data_manager->ExportMeshToFile(res_name);
		std::cout << "Result Save Success!" << std::endl;

		predicted_normal_exten.clear();
	}

	/*----------------------------------------------------------*/
	// get visualization
	TriMesh denoised_mesh = m_data_manager->getDenoisedMesh();
	denoised_mesh.request_face_normals();
	denoised_mesh.update_face_normals();

	// calculate error map
	std::vector<glm::dvec3> res_color(m_num_faces);
	// std::vector<int> error_idx;

	int e_count = 0;
	for (TriMesh::FaceIter f_it = denoised_mesh.faces_begin(); f_it != denoised_mesh.faces_end(); f_it++)
	{
		TriMesh::Normal temp_normal = denoised_mesh.normal(*f_it);
		double angle = temp_normal | gt_normals[f_it->idx()];
		angle = std::min(1.0, std::max(angle, -1.0));
		// cross_value = cross_value * cross_value;
		angle = std::acos(angle) * 180.0 / M_PI;

		double r, g, b = 0.;

		if (angle >= 0 && angle < 20)
		{
			res_color[f_it->idx()][0] = 0. / 255.;
			res_color[f_it->idx()][1] = (0. + (angle / 20. * 255)) / 255.;
			res_color[f_it->idx()][2] = (255. - (angle / 20. * 255)) / 255.;
		}
		else if (angle >= 20 && angle <= 40)
		{
			res_color[f_it->idx()][0] = (0. + ((angle - 20) / 40. * 255)) / 255.;
			res_color[f_it->idx()][1] = (255. - ((angle - 20) / 20. * 255)) / 255.;
			res_color[f_it->idx()][2] = 0. / 255.;
		}
		else
		{
			res_color[f_it->idx()][0] = 255. / 255.;
			res_color[f_it->idx()][1] = 0. / 255.;
			res_color[f_it->idx()][2] = 0. / 255.;
		}
	}

	int m_i = 0;
	for (TriMesh::FaceIter f_it = denoised_mesh.faces_begin(); f_it != denoised_mesh.faces_end(); f_it++)
	{
		OpenMesh::Vec3d normal = denoised_mesh.normal(*f_it);

		int v_i = 0;
		for (TriMesh::FaceVertexIter fv_it = denoised_mesh.fv_iter(*f_it); fv_it.is_valid(); fv_it++)
		{
			OpenMesh::Vec3d position = denoised_mesh.point(*fv_it);

			// for position
			for (int i = 0; i < 3; i++)
			{
				m_denoised_mesh_visual_data[m_i * 3 * 9 + v_i * 9 + i] = (position[i] - m_center[i]) / m_max;
			}

			// for normal
			for (int i = 3; i < 6; i++)
			{
				m_denoised_mesh_visual_data[m_i * 3 * 9 + v_i * 9 + i] = normal[i - 3];
				m_denoised_mesh_visual_data[m_i * 3 * 9 + v_i * 9 + i + 3] = (normal[i - 3] + 1.) / 2.; // for normal map
				// m_denoised_mesh_visual_data[m_i * 3 * 9 + v_i * 9 + i + 3] = res_color[f_it->idx()][i - 3]; // for error map
			}
			v_i++;
		}
		m_i++;
	}
	m_is_reload = true;
	m_is_denoised = true;
	m_is_noise = false;
	m_is_gt = false;

	update();

	denoised_mesh.clean();

	delete mesh_denoising;
}