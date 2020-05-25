#pragma once
#include "QtWidgets/qopenglwidget.h"
#include "qopenglfunctions_4_5_core.h"
#include "qopenglfunctions.h"
#include "qopenglbuffer.h"
#include "qopenglvertexarrayobject.h"

#include "qdebug.h"
#include "qmatrix4x4.h"
#include "qevent.h"

#include "Shader.h"

#include "PatchData.h"
#include "DataManager.h"
#include "MeshDenoisingBase.h"

class MeshViewer : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT

public:
	MeshViewer(QWidget *parent = 0);
	~MeshViewer();

	static bool isTransparent() { return m_transparent; };
	static void setTransparent(bool t) { m_transparent = t; };

	QSize minimumSizeHint() const override;
	QSize sizeHint() const override;

public slots:
	void setXRotation(int angle);
	void setYRotation(int angle);
	void setZRotation(int angle);
	void cleanup();

signals:
	void xRotationChanged(int angle);
	void yRotationChanged(int angle);
	void zRotationChanged(int angle);

protected:
	void initializeGL() override;
	void resizeGL(int width, int height) override;
	void paintGL() override;
	void mousePressEvent(QMouseEvent *event) override;
	void mouseMoveEvent(QMouseEvent *event) override;
	void wheelEvent(QWheelEvent *event) override;
	void keyPressEvent(QKeyEvent *e);

private:
	Shader *m_shader;
	QOpenGLFunctions_4_5_Core *m_core;

	int m_xRot;
	int m_yRot;
	int m_zRot;
	QPoint m_lastPos;	// mouse's last position
	int m_projMatrixLoc;	// vertices' projection matrix location
	int m_modelMatrixLoc;	// vertices' world matrix location
	int m_viewMatrixLoc;	// vertices' camera matrix location
	int m_aLightPosLoc;    // frag's A light position location
	int m_bLightPosLoc;		// frag's B light position location
	int  m_viewPosLoc;	// frag's view position location
	QMatrix4x4 m_proj;	// project matrix
	QMatrix4x4 m_camera;	// camera transform matrix
	QMatrix4x4 m_world;		// world transform matrix
	QVector3D m_a_light_pos;
	QVector3D m_b_light_pos;
	QVector3D m_view_pos;
	static bool m_transparent;

private:
	std::string m_current_file_name;
	std::string m_model_name;
	double m_current_noise_level;
	int m_current_noise_type;

	bool m_is_reload = false;

	bool m_is_noise = true;
	bool m_is_gt = false;
	bool m_is_denoised = false;

	bool m_is_have_noise = true;
	bool m_is_have_gt = true;

	OpenMesh::Vec3d m_center;
	double m_max = 0.;

	DataManager* m_data_manager = NULL;
	TriMesh m_noised_tri_mesh;
	TriMesh m_gt_tri_mesh;
	int m_num_faces = 0;
	GLfloat* m_noised_mesh_visual_data = NULL;
	GLfloat* m_gt_mesh_visual_data = NULL;
	GLfloat* m_denoised_mesh_visual_data = NULL;

	std::vector<std::vector<TriMesh::FaceHandle>> m_all_face_neighbor;
	std::vector<double> m_face_area;
	std::vector<TriMesh::Point> m_face_centroid;

	std::vector<glm::vec3> m_vertices_kd_tree;
	shen::Geometry::EasyFlann m_flann_kd_tree;

	void meshInitializedGet(const char* file_name, bool is_original);
	void meshVisualization(GLfloat* m_mesh_visual_data, TriMesh& tri_mesh);

	// same with mesh denoising base
	enum FaceNeighborType { kVertexBased, kEdgeBased, kRadiusBased };

	void getFaceNeighbor(TriMesh &mesh, TriMesh::FaceHandle fh, FaceNeighborType face_neighbor_type, std::vector<TriMesh::FaceHandle> &face_neighbor);
	void getAllFaceNeighbor(TriMesh &mesh, std::vector< std::vector<TriMesh::FaceHandle> > &all_face_neighbor, FaceNeighborType face_neighbor_type = kVertexBased, bool include_central_face = false);
	void getFaceArea(TriMesh &mesh, std::vector<double> &area);
	void getFaceCentroid(TriMesh &mesh, std::vector<TriMesh::Point> &centroid);

public slots:
	void slotLoadNoise();
	void slotLoadGT();
	void slotDelete();
	void slotGenNoise(double noise_level, QString noise_type);
	void slotDenoise(int gcns, int normal_iterations);
};