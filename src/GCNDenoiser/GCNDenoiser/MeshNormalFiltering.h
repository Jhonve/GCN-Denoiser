#pragma once

#include "MeshDenoisingBase.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <queue>
#include <utility>

class MeshNormalFiltering : public MeshDenoisingBase
{
public:
	MeshNormalFiltering(DataManager *_data_manager);
	~MeshNormalFiltering() {}

	// shen
	void denoiseWithGuidance(std::vector<TriMesh::Normal> guided_normals);
	void finetuneWithGuidance(std::vector<TriMesh::Normal> guided_normals);

private:
	void initParameters();

	void getVertexBasedFaceNeighbor(TriMesh &mesh, TriMesh::FaceHandle fh, std::vector<TriMesh::FaceHandle> &face_neighbor);
	void getRadiusBasedFaceNeighbor(TriMesh &mesh, TriMesh::FaceHandle fh, double radius, std::vector<TriMesh::FaceHandle> &face_neighbor);
	void getAllFaceNeighborGMNF(TriMesh &mesh, FaceNeighborType face_neighbor_type, double radius, bool include_central_face, std::vector< std::vector<TriMesh::FaceHandle> > &all_face_neighbor);

	double GaussianWeight(double distance, double sigma);
	double NormalDistance(const TriMesh::Normal &n1, const TriMesh::Normal &n2);

	void getFaceNeighborInnerEdge(TriMesh &mesh, std::vector<TriMesh::FaceHandle> &face_neighbor, std::vector<TriMesh::EdgeHandle> &inner_edge);

	double getRadius(double multiple, TriMesh &mesh);
	double getSigmaS(double multiple, std::vector<TriMesh::Point> &centroid, TriMesh &mesh);

	// shen
	void updateFilteredNormalsWithGuidance(TriMesh &mesh, std::vector<TriMesh::Normal> guided_normals, std::vector<TriMesh::Normal> &filtered_normals);
	void updateFilteredNormalsUseGuidance(TriMesh &mesh, std::vector<TriMesh::Normal> guided_normals, std::vector<TriMesh::Normal> &filtered_normals);

private:
	int denoise_index;
	int face_neighbor_index;
	bool include_central_face;
	double multiple_radius;
	double	multiple_sigma_s;
	int	normal_iteration_number;
	double	sigma_r;
	int	vertex_iteration_number;
	double smoothness;
};