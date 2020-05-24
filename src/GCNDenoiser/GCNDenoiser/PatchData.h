#pragma once

#include "DataManager.h"

#include <vector>
#include <glm.hpp>

#include "FlannKDTree.h"

class PatchData
{
public:
	PatchData(TriMesh &mesh, std::vector<std::vector<TriMesh::FaceHandle>> &all_face_neighbor,
		std::vector<double> &face_area, std::vector<TriMesh::Point> &face_centroid,
		TriMesh::FaceIter &iter_face, shen::Geometry::EasyFlann &kd_tree, glm::dvec3& gt_normal, int num_ring = 2, int radius = 16);
	~PatchData();

	double m_radius_region;
	void patchAlignment(glm::dvec3& gt_normal);

public:
	int m_patch_num_faces = 0;	// faces in n-ring
	std::vector<glm::dvec4> m_patch_faces_centers; // [0] for the center point, length = patch_num_faces
	std::vector<glm::dvec3> m_patch_points_positions; // [0: 3] for the center point, length = patch_num_faces * 3
	std::vector<glm::dvec3> m_patch_faces_normals; // [0] for the center point, length = patch_num_faces

	int m_aligned_patch_num_faces = 0; // faces in the fixed region
	std::vector<glm::dvec3> m_aligned_patch_points_positions; // [0: 3] for the center point, length = patch_num_faces * 3
	std::vector<glm::dvec3> m_aligned_patch_faces_normals; // [0] for the center point, length = patch_num_faces

	int m_center_index = 0;

	unsigned char* m_patch_graph_matrix = NULL;
	unsigned char* m_temp_graph_matrix = NULL;
	double* m_temp_network_input = NULL;
	double* m_patch_node_features = NULL;
	glm::dvec3 m_center_normal;

	glm::dmat3x3 m_matrix;

private:
	int* m_matrix_indices;
};