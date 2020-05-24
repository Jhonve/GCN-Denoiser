#include "PatchData.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <set>

PatchData::PatchData(TriMesh &mesh, std::vector<std::vector<TriMesh::FaceHandle>> &all_face_neighbor,
	std::vector<double> &face_area, std::vector<TriMesh::Point> &face_centroid,
	TriMesh::FaceIter &iter_face, shen::Geometry::EasyFlann &kd_tree, glm::dvec3& gt_normal, int num_ring, int radius)
{
	std::vector<std::vector<TriMesh::FaceHandle>> face_neighbor_ring(num_ring + 1);
	std::vector<std::vector<bool>> flag(num_ring + 1, std::vector<bool>(mesh.n_faces(), false));

	face_neighbor_ring[0].push_back(*iter_face);
	if (num_ring >= 1)
	{
		face_neighbor_ring[1] = all_face_neighbor[iter_face->idx()];
	}
	flag[0][iter_face->idx()] = true;

	if (num_ring >= 1)
	{
		for (int i = 0; i < (int)face_neighbor_ring[1].size(); i++)
		{
			flag[1][face_neighbor_ring[1][i].idx()] = true;
		}
	}

	for (int ring = 1; ring < num_ring; ring++)
	{
		for (int i = 0; i < (int)face_neighbor_ring[ring].size(); i++)
		{
			std::vector<TriMesh::FaceHandle> temp_neighbor = all_face_neighbor[face_neighbor_ring[ring][i].idx()];
			for (int t = 0; t < (int)temp_neighbor.size(); t++)
			{
				if ((!flag[ring - 1][temp_neighbor[t].idx()]) && (!flag[ring][temp_neighbor[t].idx()]) && (!flag[ring + 1][temp_neighbor[t].idx()]))
				{
					face_neighbor_ring[ring + 1].push_back(temp_neighbor[t]);
					flag[ring + 1][temp_neighbor[t].idx()] = true;
				}
			}
		}
	}

	// for center point position
	TriMesh::Point centroidc = face_centroid[iter_face->idx()];
	double areac = face_area[iter_face->idx()];
	glm::dvec4 center_face_centorid(areac, centroidc[0], centroidc[1], centroidc[2]);
	m_patch_faces_centers.push_back(center_face_centorid);

	glm::dvec3 normalc(mesh.normal(*iter_face)[0], mesh.normal(*iter_face)[1], mesh.normal(*iter_face)[2]);
	m_patch_faces_normals.push_back(normalc);
	for (TriMesh::FaceVertexIter fv_it = mesh.fv_iter(*iter_face); fv_it.is_valid(); fv_it++)
	{
		glm::dvec3 positionc(mesh.point(*fv_it)[0], mesh.point(*fv_it)[1], mesh.point(*fv_it)[2]);
		m_patch_points_positions.push_back(positionc);
	}

	int num_faces = 1;
	for (int ring = 1; ring <= num_ring; ring++)
	{
		for (int i = 0; i < (int)face_neighbor_ring[ring].size(); i++)
		{
			TriMesh::Point centroid = face_centroid[face_neighbor_ring[ring][i].idx()];

			glm::dvec4 neighbor_faces_centorid(areac, centroid[0], centroid[1], centroid[2]);

			// for kinect fussion calculate the mean area of 2-ring neighbors
			// glm::dvec4 neighbor_faces_centorid(face_area[face_neighbor_ring[ring][i].idx()], centroid[0], centroid[1], centroid[2]);

			m_patch_faces_centers.push_back(neighbor_faces_centorid);

			glm::dvec3 normal(mesh.normal(face_neighbor_ring[ring][i])[0], mesh.normal(face_neighbor_ring[ring][i])[1], mesh.normal(face_neighbor_ring[ring][i])[2]);
			m_patch_faces_normals.push_back(normal);
			for (TriMesh::FaceVertexIter fv_it = mesh.fv_iter(face_neighbor_ring[ring][i]); fv_it.is_valid(); fv_it++)
			{
				glm::dvec3 position(mesh.point(*fv_it)[0], mesh.point(*fv_it)[1], mesh.point(*fv_it)[2]);
				m_patch_points_positions.push_back(position);
			}
			num_faces++;
		}
	}
	m_patch_num_faces = num_faces;

	// no face-face neighbors
	if (m_patch_num_faces == 1)
	{
		return;
	}

	m_radius_region = sqrt(areac * double(radius));

	//// for kinect fussion calculate the mean area of 2-ring neighbors
	//double ring_2_mean_area = areac;
	//double ring_2_sum_area = 0;
	//for (int i_ring_area = 0; i_ring_area < m_patch_num_faces; i_ring_area++)
	//{
	//	ring_2_sum_area += m_patch_faces_centers[i_ring_area][0];
	//}
	//ring_2_mean_area = ring_2_sum_area / double(m_patch_num_faces);
	//m_radius_region = sqrt(ring_2_mean_area * double(radius));

	glm::vec3 query_point(m_patch_faces_centers[0][1], m_patch_faces_centers[0][2], m_patch_faces_centers[0][3]);
	std::vector<float> points_distances;
	std::vector<int> points_indices;
	kd_tree.radiusSearchNoLimit(query_point, points_distances, points_indices, m_radius_region * m_radius_region);

	int num_region_points = points_indices.size();
	std::vector<TriMesh::VertexIter> vertex_iters;

	for (int i = 0; i < num_region_points; i++)
	{
		TriMesh::VertexIter temp_iter = mesh.vertices_begin();
		for (int i_iter = 0; i_iter < points_indices[i]; i_iter++)
		{
			temp_iter++;
		}
		vertex_iters.push_back(temp_iter);
	}
	points_indices.clear();
	points_distances.clear();

	std::set<int> face_indices_set;		// faces in the fixed region
	std::vector<int> face_indices;
	std::vector<TriMesh::VertexFaceIter> patch_faces_iter;
	for (int i = 0; i < num_region_points; i++)
	{
		for (TriMesh::VertexFaceIter vf_it = mesh.vf_iter(vertex_iters[i]); vf_it.is_valid(); vf_it++)
		{
			if (face_indices_set.find(vf_it->idx()) == face_indices_set.end())
			{
				face_indices_set.insert(vf_it->idx());
				face_indices.push_back(vf_it->idx());
				patch_faces_iter.push_back(vf_it);
				if (vf_it->idx() == iter_face->idx())
				{
					m_center_index = face_indices.size() - 1;
				}
				glm::dvec3 normal(mesh.normal(*vf_it)[0], mesh.normal(*vf_it)[1], mesh.normal(*vf_it)[2]);
				m_aligned_patch_faces_normals.push_back(normal);
				for (TriMesh::FaceVertexIter fv_it = mesh.fv_iter(vf_it); fv_it.is_valid(); fv_it++)
				{
					glm::dvec3 position(mesh.point(*fv_it)[0], mesh.point(*fv_it)[1], mesh.point(*fv_it)[2]);
					m_aligned_patch_points_positions.push_back(position);
				}
				m_aligned_patch_num_faces++;
			}
		}
	}

	/*std::cout << m_aligned_patch_num_faces << std::endl;*/
	if (m_aligned_patch_num_faces <= 1)
	{
		// std::cout << m_aligned_patch_num_faces << std::endl;
		return;
	}

	// initialize the patch graph
	m_temp_graph_matrix = new unsigned char[m_aligned_patch_num_faces * m_aligned_patch_num_faces];
	m_patch_graph_matrix = new unsigned char[m_aligned_patch_num_faces * m_aligned_patch_num_faces];
	m_temp_network_input = new double[m_aligned_patch_num_faces * (17 + 3)];
	m_patch_node_features = new double[m_aligned_patch_num_faces * 17];	// 17 kinds of features, face center(3),  face normal (3), area(1), num of neighbors(1), points position (9),
	m_matrix_indices = new int[m_aligned_patch_num_faces];
	for (int i = 0; i < m_aligned_patch_num_faces * m_aligned_patch_num_faces; i++)
	{
		m_temp_graph_matrix[i] = 0;
		m_patch_graph_matrix[i] = 0;
	}

	for (int i_f = 0; i_f < m_aligned_patch_num_faces; i_f++)
	{
		int neighbor_count = 0;
		for (TriMesh::FaceFaceIter ff_it = mesh.ff_iter(patch_faces_iter[i_f]); ff_it.is_valid(); ff_it++)
		{
			int temp_neighbor_index = ff_it->idx();
			for (int j_f = 0; j_f < m_aligned_patch_num_faces; j_f++)
			{
				if (temp_neighbor_index == face_indices[j_f])
				{
					m_temp_graph_matrix[i_f * m_aligned_patch_num_faces + j_f] = 1;
					if (neighbor_count < 3)
					{
						if (j_f < 64)
						{
							m_temp_network_input[i_f * 20 + 17 + neighbor_count] = (double)j_f;
						}
						else
						{
							/*m_temp_network_input[i_f * 20 + 17 + neighbor_count] = 63.;*/
							neighbor_count--;
						}
					}
					neighbor_count++;
					break;
				}
			}
		}

		// padding of neighbors
		if (neighbor_count == 2)
		{
			m_temp_network_input[i_f * 20 + 17 + 2] = m_temp_network_input[i_f * 20 + 17 + 1];
		}
		else if (neighbor_count == 1)
		{
			m_temp_network_input[i_f * 20 + 17 + 1] = m_temp_network_input[i_f * 20 + 17 + 0];
			m_temp_network_input[i_f * 20 + 17 + 2] = m_temp_network_input[i_f * 20 + 17 + 1];
		}
		else if (neighbor_count == 0)
		{
			m_temp_network_input[i_f * 20 + 17 + 0] = (double)i_f;
			m_temp_network_input[i_f * 20 + 17 + 1] = (double)i_f;
			m_temp_network_input[i_f * 20 + 17 + 2] = (double)i_f;
		}

		// area
		m_patch_node_features[i_f * 17 + 6] = face_area[face_indices[i_f]];

		// num of neighbors
		int num_neighbors = all_face_neighbor[face_indices[i_f]].size();
		m_patch_node_features[i_f * 17 + 7] = ((((double)num_neighbors - 12.) / 6.) + 1.) / 2.;
		m_temp_network_input[i_f * 20 + 7] = m_patch_node_features[i_f * 17 + 7];
	}

	vertex_iters.clear();
	face_indices_set.clear();
	patchAlignment(gt_normal);
}

PatchData::~PatchData()
{
	m_patch_faces_centers.clear();
	m_patch_points_positions.clear();
	m_patch_faces_normals.clear();
	m_aligned_patch_points_positions.clear();
	m_aligned_patch_faces_normals.clear();

	if (m_temp_graph_matrix != NULL)
	{
		delete m_temp_graph_matrix;
		delete m_patch_graph_matrix;
		delete  m_patch_node_features;
		delete m_temp_network_input;
		delete m_matrix_indices;
	}
}

void PatchData::patchAlignment(glm::dvec3& gt_normal)
{
	glm::dvec4 area_center = m_patch_faces_centers[0];
	double area_max = 0;
	for (int i = 0; i < m_patch_num_faces; i++)
	{
		if (area_max < m_patch_faces_centers[i][0])
		{
			area_max = m_patch_faces_centers[i][0];
		}
	}

	// std::cout << "	\tThe max area of the patch is: " << area_max << std::endl;

	Eigen::Matrix3d tensor_feature_mat;
	Eigen::Matrix3d temp_mat;
	tensor_feature_mat = Eigen::Matrix3d::Zero();

	for (int i = 1; i < m_patch_num_faces; i++)
	{
		temp_mat = Eigen::Matrix3d::Zero();
		glm::dvec3 temp_center(m_patch_faces_centers[i][1] - m_patch_faces_centers[0][1],
			m_patch_faces_centers[i][2] - m_patch_faces_centers[0][2],
			m_patch_faces_centers[i][3] - m_patch_faces_centers[0][3]);

		double temp_area = m_patch_faces_centers[i][0] / area_max;
		double  temp_center_norm_para = exp(-3 * glm::length(temp_center));

		glm::dvec3 temp_weight = glm::cross(glm::cross(temp_center, m_patch_faces_normals[i]), temp_center);
		temp_weight = glm::normalize(temp_weight);
		glm::dvec3 temp_normal = 2. * glm::dot(m_patch_faces_normals[i], temp_weight) * temp_weight - m_patch_faces_normals[i];

		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				temp_mat(r, c) = temp_normal[r] * temp_normal[c];
			}
		}
		temp_mat *= temp_area * temp_center_norm_para;
		// temp_mat *= temp_center_norm_para;
		tensor_feature_mat += temp_mat;
	}

	Eigen::EigenSolver<Eigen::Matrix3d> eigen_solver(tensor_feature_mat);

	double max_eigen_value = eigen_solver.eigenvalues()[0].real();
	double min_eigen_value = eigen_solver.eigenvalues()[0].real();
	int max_ev_idx = 0;
	int min_ev_idx = 0;
	for (int i = 0; i < 3; i++)
	{
		if (max_eigen_value < eigen_solver.eigenvalues()[i].real())
		{
			max_eigen_value = eigen_solver.eigenvalues()[i].real();
			max_ev_idx = i;
		}
		if (min_eigen_value > eigen_solver.eigenvalues()[i].real())
		{
			min_eigen_value = eigen_solver.eigenvalues()[i].real();
			min_ev_idx = i;
		}
	}
	int middle_ev_idx = 3 - max_ev_idx - min_ev_idx;

	// only one face
	if (middle_ev_idx < 0 || middle_ev_idx > 2)
	{
		min_ev_idx = 0;
		middle_ev_idx = 1;
		max_ev_idx = 2;
		/*std::cout << "!!!____Sort ERROR____!!!" << std::endl;
		exit(0);*/
	}

	glm::dvec3 sorted_eigen_value(eigen_solver.eigenvalues()[max_ev_idx].real(),
		eigen_solver.eigenvalues()[middle_ev_idx].real(),
		eigen_solver.eigenvalues()[min_ev_idx].real());

	glm::dvec3 max_eigen_vec(eigen_solver.eigenvectors().col(max_ev_idx).row(0).value().real(),
		eigen_solver.eigenvectors().col(max_ev_idx).row(1).value().real(),
		eigen_solver.eigenvectors().col(max_ev_idx).row(2).value().real());
	glm::dvec3 middel_eigen_vec(eigen_solver.eigenvectors().col(middle_ev_idx).row(0).value().real(),
		eigen_solver.eigenvectors().col(middle_ev_idx).row(1).value().real(),
		eigen_solver.eigenvectors().col(middle_ev_idx).row(2).value().real());
	glm::dvec3 min_eigen_vec(eigen_solver.eigenvectors().col(min_ev_idx).row(0).value().real(),
		eigen_solver.eigenvectors().col(min_ev_idx).row(1).value().real(),
		eigen_solver.eigenvectors().col(min_ev_idx).row(2).value().real());

	glm::dmat3x3 sorted_eigen_vec(max_eigen_vec, middel_eigen_vec, min_eigen_vec);

	if (sorted_eigen_vec[0][0] * m_patch_faces_normals[0][0]
		+ sorted_eigen_vec[0][1] * m_patch_faces_normals[0][1]
		+ sorted_eigen_vec[0][2] * m_patch_faces_normals[0][2] < 0.)
	{
		sorted_eigen_vec[0] *= -1.;
		sorted_eigen_vec[1] *= -1.;
		sorted_eigen_vec[2] *= -1.;
	}

	m_matrix = sorted_eigen_vec;

	sorted_eigen_value = sorted_eigen_value / sqrt(pow(sorted_eigen_value[0], 2) + pow(sorted_eigen_value[1], 2) + pow(sorted_eigen_value[2], 2));

	// std::cout << sorted_eigen_value[0] << " " << sorted_eigen_value[1] << " " << sorted_eigen_value[2] << std::endl;

	// for normal base change
	glm::dmat3x3 inversed_sorted_vec = glm::inverse(sorted_eigen_vec);

	/* std::cout << inversed_sorted_vec[0][0] << " " << inversed_sorted_vec[1][0] << " " << inversed_sorted_vec[2][0] << std::endl <<
		 inversed_sorted_vec[0][1] << " " << inversed_sorted_vec[1][1] << " " << inversed_sorted_vec[2][1] << std::endl <<
		 inversed_sorted_vec[0][2] << " " << inversed_sorted_vec[1][2] << " " << inversed_sorted_vec[2][2] << std::endl;*/

	for (int i = 0; i < m_aligned_patch_num_faces; i++)
	{
		m_aligned_patch_faces_normals[i] = inversed_sorted_vec * m_aligned_patch_faces_normals[i];
	}

	gt_normal = inversed_sorted_vec * gt_normal;

	// for position base change
	glm::dmat4x4 trans_mat(0);
	for (int i = 0; i < 4; i++)
	{
		trans_mat[i][i] = 1;
		if (i < 3)
		{
			trans_mat[3][i] = -1 * m_patch_faces_centers[0][i + 1];
		}
	}

	glm::dmat4x4 rot_mat(0);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (i < 3 && j < 3)
			{
				rot_mat[i][j] = inversed_sorted_vec[i][j];
			}
		}
	}
	rot_mat[3][3] = 1.;

	double determinant = glm::determinant(inversed_sorted_vec);

	double coord_max = 0.;
	std::vector<glm::dvec3> temp_positions;
	for (int i_face = 0; i_face < m_aligned_patch_num_faces; i_face++)
	{
		for (int i_vertex = 0; i_vertex < 3; i_vertex++)
		{
			glm::dvec4 temp_position;
			if (determinant < 0)
			{
				temp_position = glm::dvec4(m_aligned_patch_points_positions[i_face * 3 + 2 - i_vertex][0],
					m_aligned_patch_points_positions[i_face * 3 + 2 - i_vertex][1],
					m_aligned_patch_points_positions[i_face * 3 + 2 - i_vertex][2], 1.);
			}
			else if (determinant > 0)
			{
				temp_position = glm::dvec4(m_aligned_patch_points_positions[i_face * 3 + i_vertex][0],
					m_aligned_patch_points_positions[i_face * 3 + i_vertex][1],
					m_aligned_patch_points_positions[i_face * 3 + i_vertex][2], 1.);
			}
			else
			{
				/* std::cout << "!!!____Transform matrix Error, Not positive definite matrix____!!!" << std::endl;
				 exit(0);*/
				temp_position = glm::dvec4(m_aligned_patch_points_positions[i_face * 3 + i_vertex][0],
					m_aligned_patch_points_positions[i_face * 3 + i_vertex][1],
					m_aligned_patch_points_positions[i_face * 3 + i_vertex][2], 1.);
			}

			temp_position = trans_mat * temp_position;
			temp_position = rot_mat * temp_position;

			for (int i_v = 0; i_v < 3; i_v++)
			{
				if (coord_max < abs(temp_position[i_v]))
				{
					coord_max = abs(temp_position[i_v]);
				}
			}

			temp_positions.push_back(glm::dvec3(temp_position[0],
				temp_position[1],
				temp_position[2]));
		}
	}

	for (int i_v = 0; i_v < m_aligned_patch_num_faces * 3; i_v++)
	{
		m_aligned_patch_points_positions[i_v] = temp_positions[i_v] / m_radius_region;
	}

	for (int i_f = 0; i_f < m_aligned_patch_num_faces; i_f++)
	{
		m_patch_node_features[i_f * 17 + 6] /= m_radius_region * m_radius_region;
		m_temp_network_input[i_f * 20 + 6] = m_patch_node_features[i_f * 17 + 6];
		// face center and normal
		for (int i_v = 0; i_v < 3; i_v++)
		{
			m_patch_node_features[i_f * 17 + i_v] = ((m_aligned_patch_points_positions[i_f * 3][i_v]
				+ m_aligned_patch_points_positions[i_f * 3 + 1][i_v]
				+ m_aligned_patch_points_positions[i_f * 3 + 2][i_v]) / 3 + 1) / 2.;
			m_temp_network_input[i_f * 20 + i_v] = m_patch_node_features[i_f * 17 + i_v];

			m_patch_node_features[i_f * 17 + i_v + 3] = (m_aligned_patch_faces_normals[i_f][i_v] + 1) / 2.;
			m_temp_network_input[i_f * 20 + i_v + 3] = m_patch_node_features[i_f * 17 + i_v + 3];
		}
		// face vertices
		for (int i_p = 0; i_p < 3; i_p++)
		{
			m_patch_node_features[i_f * 17 + i_p * 3 + 8] = (m_aligned_patch_points_positions[i_f * 3][0] + 1) / 2.;
			m_patch_node_features[i_f * 17 + i_p * 3 + 8 + 1] = (m_aligned_patch_points_positions[i_f * 3][1] + 1) / 2.;
			m_patch_node_features[i_f * 17 + i_p * 3 + 8 + 2] = (m_aligned_patch_points_positions[i_f * 3][2] + 1) / 2.;

			m_temp_network_input[i_f * 20 + i_p * 3 + 8] = m_patch_node_features[i_f * 17 + i_p * 3 + 8];
			m_temp_network_input[i_f * 20 + i_p * 3 + 8 + 1] = m_patch_node_features[i_f * 17 + i_p * 3 + 8 + 1];
			m_temp_network_input[i_f * 20 + i_p * 3 + 8 + 2] = m_patch_node_features[i_f * 17 + i_p * 3 + 8 + 2];
		}
	}

	m_center_normal = glm::dvec3(0., 0., 0.);
	for (int i_n = 0; i_n < 3; i_n++)
	{
		m_center_normal[i_n] = m_aligned_patch_faces_normals[m_center_index][i_n];
	}
}