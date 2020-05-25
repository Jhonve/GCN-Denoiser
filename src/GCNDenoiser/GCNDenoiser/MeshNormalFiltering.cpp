#include "MeshNormalFiltering.h"
#include <ctime>
#include <iostream>

MeshNormalFiltering::MeshNormalFiltering(DataManager *_data_manager)
	: MeshDenoisingBase(_data_manager)
{
	initParameters();
}

void MeshNormalFiltering::denoiseWithPredictedNormal(std::vector<TriMesh::Normal> guided_normals, int normal_iterations)
{
	// get data
	TriMesh mesh = data_manager_->getDenoisedMesh();

	if (mesh.n_vertices() == 0)
		return;

	// update face normal
	std::vector<TriMesh::Normal> filtered_normals;
	normal_iteration_number = normal_iterations;
	updateFilteredNormalsWithPredictedNormal(mesh, guided_normals);

	// update data
	data_manager_->setMesh(mesh);
	data_manager_->setDenoisedMesh(mesh);
}

void MeshNormalFiltering::initParameters()
{
	denoise_index = 0;

	face_neighbor_index = 0;
	include_central_face = true;
	multiple_radius = 2;
	multiple_sigma_s = 1;
	normal_iteration_number = 12;
	sigma_r = 0.3;
	vertex_iteration_number = 16;
}

void MeshNormalFiltering::getVertexBasedFaceNeighbor(TriMesh &mesh, TriMesh::FaceHandle fh, std::vector<TriMesh::FaceHandle> &face_neighbor)
{
	getFaceNeighbor(mesh, fh, kVertexBased, face_neighbor);
}

void MeshNormalFiltering::getRadiusBasedFaceNeighbor(TriMesh &mesh, TriMesh::FaceHandle fh, double radius, std::vector<TriMesh::FaceHandle> &face_neighbor)
{
	TriMesh::Point ci = mesh.calc_face_centroid(fh);
	std::vector<bool> flag((int)mesh.n_faces(), false);

	face_neighbor.clear();
	flag[fh.idx()] = true;
	std::queue<TriMesh::FaceHandle> queue_face_handle;
	queue_face_handle.push(fh);

	std::vector<TriMesh::FaceHandle> temp_face_neighbor;
	while (!queue_face_handle.empty())
	{
		TriMesh::FaceHandle temp_face_handle_queue = queue_face_handle.front();
		if (temp_face_handle_queue != fh)
			face_neighbor.push_back(temp_face_handle_queue);
		queue_face_handle.pop();
		getVertexBasedFaceNeighbor(mesh, temp_face_handle_queue, temp_face_neighbor);
		for (int i = 0; i < (int)temp_face_neighbor.size(); i++)
		{
			TriMesh::FaceHandle temp_face_handle = temp_face_neighbor[i];
			if (!flag[temp_face_handle.idx()])
			{
				TriMesh::Point cj = mesh.calc_face_centroid(temp_face_handle);
				double distance = (ci - cj).length();
				if (distance <= radius)
					queue_face_handle.push(temp_face_handle);
				flag[temp_face_handle.idx()] = true;
			}
		}
	}
}

void MeshNormalFiltering::getAllFaceNeighborGMNF(TriMesh &mesh, MeshDenoisingBase::FaceNeighborType face_neighbor_type, double radius, bool include_central_face,
	std::vector<std::vector<TriMesh::FaceHandle> > &all_face_neighbor)
{
	std::vector<TriMesh::FaceHandle> face_neighbor;
	for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
	{
		if (face_neighbor_type == kVertexBased)
			getVertexBasedFaceNeighbor(mesh, *f_it, face_neighbor);
		else if (face_neighbor_type == kRadiusBased)
			getRadiusBasedFaceNeighbor(mesh, *f_it, radius, face_neighbor);

		if (include_central_face)
			face_neighbor.push_back(*f_it);
		all_face_neighbor[f_it->idx()] = face_neighbor;
	}
}

double MeshNormalFiltering::GaussianWeight(double distance, double sigma)
{
	return std::exp(-0.5 * distance * distance / (sigma * sigma));
}

double MeshNormalFiltering::NormalDistance(const TriMesh::Normal &n1, const TriMesh::Normal &n2)
{
	return (n1 - n2).length();
}

void MeshNormalFiltering::getFaceNeighborInnerEdge(TriMesh &mesh, std::vector<TriMesh::FaceHandle> &face_neighbor, std::vector<TriMesh::EdgeHandle> &inner_edge)
{
	inner_edge.clear();
	std::vector<bool> edge_flag((int)mesh.n_edges(), false);
	std::vector<bool> face_flag((int)mesh.n_faces(), false);

	for (int i = 0; i < (int)face_neighbor.size(); i++)
		face_flag[face_neighbor[i].idx()] = true;

	for (int i = 0; i < (int)face_neighbor.size(); i++)
	{
		for (TriMesh::FaceEdgeIter fe_it = mesh.fe_iter(face_neighbor[i]); fe_it.is_valid(); fe_it++)
		{
			if ((!edge_flag[fe_it->idx()]) && (!mesh.is_boundary(*fe_it)))
			{
				edge_flag[fe_it->idx()] = true;
				TriMesh::HalfedgeHandle heh = mesh.halfedge_handle(*fe_it, 0);
				TriMesh::FaceHandle f = mesh.face_handle(heh);
				TriMesh::HalfedgeHandle heho = mesh.opposite_halfedge_handle(heh);
				TriMesh::FaceHandle fo = mesh.face_handle(heho);
				if (face_flag[f.idx()] && face_flag[fo.idx()])
					inner_edge.push_back(*fe_it);
			}
		}
	}
}

double MeshNormalFiltering::getRadius(double multiple, TriMesh &mesh)
{
	std::vector<TriMesh::Point> centroid;
	getFaceCentroid(mesh, centroid);

	double radius = 0.0;
	double num = 0.0;
	for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
	{
		TriMesh::Point fi = centroid[f_it->idx()];
		for (TriMesh::FaceFaceIter ff_it = mesh.ff_iter(*f_it); ff_it.is_valid(); ff_it++)
		{
			TriMesh::Point fj = centroid[ff_it->idx()];
			radius += (fj - fi).length();
			num++;
		}
	}
	return radius * multiple / num;
}

double MeshNormalFiltering::getSigmaS(double multiple, std::vector<TriMesh::Point> &centroid, TriMesh &mesh)
{
	double sigma_s = 0.0, num = 0.0;
	for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
	{
		TriMesh::Point fi = centroid[f_it->idx()];
		for (TriMesh::FaceFaceIter ff_it = mesh.ff_iter(*f_it); ff_it.is_valid(); ff_it++)
		{
			TriMesh::Point fj = centroid[ff_it->idx()];
			sigma_s += (fj - fi).length();
			num++;
		}
	}
	return sigma_s * multiple / num;
}

void MeshNormalFiltering::updateFilteredNormalsWithPredictedNormal(TriMesh &mesh, std::vector<TriMesh::Normal> guided_normals)
{
	std::vector<TriMesh::Normal> filtered_normals;
	filtered_normals.resize((int)mesh.n_faces());
	guided_normals.resize((int)mesh.n_faces());

	FaceNeighborType face_neighbor_type = (face_neighbor_index == 0) ? kRadiusBased : kVertexBased;

	double radius = 1.0;
	if (face_neighbor_type == kRadiusBased)
		radius = getRadius(multiple_radius, mesh);

	std::vector<std::vector<TriMesh::FaceHandle> > all_face_neighbor((int)mesh.n_faces());
	getAllFaceNeighborGMNF(mesh, face_neighbor_type, radius, include_central_face, all_face_neighbor);

	getFaceNormal(mesh, filtered_normals);

	std::vector<double> face_area((int)mesh.n_faces());
	std::vector<TriMesh::Point> face_centroid((int)mesh.n_faces());
	std::vector<TriMesh::Normal> previous_normals((int)mesh.n_faces());
	std::vector<std::pair<double, TriMesh::Normal> > range_and_mean_normal((int)mesh.n_faces());
	for (int iter = 0; iter < normal_iteration_number; iter++)
	{
		if (iter == 0)
		{
			std::cout << "\tNormal Filtering..." << std::endl;;
		}
		else
		{
			std::cout << "\tNormal Iteration" << iter << "/" << normal_iteration_number;
			printf("\33[2k\r");
		}
		getFaceCentroid(mesh, face_centroid);
		double sigma_s = getSigmaS(multiple_sigma_s, face_centroid, mesh);
		getFaceArea(mesh, face_area);
		getFaceNormal(mesh, previous_normals);

		// Filtered Face Normals
		for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
		{
			int index = f_it->idx();
			const std::vector<TriMesh::FaceHandle> face_neighbor = all_face_neighbor[index];
			TriMesh::Normal filtered_normal(0.0, 0.0, 0.0);
			for (int j = 0; j < (int)face_neighbor.size(); j++)
			{
				int current_face_index = face_neighbor[j].idx();

				double spatial_dis = (face_centroid[index] - face_centroid[current_face_index]).length();
				double spatial_weight = GaussianWeight(spatial_dis, sigma_s);
				double range_dis = (guided_normals[index] - guided_normals[current_face_index]).length();

				double range_weight = GaussianWeight(range_dis, sigma_r);

				if (iter == 0)
				{
					filtered_normal += guided_normals[current_face_index] * (face_area[current_face_index] * spatial_weight * range_weight);
				}
				else
				{
					filtered_normal += previous_normals[current_face_index] * (face_area[current_face_index] * spatial_weight * range_weight);
				}
			}
			if (face_neighbor.size())
			{
				filtered_normals[index] = filtered_normal.normalize();
			}
		}

		// immediate update vertex position
		updateVertexPosition(mesh, filtered_normals, vertex_iteration_number, false);
	}

	mesh.request_face_normals();
	mesh.update_face_normals();
}