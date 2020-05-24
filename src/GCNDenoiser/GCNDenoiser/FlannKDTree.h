#pragma once

#include <flann/flann.hpp>

#include <glm.hpp>
#include <vector>

namespace shen
{
	namespace Geometry
	{
		class EasyFlann
		{
		public:
			EasyFlann()
			{

			}
			EasyFlann(const std::vector<glm::vec3> &_dataset)
			{
				flann::Matrix<float> dataset = flann::Matrix<float>(const_cast<float *>(&_dataset[0].x), _dataset.size(), 3);

				m_index = std::make_shared<flann::Index<flann::L2<float>>>(dataset, flann::KDTreeIndexParams(4));
				m_index->buildIndex();

			}
			~EasyFlann()
			{

			}

			void knnSearch(glm::vec3 &_query, int _k, std::vector<int> &_indices, std::vector<float> &_dists)
			{
				flann::Matrix<float> query(&_query.x, 1, 3);
				std::vector<std::vector<float>> dists;
				std::vector<std::vector<int>> indices;
				m_index->knnSearch(query, indices, dists, (size_t)_k, flann::SearchParams(128));

				_dists = dists[0];
				_indices = indices[0];
			}

			void knnSearch(glm::vec3 &_query, float &_dists, int &_indices)
			{
				flann::Matrix<float> query(&_query.x, 1, 3);
				std::vector<std::vector<float>> dists;
				std::vector<std::vector<int>> indices;
				m_index->knnSearch(query, indices, dists, (size_t)1, flann::SearchParams(128));

				_dists = dists[0][0];
				_indices = indices[0][0];
			}

			void knnSearch(glm::vec3 &_query, int &_indices)
			{
				flann::Matrix<float> query(&_query.x, 1, 3);
				std::vector<std::vector<float>> dists;
				std::vector<std::vector<int>> indices;
				m_index->knnSearch(query, indices, dists, (size_t)1, flann::SearchParams(128));

				_indices = indices[0][0];
			}

			void knnSearch(const glm::vec3 &_query, int &_indices)
			{
				flann::Matrix<float> query(const_cast<float*>(&_query.x), 1, 3);
				std::vector<std::vector<float>> dists;
				std::vector<std::vector<int>> indices;
				m_index->knnSearch(query, indices, dists, (size_t)1, flann::SearchParams(128));

				_indices = indices[0][0];
			}

			void radiusSearchNoLimit(glm::vec3 &_query, std::vector<float> &_dists, std::vector<int> &_indices, float r)
			{
				flann::Matrix<float> query = flann::Matrix<float>(&_query.x, 1, 3);
				std::vector<std::vector<float>> dists;
				std::vector<std::vector<int>> indices;
				m_index->radiusSearch(query, indices, dists, r, flann::SearchParams(1024));

				_dists = dists[0];
				_indices = indices[0];
			}

			void radiusSearch(glm::vec3 &_query, std::vector<float> &_dists, std::vector<int> &_indices, float r)
			{
				flann::Matrix<float> query = flann::Matrix<float>(&_query.x, 1, 3);
				std::vector<std::vector<float>> dists;
				std::vector<std::vector<int>> indices;
				m_index->radiusSearch(query, indices, dists, r, flann::SearchParams(128));

				_dists = dists[0];
				_indices = indices[0];
			}

			void radiusSearch(glm::vec3 &_query, std::vector<int> &_indices, float r)
			{
				flann::Matrix<float> query = flann::Matrix<float>(&_query.x, 1, 3);
				std::vector<std::vector<float>> dists;
				std::vector<std::vector<int>> indices;
				m_index->radiusSearch(query, indices, dists, r, flann::SearchParams(128));

				_indices = indices[0];
			}

		private:
			std::shared_ptr<flann::Index<flann::L2<float>>> m_index;
		};
	}
}