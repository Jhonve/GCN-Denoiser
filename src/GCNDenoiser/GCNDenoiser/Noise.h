#ifndef NOISE_H
#define NOISE_H

#include "Datamanager.h"
#include <vector>
#include <utility>
#include <algorithm>

class Noise
{
public:
	explicit Noise(DataManager *_data_manager, double noise_level = 0.2, int noise_type = 0);
	~Noise() {}

public:
	void addNoise();

private:
	enum NoiseType { kGaussian, kImpulsive };
	enum NoiseDirection { kNormal, kRandom };

	void initParameters(int noise_type_, double noise_level_);

	double generateRandomGaussian(double mean, double StandardDerivation);
	TriMesh::Normal generateRandomDirection();

	void randomGaussianNumbers(double mean, double StandardDerivation, int number, std::vector<double> &RandomNumbers);
	void randomImpulsiveNumbers(int min, int max, int number,
		double mean, double StandardDerivation,
		std::vector< std::pair<int, double> > &VertexListAndRandomNumbers);
	void randomDirections(int number, std::vector<TriMesh::Normal> &RandomDirections);

private:

	DataManager *data_manager_;

	double noise_level, impulsive_level;
	int noise_type_index, noise_direction_index;
	double m_noise_level;
	int m_noise_type;
};

#endif // NOISE_H