#include "Datamanager.h"

DataManager::DataManager()
{

}

bool DataManager::ImportMeshFromFile(std::string filename, bool isOriginal)
{
	if (!OpenMesh::IO::read_mesh(mesh_, filename))
		return false;
	else
	{
		if (isOriginal) original_mesh_ = mesh_;
		else
		{
			noisy_mesh_ = mesh_;
			denoised_mesh_ = mesh_;
		}
		return true;
	}
}

bool DataManager::ExportMeshToFile(std::string filename)
{
	return OpenMesh::IO::write_mesh(mesh_, filename);
}
