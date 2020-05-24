import os
import random

import numpy as np
import scipy.io as sio
import h5py

import torch
import torch.utils.data as Tdata

from parsers import getParser

k_data_path = "./testsamples/"
k_h5_path = "./testsamples/"

class MatrixDataset(Tdata.Dataset):
    def __init__(self, parser, data_path, num_neighbors, is_train):
        super(MatrixDataset).__init__()
        self.batch_size = parser.batch_size
        self.num_workers = parser.num_workers
        self.data_path = data_path
        self.num_neighbors = num_neighbors
        self.SIZE = len(self.data_path)

        self.is_train = is_train

    def __len__(self):
        return self.SIZE

    def loadMAT(self, data_path):
        source_data = sio.loadmat(data_path)
        input_matrix = source_data["MAT"]
        input_matrix = np.array(input_matrix)
        
        input_features = source_data["FEA"]
        input_features = np.array(input_features)
        input_features = input_features.T

        num_faces = input_matrix.shape[0]
        if(num_faces >= self.num_neighbors):
            input_matrix = input_matrix[0:self.num_neighbors, 0:self.num_neighbors]
            input_features = input_features[0:self.num_neighbors]
        else:
            # Matrix Padding
            input_matrix = np.pad(input_matrix, ((0, self.num_neighbors - num_faces), (0, self.num_neighbors - num_faces)), \
                'constant', constant_values = (0, 0))
            input_features = np.pad(input_features, ((0, self.num_neighbors - num_faces), (0, 0)), \
                'constant', constant_values = (0, 0))

        # input_features = input_features[:, 0:7]

        input_indices = []
        for i in range(self.num_neighbors):
            temp_idx = np.array((input_matrix[i] == 1).nonzero()).reshape(-1)
            temp_idx = list(temp_idx)
            if(len(temp_idx) == 0):
                temp_idx = [self.num_neighbors - 1, self.num_neighbors - 1, self.num_neighbors - 1]
            elif(len(temp_idx) == 1):
                temp_idx.append(temp_idx[0])
                temp_idx.append(temp_idx[0])
            elif(len(temp_idx) == 2):
                temp_idx.append(temp_idx[1])

            input_indices.append(temp_idx)

        input_indices = np.array(input_indices)

        gt_norm = source_data["GT"]
        gt_norm = np.array(gt_norm).reshape(-1).astype(np.float32)

        center_norm = source_data["NOR"]
        center_norm = np.array(center_norm).reshape(-1).astype(np.float32)
        
        gt_res = ((np.dot(gt_norm, center_norm) * gt_norm) - center_norm + 1.) / 2.

        # gt_norm = (gt_norm + 1.) / 2.

        inputs = np.c_[input_features, input_indices]

        return inputs, gt_res, gt_norm, center_norm

    def __getitem__(self, index):
        inputs, gt, gt_norm, center_norm = self.loadMAT(self.data_path[index])
        return inputs, gt, gt_norm, center_norm

    def getDataloader(self):
        if(self.is_train):
            return Tdata.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)
        else:
            return Tdata.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=True)

def preDataPath(folder_list):
    all_file_list = []
    for i in range(len(folder_list)):
        file_list = os.listdir(k_data_path + folder_list[i])
        for j in range(len(file_list)):
            if(file_list[j][0] == '9'):
                continue
            file_list[j] = k_data_path + folder_list[i] + '/' + file_list[j]
            all_file_list.append(file_list[j])

    return all_file_list

def preSplitDataPath(folder_list):
    file_list = []
    for i in range(len(folder_list)):
        temp_file_list = os.listdir(k_data_path + folder_list[i])
        for j in range(len(temp_file_list)):
            if(temp_file_list[j][0] == '1' or temp_file_list[j][0] == '2'):
                temp_file_list[j] = k_data_path + folder_list[i] + '/' + temp_file_list[j]
                file_list.append(temp_file_list[j])

    return file_list

def saveH5(files_path, target_name="dataPath.h5"):
    files_path = np.array(files_path)
    
    with h5py.File(k_h5_path + target_name, 'w') as data_file:
        data_type = h5py.special_dtype(vlen=str)
        data = data_file.create_dataset("data_path", files_path.shape, dtype=data_type)
        data[:] = files_path
        data_file.close()

    print("Save path done!")

def selfTest():
    # initialize data path to dataPath.h5
    folder_list = os.listdir(k_data_path)
    print("Number of models: ", len(folder_list))

    files_path = preDataPath(folder_list)
    print("Number of data: ", len(files_path))
    saveH5(files_path, "TestDataPath.h5")
    
    '''
    # test dataset loader
    opt = getParser()

    data_path_file = opt.data_path_file
    data_path = h5py.File(data_path_file, 'r')

    data_path = np.array(data_path["data_path"])[1300 * 1024 : -1]

    dataset = MatrixDataset(opt, data_path, opt.num_neighbors, False)
    data_loader = dataset.getDataloader()

    for i, data in enumerate(data_loader, 0):
        inputs, gt, gt_norm, center_norm = data
        print(inputs.shape, i)
    '''

if __name__ == '__main__':
    selfTest()