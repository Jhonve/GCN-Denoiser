import os
import math

import h5py
import numpy as np
import scipy.io as sio

import torch
import torch.nn.functional as F

from datautils import MatrixDataset
from GCNModel import DGCNN
from parsers import getParser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

k_opt = getParser()

def test():
    weight_alpha = 0.
    weight_beta = 1.

    print("Loading pre-trained model: ", k_opt.current_model)

    dgcnn = DGCNN(6, 17, 1024, 0.5)
    # dgcnn = torch.nn.DataParallel(dgcnn)
    dgcnn.load_state_dict(torch.load(k_opt.current_model))
    print("Load ", k_opt.current_model, " Success!")
    dgcnn.cuda()
    dgcnn.eval()

    cos_target = torch.tensor(np.ones((k_opt.batch_size, 1)))
    cos_target = cos_target.type(torch.FloatTensor).cuda()

    # all_file_list = getTestList(True)

    data_path_file = k_opt.data_path_file
    data_path = h5py.File(data_path_file, 'r')
    data_path = np.array(data_path["data_path"])
    print("Respilt Data Success!")

    test_dataset = MatrixDataset(k_opt, data_path, k_opt.num_neighbors, is_train=False)
    test_data_loader = test_dataset.getDataloader()
    
    val_cos_loss = []
    val_value_loss = []
    val_loss = []
    for i_test, data in enumerate(test_data_loader, 0):
        inputs, gt_res, gt_norm, center_norm = data
        inputs = inputs.type(torch.FloatTensor)
        inputs = inputs.permute(0, 2, 1)
        gt_res = gt_res.type(torch.FloatTensor)
        gt_norm = gt_norm.type(torch.FloatTensor)
        center_norm = center_norm.type(torch.FloatTensor)

        inputs = inputs.cuda()
        gt_res = gt_res.cuda()
        gt_norm = gt_norm.cuda()
        center_norm = center_norm.cuda()

        output = dgcnn(inputs)

        cos_loss = F.cosine_embedding_loss(output, gt_norm, cos_target)
        value_loss = F.mse_loss(output, gt_norm)

        loss = weight_alpha * cos_loss + weight_beta * value_loss

        val_loss.append(loss.data.item())
        val_cos_loss.append(cos_loss.data.item())
        val_value_loss.append(value_loss.data.item())

        print("Val Batch: %d/%d, || cos loss: %.7f, || value loss: %.7f" % \
                (i_test + 1, k_opt.num_val_batch, cos_loss.data.item(), value_loss.data.item()))

if __name__ == '__main__':
    test()