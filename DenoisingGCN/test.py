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

from tensorboardX import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

k_opt = getParser()

k_loss_writer = SummaryWriter('runs/val_losses')

def reSplitData(data_path):
    num_data = data_path.shape[0]
    val_index = np.load(k_opt.val_res_path + "val_index.npy")
    val_index = list(val_index)
    train_index = list(set(range(0, num_data)) - set(val_index))

    train_path = data_path[train_index]
    val_path = data_path[val_index]

    return train_path, val_path

def test():
    weight_alpha = 0.
    weight_beta = 1.

    for i_ckpt in range(24):
        model_name = "checkpoints/" + str(i_ckpt) + "_model.t7"
        print("Test epoch ", i_ckpt + 1, "/24, model name: ", model_name)

        dgcnn = DGCNN(6, 17, 1024, 0.5)
        # dgcnn = torch.nn.DataParallel(dgcnn)
        # dgcnn.load_state_dict(torch.load(k_opt.current_model))
        dgcnn.load_state_dict(torch.load(model_name))
        print("Load ", model_name, " Success!")
        dgcnn.cuda()
        dgcnn.eval()

        cos_target = torch.tensor(np.ones((k_opt.batch_size, 1)))
        cos_target = cos_target.type(torch.FloatTensor).cuda()

        # all_file_list = getTestList(True)

        data_path_file = k_opt.data_path_file
        data_path = h5py.File(data_path_file, 'r')
        data_path = np.array(data_path["data_path"])
        train_path, val_path = reSplitData(data_path)
        print("Respilt Data Success!")

        test_dataset = MatrixDataset(k_opt, val_path, k_opt.num_neighbors, is_train=False)
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

        val_cos_loss = np.array(val_cos_loss)
        val_value_loss = np.array(val_value_loss)
        val_loss = np.array(val_loss)

        mean_val_cos_loss = np.mean(val_cos_loss)
        mean_val_value_loss = np.mean(val_value_loss)

        print("---", mean_val_cos_loss, "---")

        k_loss_writer.add_scalar('val_cos_loss', mean_val_cos_loss, global_step=i_ckpt + 1)
        k_loss_writer.add_scalar('val_value_loss', mean_val_value_loss, global_step=i_ckpt + 1)

if __name__ == '__main__':
    test()