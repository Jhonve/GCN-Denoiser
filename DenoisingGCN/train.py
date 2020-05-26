import os

import torch
import torch.optim as Toptim
import torch.utils.data
import torch.nn.functional as F

import h5py
import numpy as np
import random

from GCNModel import DGCNN
from parsers import getParser
from datautils import MatrixDataset

from tensorboardX import SummaryWriter

k_opt = getParser()
k_epoch = k_opt.num_epoch

k_loss_writer = SummaryWriter('runs/losses')

if not os.path.exists(k_opt.val_res_path):
    os.makedirs(k_opt.val_res_path)

if not os.path.exists(k_opt.ckpt_path):
    os.makedirs(k_opt.ckpt_path)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def splitData(data_path, num_val_batch):
    num_data = data_path.shape[0]
    num_val_data = num_val_batch * k_opt.batch_size
    num_train_data = num_data - num_val_data

    val_index = random.sample(range(0, num_data), num_val_data)
    train_index = list(set(range(0, num_data)) - set(val_index))

    train_path = data_path[train_index]
    val_path = data_path[val_index]

    val_index = np.array(val_index)
    np.save(k_opt.val_res_path + "val_index.npy", val_index)

    return train_path, val_path

def reSplitData(data_path):
    num_data = data_path.shape[0]
    val_index = np.load(k_opt.val_res_path + "val_index.npy")
    val_index = list(val_index)
    train_index = list(set(range(0, num_data)) - set(val_index))

    train_path = data_path[train_index]
    val_path = data_path[val_index]

    return train_path, val_path

def train():
    data_path_file = k_opt.data_path_file
    data_path = h5py.File(data_path_file, 'r')
    data_path = np.array(data_path["data_path"])

    if k_opt.current_model != "":
        train_path, val_path = reSplitData(data_path)
    else:
        train_path, val_path = splitData(data_path, k_opt.num_val_batch)
    num_train_batch = int(train_path.shape[0] / k_opt.batch_size)

    # initialize Dataloader
    train_dataset = MatrixDataset(k_opt, train_path, k_opt.num_neighbors, is_train=True)
    train_data_loader = train_dataset.getDataloader()

    val_dataset = MatrixDataset(k_opt, val_path, k_opt.num_neighbors, is_train=False)
    val_data_loader = val_dataset.getDataloader()

    # initialize Network structure etc.
    current_epoch = 0
    dgcnn = DGCNN(8, 17, 1024, 0.5)
    # dgcnn = torch.nn.DataParallel(dgcnn)
    if k_opt.current_model != "":
        dgcnn.load_state_dict(torch.load(k_opt.current_model))
        print("Load ", k_opt.current_model, " Success!")
        current_epoch = int(k_opt.current_model.split('/')[-1].split('_')[0]) + 1
    optimizer = Toptim.Adam(dgcnn.parameters(), lr=k_opt.learning_rate, betas=(0.9, 0.999))
    dgcnn.cuda()

    cos_target = torch.tensor(np.ones((k_opt.batch_size, 1)))
    cos_target = cos_target.type(torch.FloatTensor).cuda()
    weight_alpha = 0.
    weight_beta = 1.

    last_val_cos_loss = 999.
    last_val_value_loss = 999.
    for epoch in range(current_epoch, k_epoch):
        for i_train, data in enumerate(train_data_loader, 0):
            inputs, gt_res, gt_norm, center_norm = data
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.permute(0, 2, 1)
            gt_norm = gt_norm.type(torch.FloatTensor)

            inputs = inputs.cuda()
            gt_norm = gt_norm.cuda()

            optimizer.zero_grad()
            dgcnn = dgcnn.train()

            output = dgcnn(inputs)

            cos_loss = F.cosine_embedding_loss(output, gt_norm, cos_target)
            value_loss = F.mse_loss(output, gt_norm)

            if(i_train % 100 == 0):
                k_loss_writer.add_scalar('cos_loss', cos_loss, global_step=epoch * num_train_batch + i_train + 1)
                k_loss_writer.add_scalar('value_loss', value_loss, global_step=epoch * num_train_batch + i_train + 1)

            loss = weight_alpha * cos_loss + weight_beta * value_loss
            loss.backward()
            optimizer.step()

            print("Epcoh: %d, || Batch: %d/%d, || cos loss: %.7f, || value loss: %.7f, || val cos loss: %.7f || val value loss: %.7f" % \
                (epoch, i_train + 1, num_train_batch, cos_loss.data.item(), value_loss.data.item(), last_val_cos_loss, last_val_value_loss))
        
        #______Validation______
        torch.save(dgcnn.state_dict(), k_opt.ckpt_path + str(epoch) + "_model.t7")
        val_cos_loss = []
        val_value_loss = []
        val_loss = []
        dgcnn.eval()
        for i_val, data in enumerate(val_data_loader, 0):
            inputs, gt_res, gt_norm, center_norm = data
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.permute(0, 2, 1)
            gt_norm = gt_norm.type(torch.FloatTensor)

            inputs = inputs.cuda()
            gt_norm = gt_norm.cuda()

            output = dgcnn(inputs)

            cos_loss = F.cosine_embedding_loss(output, gt_norm, cos_target)
            value_loss = F.mse_loss(output, gt_norm)

            loss = weight_alpha * cos_loss + weight_beta * value_loss

            val_loss.append(loss.data.item())
            val_cos_loss.append(cos_loss.data.item())
            val_value_loss.append(value_loss.data.item())

            print("Epcoh: %d, || Val Batch: %d/%d, || cos loss: %.7f, || value loss: %.7f" % \
                (epoch, i_val + 1, k_opt.num_val_batch, cos_loss.data.item(), value_loss.data.item()))
        
        val_cos_loss = np.array(val_cos_loss)
        val_value_loss = np.array(val_value_loss)
        val_loss = np.array(val_loss)

        last_val_cos_loss = np.mean(val_cos_loss)
        last_val_value_loss = np.mean(val_value_loss)

        k_loss_writer.add_scalar('val_cos_loss', last_val_cos_loss, global_step=epoch + 1)
        k_loss_writer.add_scalar('val_value_loss', last_val_value_loss, global_step=epoch + 1)

if __name__ == '__main__':
    train()
