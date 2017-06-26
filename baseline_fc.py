from __future__ import print_function

import os
import sys
import atexit
import random
import argparse
import subprocess
import numpy as np
from logger import Logger
from datetime import datetime
import rpy2.robjects as robjects

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
from torch.nn import init
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--n_epoch', type=int, default=32, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--save_folder', default='.', help='folder to output summaries and checkpoints')
parser.add_argument('--manual_seed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

###############################################################################
n_gpu = opt.n_gpu

###############################################################################
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

cudnn.benchmark = True

###############################################################################
if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manual_seed)

###############################################################################
x_filename = os.path.join(opt.data_root, 'x_training.RData')
y_filename = os.path.join(opt.data_root, 'y_training.RData')

robjects.r['load'](x_filename)
x = np.array(robjects.r['x_train'])
robjects.r['load'](y_filename)
y = np.array(robjects.r['ys']).transpose()

sample_num = x.shape[0]
x_dim = 6026
y_dim = 3

split = int(0.8 * sample_num)
x_train = torch.Tensor(x[:split, :])
y_train = torch.Tensor(y[:split, :])
x_val = torch.Tensor(x[split:, :])
y_val = torch.Tensor(y[split:, :])

dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
dataset_val = torch.utils.data.TensorDataset(x_val, y_val)
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True)


###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


###############################################################################
class Model(nn.Module):
    def __init__(self, n_gpu):
        super(Model, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            nn.Linear(x_dim, 1024, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, y_dim, bias=True),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.n_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.n_gpu))
        else:
            output = self.main(input)

        return output


model = Model(n_gpu)
model.apply(weights_init)
print(model)

###############################################################################
criterion_mse = nn.MSELoss()

x_batch = torch.FloatTensor(opt.batch_size, x_dim)
y_batch = torch.FloatTensor(opt.batch_size, y_dim)

if opt.cuda:
    model.cuda()
    criterion_mse.cuda()
    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

x_batch = Variable(x_batch)
y_batch = Variable(y_batch)

# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

###############################################################################

###############################################################################
model_name = 'baseline_fc'
time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
base_folder = os.path.join(opt.save_folder, model_name, time_string)
folder_ckpt = os.path.join(base_folder, 'ckpts')
folder_summary = os.path.join(base_folder, 'summary')
folder_images = os.path.join(base_folder, 'images')

folders = [opt.save_folder, folder_ckpt, folder_summary, folder_images]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
tensorboard_logger = Logger(folder_summary)

for epoch_idx in range(opt.n_epoch):
    for batch_idx, data_train in enumerate(data_loader_train, 0):
        iter_idx = epoch_idx * len(data_loader_train) + batch_idx

        x_batch_train, y_batch_train = data_train
        x_batch.data.resize_(x_batch_train.size()).copy_(x_batch_train)
        y_batch.data.resize_(y_batch_train.size()).copy_(y_batch_train)

        y_preds = model(x_batch)
        loss_mse = criterion_mse(y_preds, y_batch)
        optimizer.zero_grad()
        loss_mse.backward()
        optimizer.step()

        tensorboard_logger.log_scalar('loss/loss_mse', loss_mse.data[0], iter_idx)
        print('%s-[%03d/%03d][%04d/%04d]--loss_mse: %6.4f' % (
            datetime.now(), epoch_idx, opt.n_epoch, batch_idx, len(data_loader_train), loss_mse.data[0]))
        sys.stdout.flush()

    print('%s-Saving checkpoints...' % (datetime.now()), end='')
    torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (folder_ckpt, epoch_idx))
    print('Done.')

    print('%s-Evaluating model at epoch %03d...' % (datetime.now(), epoch_idx))
    loss_mse_avg = 0
    for batch_idx, data_val in enumerate(data_loader_val, 0):
        x_batch_val, y_batch_val = data_val
        x_batch.data.resize_(x_batch_val.size()).copy_(x_batch_val)
        y_batch.data.resize_(y_batch_val.size()).copy_(y_batch_val)

        y_preds = model(x_batch)
        loss_mse = criterion_mse(y_preds, y_batch)
        loss_mse_avg = loss_mse_avg + loss_mse.data[0]
    loss_mse_avg = loss_mse_avg / len(data_loader_val)

    tensorboard_logger.log_scalar('loss/loss_mse_val', loss_mse_avg, epoch_idx * len(data_loader_train))
    print('%s-[%03d/%03d]--evaluation loss_mse: %6.4f' % (datetime.now(), epoch_idx, opt.n_epoch, loss_mse_avg))
    sys.stdout.flush()

print('%s-Finished Training.' % (datetime.now()))
