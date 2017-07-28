from __future__ import print_function

import os
import sys
import atexit
import random
import argparse
import subprocess
import numpy as np
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

from logger import Logger

import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size') #64
parser.add_argument('--n_epoch', type=int, default=32, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--save_folder', default='.', help='folder to output summaries and checkpoints')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--port', type=int, help='port for tensorboard visualization')
parser.add_argument('--add_noise', type=int, default = 0, help='port for tensorboard visualization')


opt = parser.parse_args()
print(opt)

###############################################################################
n_gpu = opt.n_gpu

###############################################################################
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

cudnn.benchmark = True

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

###############################################################################
if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manual_seed)

###############################################################################
###############################################################################
data_filename_np = os.path.join(opt.data_root, 'lincs_signatures_cmpd_landmark_all.npy')
if os.path.exists(data_filename_np):
    x = np.load(data_filename_np)
else:
    data_filename = os.path.join(opt.data_root, 'lincs_signatures_cmpd_landmark_all.RData')
    robj = robjects.r['load'](data_filename)
    x = np.array(robjects.r['lincs_signatures'])
    np.save(data_filename_np, data)
    
x_pred = x[56512:76511, ]
x = x[1:56511, ]
sample_num = x.shape[0]
x_dim = x.shape[1]

split = int(0.8*sample_num)
x_train = torch.Tensor(x[:split, :])
x_val = torch.Tensor(x[split:, :])

dataset_train = torch.utils.data.TensorDataset(x_train, torch.Tensor(x_train.size(0),1).zero_())
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
dataset_val = torch.utils.data.TensorDataset(x_val, torch.Tensor(x_val.size(0), 1).zero_())
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True)

###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins

###############################################################################
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        #self.is_training = is_training
        
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 528, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(528),
            nn.Dropout(0.2),
            nn.Linear(528, 256, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 64, bias=True),
            #nn.LeakyReLU(0.2, inplace=True),
           # nn.Dropout(0.2),
           # nn.Linear(64, 32, bias=True),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(32, 10, bias=True),
        )
        self.decoder = nn.Sequential(
            #nn.Linear(10, 32, bias=True),
           # nn.LeakyReLU(0.2, inplace=True),
           # nn.Linear(32, 64, bias=True),
           # nn.LeakyReLU(0.2, inplace=True),
           # nn.Dropout(0.2),
            nn.Linear(64, 256, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),           
            nn.Dropout(0.2),
            nn.Linear(256, 528, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(528),                       
            nn.Dropout(0.2),
            nn.Linear(528, x_dim, bias=True),
           # nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        #add noise
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

#def criterion_mse(x, y):
#    return (-np.corrcoef(x,y))
    
model = AutoEncoder() #n_gpu
model.apply(weights_init)
print(model)

###############################################################################
criterion_mse = nn.MSELoss() #corLoss(x,y) #

x_batch = torch.FloatTensor(opt.batch_size, x_dim)

if opt.cuda:
    model.cuda()
    criterion_mse.cuda()
    x_batch = x_batch.cuda() 

x_batch = Variable(x_batch)

# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

###############################################################################
#process_tensorboard = subprocess.Popen(["tensorboard", "--logdir", folder_summary, "--port", str(opt.port)])
#atexit.register(process_tensorboard.terminate)
#print('{}-Tensorboard is ready to view at port {}!'.format(datetime.now(), opt.port))

tensorboard_logger = Logger(folder_summary)
for epoch_idx in range(opt.n_epoch):
    for batch_idx, data_train in enumerate(data_loader_train):
        iter_idx = epoch_idx*len(data_loader_train) + batch_idx

        x_batch_train, y  = data_train #y_batch_train
        x_batch.data.resize_(x_batch_train.size()).copy_(x_batch_train)
        b_y = x_batch
        if opt.add_noise:
            x_batch = gaussian(x_batch, True, 0, 0.5)
            
        encoded, decoded = model(x_batch)
        loss_mse = criterion_mse(decoded, b_y)
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
    for batch_idx, data_val in enumerate(data_loader_val):
        x_batch_val,y  = data_val #y_batch_val
        x_batch.data.resize_(x_batch_val.size()).copy_(x_batch_val)
        by_y = x_batch
                
        encoded, decoded = model(x_batch)
        loss_mse = criterion_mse(decoded, by_y)
        loss_mse_avg = loss_mse_avg + loss_mse.data[0]
              
    loss_mse_avg = loss_mse_avg/len(data_loader_val)
    tensorboard_logger.log_scalar('loss/loss_mse_val', loss_mse_avg, epoch_idx*len(data_loader_train))

    print('%s-[%03d/%03d]--evaluation loss_mse: %6.4f' % (datetime.now(), epoch_idx, opt.n_epoch, loss_mse_avg))
    sys.stdout.flush()

view_data = Variable(torch.Tensor(x_pred))
if opt.cuda:
 view_data = view_data.cuda()

encoded_data, decoded_data = model(view_data)

if opt.cuda:
 encoded_data = encoded_data.cpu()
 decoded_data = decoded_data.cpu()

np.savetxt(os.path.join(opt.data_root,"encoder_pred.txt"), encoded_data.data.numpy(),   delimiter = "\t")
np.savetxt(os.path.join(opt.data_root, "decoded_data.txt"), decoded_data.data.numpy()[:, :],   delimiter = "\t") #dfmt = "%1.4f",

print('%s-Finished Training.' % (datetime.now()))
