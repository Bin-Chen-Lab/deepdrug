import os
import sys
import random
import argparse
import subprocess
import atexit
import numpy as np
from logger import Logger
from datetime import datetime
#import rpy2.robjects as robjects

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


###############################################################################
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(data_dim, 512, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, data_dim, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
class Generator1(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(data_dim, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, data_dim, bias=True),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

def gaussian(ins, is_training = True, mean = 0, stddev = 0.1):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins

def get_next_batch(itor, data_loader):
    try:
        batch, _ = next(itor)
    except StopIteration:
        itor = iter(data_loader)
        batch, _ = next(itor)
    return batch, itor


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True, help='path to dataset')
parser.add_argument('--model_root', required=True, help='path to model')
parser.add_argument('--model_id', type = str, help='model id')
parser.add_argument('--cuda', action='store_true', help='enables cuda') #if cuda is used in training, it should be used in prediction. otherwise, the output is different


parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--n_epoch', type=int, default=4096, help='number of epochs to train for')
parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--lr_d', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--dscrmntor_ckpts', default='', help="path to load dscrmntor checkpoints")
parser.add_argument('--generator_ckpts', default='', help="path to load generator checkpoints")
parser.add_argument('--save_folder', default='.', help='folder to output summaries and checkpoints')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--port', type=int, help='port for tensorboard visualization')
parser.add_argument('--train_d', type=int, default=30, help='# of training D times')
parser.add_argument('--train_g', type=int, default=10, help='# of training G times')
parser.add_argument('--add_noise', type=int, default=0, help='add noise to fake data')



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
data_filename = os.path.join(opt.data_root, 'lincs_signatures_cmpd_landmark_all.npy')

#robj = robjects.r['load'](data_filename)
# for x in robj:
#    print(x)
#data = np.array(robjects.r['lincs_signatures'])
#np.save("data.npy", data)
data = np.load(data_filename)
print(data.shape)

data_dim = data.shape[1]
split = 66511
data_real = data[:split, :]
data_std = np.std(data_real, axis=0)
data_std_tensor = torch.Tensor(data_std)
data_tensor_real = torch.Tensor(data_real)
data_tensor_fake = torch.Tensor(data[split:, :])
dataset_real = torch.utils.data.TensorDataset(data_tensor_real, torch.Tensor(np.ones(shape=(split))))
dataset_fake = torch.utils.data.TensorDataset(data_tensor_fake, torch.Tensor(np.ones(shape=(data.shape[0] - split))))
data_loader_real = torch.utils.data.DataLoader(dataset_real, batch_size=opt.batch_size, shuffle=True)
data_loader_fake = torch.utils.data.DataLoader(dataset_fake, batch_size=opt.batch_size, shuffle=False)

batch_real = torch.FloatTensor(opt.batch_size, data_dim)
batch_fake = torch.FloatTensor(opt.batch_size, data_dim)
zeros = torch.FloatTensor(opt.batch_size, data_dim)
label_real = torch.LongTensor(opt.batch_size)
label_fake = torch.LongTensor(opt.batch_size)
standard_deviation = torch.FloatTensor(opt.batch_size, data_dim)

generator = Generator(ngpu=1)
#generator.apply(weights_init)
if opt.cuda:
    batch_real = batch_real.cuda()
    batch_fake = batch_fake.cuda()
    label_real = label_real.cuda()
    label_fake = label_fake.cuda()
    zeros = zeros.cuda()
    standard_deviation = standard_deviation.cuda()
    generator.cuda()

#dataset_fake = Variable(dataset_fake)
batch_real = Variable(batch_real)
batch_fake = Variable(batch_fake)
label_real = Variable(label_real)
label_fake = Variable(label_fake)
zeros = Variable(zeros)
standard_deviation = Variable(standard_deviation)

model_filename = os.path.join(opt.model_root + "/ckpts/generator_iter_" + str(opt.model_id) + ".pth")

generator.load_state_dict(torch.load(model_filename))
generator.eval()

pred = np.zeros((0,978))
for batch_idx, data_train in enumerate(data_loader_fake):
    batch_fake.data.resize_(data_train[0].size()).copy_(data_train[0])
    standard_deviation.data.resize_(batch_fake.size()).copy_(data_std_tensor.expand_as(batch_fake))
    residual = generator(batch_fake) * (6 * standard_deviation)
    # print(np.mean(residual, axis=0))
    #print(residual[1:,:])
    batch_fake = batch_fake + residual
    # print(batch_fake.data.cpu().numpy().shape()) 
    pred = np.concatenate((pred, batch_fake.data.cpu().numpy()), axis = 0)

#
#pred = dataset_fake + residual
#print(np.around(pred.data.cpu().numpy(),decimals = 4))
pred_path = os.path.join( opt.model_root, "pred")
if not os.path.exists(pred_path):
 os.makedirs(pred_path)

np.savetxt(pred_path + "/" +  str(opt.model_id) + ".txt", pred, fmt = "%1.4f",  delimiter = "\t")

