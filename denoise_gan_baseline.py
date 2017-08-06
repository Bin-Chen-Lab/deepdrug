from __future__ import print_function

import os
import sys
import random
import argparse
import numpy as np
from logger import Logger
from datetime import datetime
import rpy2.robjects as robjects

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


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
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--n_epoch', type=int, default=4096, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--d_only', action='store_true', help="train discriminator only")
parser.add_argument('--dscrmntor_ckpts', default='', help="path to load dscrmntor checkpoints")
parser.add_argument('--generator_ckpts', default='', help="path to load generator checkpoints")
parser.add_argument('--encoder_1_ckpts', default='', help="path to load encoder_1 checkpoints")
parser.add_argument('--encoder_2_ckpts', default='', help="path to load encoder_2 checkpoints")
parser.add_argument('--encoder_3_ckpts', default='', help="path to load encoder_3 checkpoints")
parser.add_argument('--save_folder', default='.', help='folder to output summaries and checkpoints')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--port', type=int, help='port for tensorboard visualization')

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
data_filename_np = os.path.join(opt.data_root, 'lincs_signatures_cmpd_landmark_all_feature.npy')
if os.path.exists(data_filename_np):
    data = np.load(data_filename_np)
else:
    data_filename = os.path.join(opt.data_root, 'lincs_signatures_cmpd_landmark_all_feature.RData')
    robj = robjects.r['load'](data_filename)
    data = np.array(robjects.r['lincs_signatures'])
    np.save(data_filename_np, data)
    # for x in robj:
    #    print(x)

print('Input shape is', data.shape)
feature_1_dim = 978
feature_1_encode_dim = 1024
feature_2_dim = 4
feature_2_encode_dim = 512
feature_3_dim = 1024
feature_3_encode_dim = 512
feature_dim = feature_1_encode_dim + feature_2_encode_dim + feature_3_encode_dim
train_num = 66511

if np.isnan(data).any():
    if np.isnan(data[:, :feature_1_dim]).any():
        print('Feature 1 has NaN!')

    if np.isnan(data[:, feature_1_dim:feature_1_dim+feature_2_dim]).any():
        print('Feature 2 has NaN!')

    if np.isnan(data[:, :feature_1_dim+feature_2_dim:]).any():
        print('Feature 3 has NaN!')

    data = np.nan_to_num(data, 0.0)
    print('Replaced all NaNs with zeros.')

data_real = data[:train_num, :]
data_std = np.std(data_real[:, :feature_1_dim], axis=0)
data_std_tensor = torch.Tensor(data_std)
data_tensor_real = torch.Tensor(data_real)
data_tensor_fake = torch.Tensor(data[train_num:, :])
dataset_real = torch.utils.data.TensorDataset(data_tensor_real, torch.Tensor(np.ones(shape=(train_num))))
data_loader_real = torch.utils.data.DataLoader(dataset_real, batch_size=opt.batch_size, shuffle=True)
dataset_fake = torch.utils.data.TensorDataset(data_tensor_fake,
                                              torch.Tensor(np.zeros(shape=(data.shape[0] - train_num))))
data_loader_fake = torch.utils.data.DataLoader(dataset_fake, batch_size=opt.batch_size, shuffle=True)


###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


###############################################################################
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, feature_1_dim),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


generator = Generator(n_gpu)
generator.apply(weights_init)
if opt.generator_ckpts != '':
    generator.load_state_dict(torch.load(opt.generator_ckpts))
print(generator)


###############################################################################
class Dscrmntor(nn.Module):
    def __init__(self, n_gpu):
        super(Dscrmntor, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.n_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.n_gpu))
        else:
            output = self.main(input)

        return output


dscrmntor = Dscrmntor(n_gpu)
dscrmntor.apply(weights_init)
if opt.dscrmntor_ckpts != '':
    dscrmntor.load_state_dict(torch.load(opt.dscrmntor_ckpts))
print(dscrmntor)


###############################################################################
class Encoder_1(nn.Module):
    def __init__(self, ngpu):
        super(Encoder_1, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(feature_1_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, feature_1_encode_dim),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


encoder_1 = Encoder_1(n_gpu)
encoder_1.apply(weights_init)
if opt.encoder_1_ckpts != '':
    encoder_1.load_state_dict(torch.load(opt.encoder_1_ckpts))
print(encoder_1)


###############################################################################
class Encoder_2(nn.Module):
    def __init__(self, ngpu):
        super(Encoder_2, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose1d(feature_2_dim, 128, 1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(128, 256, 1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(256, feature_2_encode_dim, 1, groups=4),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


encoder_2 = Encoder_2(n_gpu)
encoder_2.apply(weights_init)
if opt.encoder_2_ckpts != '':
    encoder_2.load_state_dict(torch.load(opt.encoder_2_ckpts))
print(encoder_2)


###############################################################################
class Encoder_3(nn.Module):
    def __init__(self, ngpu):
        super(Encoder_3, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(feature_3_dim, 768),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(768, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, feature_3_encode_dim),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


encoder_3 = Encoder_3(n_gpu)
encoder_3.apply(weights_init)
if opt.encoder_3_ckpts != '':
    encoder_3.load_state_dict(torch.load(opt.encoder_3_ckpts))
print(encoder_3)


###############################################################################
criterion_l1 = nn.L1Loss()
criterion_cse = nn.CrossEntropyLoss()

batch_real_1 = torch.FloatTensor(opt.batch_size, feature_1_dim)
batch_real_2 = torch.FloatTensor(opt.batch_size, feature_2_dim)
batch_real_3 = torch.FloatTensor(opt.batch_size, feature_3_dim)
batch_fake_1 = torch.FloatTensor(opt.batch_size, feature_1_dim)
batch_fake_2 = torch.FloatTensor(opt.batch_size, feature_2_dim)
batch_fake_3 = torch.FloatTensor(opt.batch_size, feature_3_dim)
zeros = torch.FloatTensor(opt.batch_size, feature_dim)
label_real = torch.LongTensor(opt.batch_size)
label_fake = torch.LongTensor(opt.batch_size)
standard_deviation = torch.FloatTensor(opt.batch_size, feature_1_dim)

if opt.cuda:
    batch_real_1 = batch_real_1.cuda()
    batch_real_2 = batch_real_2.cuda()
    batch_real_3 = batch_real_3.cuda()
    batch_fake_1 = batch_fake_1.cuda()
    batch_fake_2 = batch_fake_2.cuda()
    batch_fake_3 = batch_fake_3.cuda()
    label_real = label_real.cuda()
    label_fake = label_fake.cuda()
    zeros = zeros.cuda()
    standard_deviation = standard_deviation.cuda()
    generator.cuda()
    dscrmntor.cuda()
    encoder_1.cuda()
    encoder_2.cuda()
    encoder_3.cuda()
    criterion_l1.cuda()
    criterion_cse.cuda()

batch_real_1 = Variable(batch_real_1)
batch_real_2 = Variable(batch_real_2)
batch_real_3 = Variable(batch_real_3)
batch_fake_1 = Variable(batch_fake_1)
batch_fake_2 = Variable(batch_fake_2)
batch_fake_3 = Variable(batch_fake_3)
label_real = Variable(label_real)
label_fake = Variable(label_fake)
zeros = Variable(zeros)
standard_deviation = Variable(standard_deviation)

# setup optimizer
optimizer_g = optim.Adam(generator.parameters(), lr=opt.lr * 10, betas=(opt.beta1, 0.9), eps=0.01)
d_parameters = list(dscrmntor.parameters())+list(encoder_1.parameters())+list(encoder_2.parameters())+list(encoder_3.parameters())
optimizer_d = optim.Adam(d_parameters, lr=opt.lr, betas=(opt.beta1, 0.9), eps=0.01)

###############################################################################
model_name = 'denoise_gan'
time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
base_folder = os.path.join(opt.save_folder, model_name, time_string + '_baseline')
folder_ckpt = os.path.join(base_folder, 'ckpts')
folder_summary = os.path.join(base_folder, 'summary')

folders = [opt.save_folder, folder_ckpt, folder_summary]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

###############################################################################
tensorboard_logger = Logger(folder_summary)
train_d = True
train_d_iter = 0
train_g_iter = 0
iter_real = iter(data_loader_real)
iter_fake = iter(data_loader_fake)
for iter_idx in range(1, opt.n_epoch * len(data_loader_real) + 1):
    samples_fake, iter_fake = get_next_batch(iter_fake, data_loader_fake)
    samples_fake_1 = samples_fake.narrow(1, 0, feature_1_dim)
    samples_fake_2 = samples_fake.narrow(1, feature_1_dim, feature_2_dim)
    samples_fake_3 = samples_fake.narrow(1, feature_1_dim + feature_2_dim, feature_3_dim)

    batch_fake_1.data.resize_(samples_fake_1.size()).copy_(samples_fake_1)
    batch_fake_2.data.resize_(samples_fake_2.size()).copy_(samples_fake_2)
    batch_fake_3.data.resize_(samples_fake_3.size()).copy_(samples_fake_3)
    standard_deviation.data.resize_(samples_fake_1.size()).copy_(data_std_tensor.expand_as(samples_fake_1))

    batch_fake_1_encode = encoder_1(batch_fake_1)
    batch_fake_2_encode = encoder_2(batch_fake_2.unsqueeze(2)).squeeze()
    batch_fake_3_encode = encoder_3(batch_fake_3)
    residual = generator(torch.cat([batch_fake_1_encode, batch_fake_2_encode, batch_fake_3_encode], dim=1))

    batch_fake_1_residual = batch_fake_1 + residual * (0 if opt.d_only else (6 * standard_deviation))
    batch_fake_1_residual_encode = encoder_1(batch_fake_1_residual)
    logits_fake = dscrmntor(torch.cat([batch_fake_1_residual_encode, batch_fake_2_encode, batch_fake_3_encode], dim=1))

    if train_d:
        samples_real, iter_real = get_next_batch(iter_real, data_loader_real)
        samples_real_1 = samples_real.narrow(1, 0, feature_1_dim)
        samples_real_2 = samples_real.narrow(1, feature_1_dim, feature_2_dim)
        samples_real_3 = samples_real.narrow(1, feature_1_dim + feature_2_dim, feature_3_dim)

        batch_real_1.data.resize_(samples_real_1.size()).copy_(samples_real_1)
        batch_real_2.data.resize_(samples_real_2.size()).copy_(samples_real_2)
        batch_real_3.data.resize_(samples_real_3.size()).copy_(samples_real_3)

        batch_real_1_encode = encoder_1(batch_real_1)
        batch_real_2_encode = encoder_2(batch_real_2.unsqueeze(2)).squeeze()
        batch_real_3_encode = encoder_3(batch_real_3)
        logits_real = dscrmntor(torch.cat([batch_real_1_encode, batch_real_2_encode, batch_real_3_encode], dim=1))

        label_real.data.resize_(samples_real.size(0)).fill_(1)
        loss_real = criterion_cse(logits_real, label_real)

        label_fake.data.resize_(samples_fake.size(0)).fill_(0)
        loss_fake = criterion_cse(logits_fake, label_fake)

        precision_real = accuracy(logits_real.data, label_real.data)[0]
        precision_fake = accuracy(logits_fake.data, label_fake.data)[0]

        loss_d = loss_real + loss_fake

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        tensorboard_logger.log_scalar('loss/d/real', loss_real.data[0], iter_idx)
        tensorboard_logger.log_scalar('loss/d/fake', loss_fake.data[0], iter_idx)
        tensorboard_logger.log_scalar('precision/d/real', precision_real[0], iter_idx)
        tensorboard_logger.log_scalar('precision/d/fake', precision_fake[0], iter_idx)
        print('%s-loss real/fake: %6.4f/%6.4f    precision real/fake: %6.4f/%6.4f' %
              (datetime.now(), loss_real.data[0], loss_fake.data[0], precision_real[0], precision_fake[0]))

        train_d_iter = train_d_iter + 1
        if (not opt.d_only) and ((precision_real[0] > 90 and precision_fake[0] > 90) or train_d_iter > 10):
            train_d_iter = 0
            train_d = False
    else:  # train g
        label_fake.data.resize_(samples_fake.size(0)).fill_(1)
        loss_fake = criterion_cse(logits_fake, label_fake)
        precision_fake = accuracy(logits_fake.data, label_fake.data)[0]

        zeros.data.resize_(samples_fake_1.size()).fill_(0.0)
        loss_rsdu = criterion_l1(residual, zeros)

        loss_g = loss_fake + 10 * loss_rsdu

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        tensorboard_logger.log_scalar('loss/g/fake', loss_fake.data[0], iter_idx)
        tensorboard_logger.log_scalar('loss/g/rsdu', loss_rsdu.data[0], iter_idx)
        tensorboard_logger.log_scalar('precision/g/fake', precision_fake[0], iter_idx)
        print('%s-loss rsdu/fake: %6.4f/%6.4f    precision fake: %6.4f' %
              (datetime.now(), loss_rsdu.data[0], loss_fake.data[0], precision_fake[0]))
        train_g_iter = train_g_iter + 1
        if precision_fake[0] > 90 or train_g_iter > 30:
            train_g_iter = 0
            train_d = True

    sys.stdout.flush()

    if iter_idx % 5000 == 0:
        print('%s-Saving checkpoints...' % (datetime.now()), end='')
        torch.save(generator.state_dict(), '%s/generator_iter_%d.pth' % (folder_ckpt, iter_idx))
        torch.save(dscrmntor.state_dict(), '%s/dscrmntor_iter_%d.pth' % (folder_ckpt, iter_idx))
        torch.save(encoder_1.state_dict(), '%s/encoder_1_iter_%d.pth' % (folder_ckpt, iter_idx))
        torch.save(encoder_2.state_dict(), '%s/encoder_2_iter_%d.pth' % (folder_ckpt, iter_idx))
        torch.save(encoder_3.state_dict(), '%s/encoder_3_iter_%d.pth' % (folder_ckpt, iter_idx))
        print('Done.')

print('%s-Finished Training.' % (datetime.now()))
