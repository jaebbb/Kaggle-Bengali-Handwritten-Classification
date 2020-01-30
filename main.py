'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from apex import amp


from models import *
from models.bengalimish import BengaliMishModel
from utils import progress_bar
from dataset import BengaliImageDataset
from dataset import BengaliParquetDataset

#from efficientnet_pytorch import EfficientNet

from tqdm import tqdm
#from albumentations import Compose, ShiftScaleRotate, Resize
#from albumentations.pytorch import ToTensor

from logger import CustomLogger
import os.path as osp


import datetime
import yaml









## Over9000 Optimizer . Inspired by Iafoss . Over and Out !
##https://github.com/mgrankin/over9000/blob/master/ralamb.py
import torch, math
from torch.optim.optimizer import Optimizer

# RAdam + LARS
class Ralamb(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(Ralamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Ralamb, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ralamb does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, radam_step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        radam_step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        radam_step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = radam_step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                radam_step = p_data_fp32.clone()
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    radam_step.addcdiv_(-radam_step_size * group['lr'], exp_avg, denom)
                else:
                    radam_step.add_(-radam_step_size * group['lr'], exp_avg)

                radam_norm = radam_step.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio

                if N_sma >= 5:
                    p_data_fp32.addcdiv_(-radam_step_size * group['lr'] * trust_ratio, exp_avg, denom)
                else:
                    p_data_fp32.add_(-radam_step_size * group['lr'] * trust_ratio, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

# Lookahead implementation from https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lookahead.py

""" Lookahead Optimizer Wrapper.
Implementation modified from: https://github.com/alphadl/lookahead.pytorch
Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610
"""
import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict

class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # print(self.k)
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)

def LookaheadAdam(params, alpha=0.5, k=6, *args, **kwargs):
     adam = Adam(params, *args, **kwargs)
     return Lookahead(adam, alpha, k)


# RAdam + LARS + LookAHead

# Lookahead implementation from https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
# RAdam + LARS implementation from https://gist.github.com/redknightlois/c4023d393eb8f92bb44b2ab582d7ec20

def Over9000(params, alpha=0.5, k=6, *args, **kwargs):
     ralamb = Ralamb(params, *args, **kwargs)
     return Lookahead(ralamb, alpha, k)


# optimizer import












CUR_DIR = os.getcwd()
INPUT_PATH = os.path.join(CUR_DIR, 'data', 'bengaliai')
INPUT_PATH_TRAIN_IMAGES = os.path.join(CUR_DIR, 'data', 'bengaliai', 'train', 'pre256')

parser = argparse.ArgumentParser(description='PyTorch Bengaliai Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch-train', type=int, default=50, help='batch size for train set')
parser.add_argument('--batch-test', type=int, default=50, help='batch size for test set')
parser.add_argument('--amp',  action='store_true', help='train with amp')
parser.add_argument('--scheduler',  action='store_true', help='train with scheduler')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])



transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])



train_dataset = BengaliImageDataset(
    csv_file=INPUT_PATH + '/train_1.csv',
    path=INPUT_PATH_TRAIN_IMAGES,
    transform=transform_train, labels=True
)

trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_train,
    num_workers=2,
    shuffle=True
)

test_dataset = BengaliImageDataset(
    csv_file=INPUT_PATH + '/val_1.csv',
    path=INPUT_PATH_TRAIN_IMAGES,
    transform=transform_test, labels=True
)


testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_test,
    num_workers=2,
    shuffle=False
)



print('==> Building model..')
net = BengaliEfficientNet()

net = net.to(device)
if device == 'cuda':
#    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



if args.resume ==False:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/20200127_152001.070413/bengali_99.pth')
    net.load_state_dict(checkpoint)
    best_acc = 0 #checkpoint['acc']
    start_epoch = 100 #checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer =Over9000(net.parameters(), lr=2e-3, weight_decay=1e-3) ## New once 

if args.amp:
    print('==> Operate amp')
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

if args.scheduler:
    print('==> Operate scheduler')
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, total_steps=None, epochs=100, steps_per_epoch=5021, pct_start=0.0,
                                   anneal_strategy='cos', cycle_momentum=True,base_momentum=0.85, max_momentum=0.95,  div_factor=100.0)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1, min_lr=1e-10, verbose=True)
   # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, total_steps=None, epochs = 100, steps_per_epoch=3617, pct_start=0.0,
    #anneal_strategy='cos', cycle_momentum=True,base_momentum=0.85, max_momentum=0.95,  div_factor=100.0) ## Scheduler . Step for each batch

# logger
here = os.getcwd()
now = datetime.datetime.now()
args.out = now.strftime('%Y%m%d_%H%M%S.%f')
log_dir = osp.join(here, 'logs', args.out)
os.makedirs(log_dir)
logger = CustomLogger(out=log_dir)

# make dirs for the checkpoint
check_dir = osp.join(here, 'checkpoint', args.out)
os.makedirs(check_dir)

# for .yaml
args.dataset = ['original']
args.optimizer = 'SGD'
args.model = 'EfficientNetB5 (fc3) & weight_decay=5e-4 & batch=50, steps_per_epoch=3617'
args.scheduler = 'OneCycleLR'

with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct_graph = 0
    correct_vowel = 0
    correct_conso = 0
    total = 0

#    tk0 = tqdm(trainloader, desc="Iteration")

    for batch_idx, batch in enumerate(trainloader):
        inputs = batch["image"]
        #print(inputs)
        l_graph = batch["l_graph"]
        l_vowel = batch["l_vowel"]
        l_conso = batch["l_conso"]

        inputs = inputs.to(device, dtype=torch.float)
        l_graph = l_graph.to(device, dtype=torch.long)
        l_vowel = l_vowel.to(device, dtype=torch.long)
        l_conso = l_conso.to(device, dtype=torch.long)

        optimizer.zero_grad()
        out_graph, out_vowel, out_conso = net(inputs)

        loss_graph = criterion(out_graph, l_graph)
        loss_vowel = criterion(out_vowel, l_vowel)
        loss_conso = criterion(out_conso, l_conso)

        loss = 1.5 * loss_graph + 0.75 * loss_vowel + 0.75 * loss_conso

        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
             loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        # for OneCycleLR scheduler
        if args.scheduler:
            scheduler.step()

        train_loss += loss.item()
        _, pred_graph = out_graph.max(1)
        _, pred_vowel = out_vowel.max(1)
        _, pred_conso = out_conso.max(1)

        total += l_graph.size(0)
        correct_graph += pred_graph.eq(l_graph).sum().item()
        correct_vowel += pred_vowel.eq(l_vowel).sum().item()
        correct_conso += pred_conso.eq(l_conso).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc: %.3f%% (%d/%d) | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct_graph/total, correct_graph, total, 100.*correct_vowel/total, correct_vowel, total, 100.*correct_conso/total, correct_conso, total))


    # Save checkpoint.
    print('Saving..')
    torch.save(net.state_dict(), osp.join(check_dir, 'bengali_{}.pth'.format(epoch)))
    logger.write(True, epoch, batch_idx, train_loss/(batch_idx+1), 100.*correct_graph/total, 100.*correct_vowel/total, 100.*correct_conso/total, get_learing_rate(optimizer))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct_graph = 0
    correct_vowel = 0
    correct_conso = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            inputs = batch["image"]
            l_graph = batch["l_graph"]
            l_vowel = batch["l_vowel"]
            l_conso = batch["l_conso"]

            inputs = inputs.to(device, dtype=torch.float)
            l_graph = l_graph.to(device, dtype=torch.long)
            l_vowel = l_vowel.to(device, dtype=torch.long)
            l_conso = l_conso.to(device, dtype=torch.long)

            out_graph, out_vowel, out_conso = net(inputs)

            loss_graph = criterion(out_graph, l_graph)
            loss_vowel = criterion(out_vowel, l_vowel)
            loss_conso = criterion(out_conso, l_conso)

            loss = 1.5 * loss_graph + 0.75 * loss_vowel + 0.75 * loss_conso

            test_loss += loss.item()
            _, pred_graph = out_graph.max(1)
            _, pred_vowel = out_vowel.max(1)
            _, pred_conso = out_conso.max(1)
 
#           _, predicted = outputs.max(1)
#            total += targets.size(0)
#            correct += predicted.eq(targets).sum().item()

            total += l_graph.size(0)
            correct_graph += pred_graph.eq(l_graph).sum().item()
            correct_vowel += pred_vowel.eq(l_vowel).sum().item()
            correct_conso += pred_conso.eq(l_conso).sum().item()

#            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc: %.3f%% (%d/%d) | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct_graph/total, correct_graph, total, 100.*correct_vowel/total, correct_vowel, total, 100.*correct_conso/total, correct_conso, total))

    logger.write(False, epoch, batch_idx, test_loss/(batch_idx+1), 100.*correct_graph/total, 100.*correct_vowel/total, 100.*correct_conso/total)
    # Save checkpoint.
#    acc = 100.*correct/total
#    if acc > best_acc:
#        print('Saving..')
#        state = {
#            'net': net.state_dict(),
#            'acc': acc,
#            'epoch': epoch,
#        }
#        if not os.path.isdir('checkpoint'):
#            os.mkdir('checkpoint')
#        torch.save(state, './checkpoint/ckpt.pth')
#        best_acc = acc
    return test_loss


def get_learing_rate(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    if epoch == 0:
        print('LR is set to {}'.format(args.lr))

    if epoch == 10 or epoch == 20 or epoch == 30 or epoch == 40 or epoch == 50:
        for param_group in optimizer.param_groups:
            lr = param_group['lr'] * 0.1
            param_group['lr'] = lr 
        print('LR is set to {}'.format(lr))

for epoch in range(start_epoch, start_epoch+100):
#    test(epoch)
#    adjust_learning_rate(optimizer, epoch, args)
    train(epoch)
    test_loss = test(epoch)
    # if args.scheduler:
    #    scheduler.step(float(test_loss))
    #scheduler.step()
