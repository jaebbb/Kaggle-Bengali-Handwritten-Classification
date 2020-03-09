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

from utils import progress_bar
from dataset import BengaliImageDataset



from tqdm import tqdm


from logger import CustomLogger
import os.path as osp
import datetime
import yaml

from torchvision.utils import save_image

import albumentations
from albumentations.pytorch import ToTensor
from albumentations import (
	Compose, ShiftScaleRotate, Blur, Resize, Cutout
)
from gridmask import GridMask


CUR_DIR = os.getcwd()
INPUT_PATH = os.path.join(CUR_DIR, 'data', 'bengaliai')
INPUT_PATH_TRAIN_IMAGES = os.path.join(CUR_DIR, 'data', 'bengaliai', 'train', '256')

parser = argparse.ArgumentParser(description='PyTorch Bengaliai Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--model', type = str, default ='Efficientnet', help='Efficientnet or Resnext101_32x4d or GhostNet')
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
transform_train = albumentations.Compose([
	# Rotate(limit=10),
    #    Resize(224, 224),
	ShiftScaleRotate(rotate_limit=15),
	albumentations.OneOf([
        GridMask(num_grid=3, rotate=15),
        GridMask(num_grid=(3,7)),
        GridMask(num_grid=3, mode=2)
    ], p=1),
	ToTensor()
])



transform_test = albumentations.Compose([
#    Resize(224, 224),
   	ToTensor()
])


train_dataset = BengaliImageDataset(
    csv_file=INPUT_PATH + '/train.csv',
    path=INPUT_PATH_TRAIN_IMAGES,
    transform=transform_train, labels=True
)


trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_train,
    num_workers=4,
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

# image 확인용 저장.
print(train_dataset[0]['image'].shape)


# Model
print('==> Building model..')

if args.model == 'Efficientnet':
    net = BengaliEfficientNet(pretrain=True, weight_init=True)
elif args.model == 'Resnext101_32x4d':
    net = BengaliSeNet()
elif args.model == 'GhostNet':
    net = BengaliGhostNet()



net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkp = './checkpoint/20200228_140532.380954/bengali_149.pth'
    checkpoint = torch.load(checkp)
    net.load_state_dict(checkpoint)
    best_acc = 0 #checkpoint['acc']
    start_epoch = int(checkp[-6:-4])+1 #checkpoint['epoch']
    
    

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=3e-4, nesterov=True)

if args.amp:
    print('==> Operate amp')
    net, optimizer = amp.initialize(net, optimizer, opt_level="O0")

if args.scheduler:
    print('==> Operate scheduler')
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1, min_lr=1e-10, verbose=True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,85], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, total_steps=None, epochs = 200, steps_per_epoch=len(trainloader), pct_start=0.0,
            anneal_strategy='cos', cycle_momentum=True,base_momentum=0.85, max_momentum=0.95,  div_factor=100.0) ## Scheduler . Step for each batch
    

    
    
    
# # logger
here = os.getcwd()
now = datetime.datetime.now()
args.out = now.strftime('%m%d%H%M_')+args.model
log_dir = osp.join(here, 'logs', args.out)
os.makedirs(log_dir)
logger = CustomLogger(out=log_dir)

# make dirs for the checkpoint
check_dir = osp.join(here, 'checkpoint', args.out)
os.makedirs(check_dir)

# for .yaml
args.dataset = ['256 original + gridmask']
args.optimizer = 'sgd'
args.model = '%s & weight_decay=3e-4 & batch=%d, steps_per_epoch=%d' %(args.model, args.batch_train, len(trainloader))
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



    for batch_idx, batch in enumerate(trainloader):
        inputs = batch["image"]

       
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

# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10"""
#     #lr = args.lr * (0.1 ** (epoch // 30))
#     if epoch == 0:
#         print('LR is set to {}'.format(args.lr))

#     if epoch % 10 == 0:
#         for param_group in optimizer.param_groups:
#             lr = param_group['lr'] * 0.1
#             param_group['lr'] = lr 
#         print('LR is set to {}'.format(lr))

for epoch in range(start_epoch, start_epoch+200):
#    test(epoch)
#    adjust_learning_rate(optimizer, epoch, args)
    train(epoch)
    test_loss = test(epoch)
    # if args.scheduler:
    #    scheduler.step(float(test_loss))
    #scheduler.step()
