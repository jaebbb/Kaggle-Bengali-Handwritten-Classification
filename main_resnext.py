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

#from efficientnet_pytorch import EfficientNet

from tqdm import tqdm
#from albumentations import Compose, ShiftScaleRotate, Resize
#from albumentations.pytorch import ToTensor

from logger import CustomLogger
import os.path as osp
import datetime
import yaml

from torchvision.utils import save_image

import albumentations
from albumentations.pytorch import ToTensor
from albumentations import (
	Compose, ShiftScaleRotate, Blur, Resize
)
from gridmask import GridMask


CUR_DIR = os.getcwd()
INPUT_PATH = os.path.join(CUR_DIR, 'data', 'bengaliai')
INPUT_PATH_TRAIN_IMAGES = os.path.join(CUR_DIR, 'data', 'bengaliai', 'train', '256')

parser = argparse.ArgumentParser(description='PyTorch Bengaliai Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch-train', type=int, default=200, help='batch size for train set')
parser.add_argument('--batch-test', type=int, default=200, help='batch size for test set')
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
    Resize(224,224),
	ShiftScaleRotate(rotate_limit=15),
	albumentations.OneOf([
        GridMask(num_grid=3, rotate=15),
        GridMask(num_grid=(3,7)),
        GridMask(num_grid=3, mode=2)
    ], p=1),
	ToTensor()
])

#mean = [x / 255 for x in [125.3, 123.0, 113.9]]
#std = [x / 255 for x in [63.0, 62.1, 66.7]]

# transform_train = transforms.Compose([
#     transforms.Resize(224),
# #    transforms.CenterCrop(224),
# #    transforms.RandomAffine(degrees=10, shear=10),
#     # transforms.RandomResizedCrop(224, scale=(0.3, 1.7)),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

transform_test = albumentations.Compose([
    Resize(224,224),
	ToTensor()
])

# transform_test = transforms.Compose([
#     transforms.Resize(224),
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


train_dataset = BengaliImageDataset(
    csv_file=INPUT_PATH + '/train.csv',
    path=INPUT_PATH_TRAIN_IMAGES,
    transform=transform_train, labels=True
)

#INPUT_PATH = '/home/heechul/pytorch-bengali/data/bengaliai'
#parq1 = INPUT_PATH + '/train_image_data_0.parquet'
#parq2 = INPUT_PATH + '/train_image_data_1.parquet'
#parq3 = INPUT_PATH + '/train_image_data_2.parquet'

#train_dataset = BengaliParquetDataset(
#    parquet_file=[parq1, parq2, parq3],
#    csv_file=INPUT_PATH + '/train.csv',
#    transform=transform_train
#)

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

#parq = [INPUT_PATH + '/train_image_data_3.parquet']
#test_dataset = BengaliParquetDataset(
#    parquet_file=parq,
#    csv_file=INPUT_PATH + '/train.csv',
#    transform=transform_train
#)

testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_test,
    num_workers=2,
    shuffle=False
)

# image 확인용 저장.
print(train_dataset[0]['image'].shape)
# save_image(train_dataset[0]['image'], './hello.jpg')

# Model
print('==> Building model..')
# net = VGG('VGG19')
#net = ResNet18()
# net = BengaliSeNet()
net = BengaliSeNet()

#print(net)
# net = PreActResNet18()
# net = GoogLeNet()
#net = DenseNet201()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = EfficientNetB0()
#net = EfficientNet.from_name('efficientnet-b5')

net = net.to(device)
if device == 'cuda':
#    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/20200111_165324.255004/bengali_99.pth')
    net.load_state_dict(checkpoint)
    best_acc = 0 #checkpoint['acc']
    start_epoch = 100 #checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

if args.amp:
    print('==> Operate amp')
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

if args.scheduler:
    print('==> Operate scheduler')
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1, min_lr=1e-10, verbose=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, total_steps=None, epochs = 100, steps_per_epoch=2826, pct_start=0.0,
    anneal_strategy='cos', cycle_momentum=True,base_momentum=0.85, max_momentum=0.95,  div_factor=100.0) ## Scheduler . Step for each batch

# # logger
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
args.dataset = ['256 original + gridmask']
args.optimizer = 'sgd'
args.model = 'SE-resnext50 (fc3) + mish & weight_decay=2e-4 & batch=50, steps_per_epoch=2826'
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
        # print(inputs.shape)
        # save_image(inputs[30], './hello2.jpg')
       
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

for epoch in range(start_epoch, start_epoch+100):
#    test(epoch)
#    adjust_learning_rate(optimizer, epoch, args)
    train(epoch)
    test_loss = test(epoch)
    # if args.scheduler:
    #    scheduler.step(float(test_loss))
    #scheduler.step()
