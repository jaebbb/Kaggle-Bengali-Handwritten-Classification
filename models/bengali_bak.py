import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class BengaliModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
#        self.backbone = torchvision.models.resnet34(pretrained=False)
        self.backbone = EfficientNet.from_name('efficientnet-b0')
        print(self.backbone)
#        self.backbone.load_state_dict(torch.load("/home/heechul/pytorch-cifar/resnet18.pth"))

        in_features = self.backbone._fc.in_features

        self.bn1_1 = nn.BatchNorm1d(num_features=in_features, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        self.bn1_2 = nn.BatchNorm1d(num_features=in_features, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        self.bn1_3 = nn.BatchNorm1d(num_features=in_features, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)

        self.bn2_1 = nn.BatchNorm1d(num_features=1000, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        self.bn2_2 = nn.BatchNorm1d(num_features=1000, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        self.bn2_3 = nn.BatchNorm1d(num_features=1000, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)

        self.bn3_1 = nn.BatchNorm1d(num_features=1000, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        self.bn3_2 = nn.BatchNorm1d(num_features=1000, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        self.bn3_3 = nn.BatchNorm1d(num_features=1000, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)

        self.fc_graph1 = torch.nn.Linear(in_features, 1000)
        self.fc_vowel1 = torch.nn.Linear(in_features, 1000)
        self.fc_conso1 = torch.nn.Linear(in_features, 1000)

        self.fc_graph2 = torch.nn.Linear(1000, 1000)
        self.fc_vowel2 = torch.nn.Linear(1000, 1000)
        self.fc_conso2 = torch.nn.Linear(1000, 1000)

        self.fc_graph3 = torch.nn.Linear(in_features, 168)
        self.fc_vowel3 = torch.nn.Linear(in_features, 11)
        self.fc_conso3 = torch.nn.Linear(in_features, 7)

    def forward(self, x):

        x = self.backbone._conv_stem(x)
        x = self.backbone._bn0(x)

        for layer in self.backbone._blocks:
            x = layer(x)

        x = self.backbone._conv_head(x)
        x = self.backbone._bn1(x)

#        print(x.size())
        x = self.backbone._avg_pooling(x)
#        print(x.size())
        x = self.backbone._dropout(x)
#        x = self.backbone.layer1(x)
#        x = self.backbone.layer2(x)
#        x = self.backbone.layer3(x)
#        x = self.backbone.layer4(x)

#        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

#        x1 = self.bn1_1(x)
#        x2 = self.bn1_2(x)
#        x3 = self.bn1_3(x)

#        x1 = F.relu(x1)
#        x2 = F.relu(x2)
#        x3 = F.relu(x3)

#        x1 = self.fc_graph1(x1)
#        x2 = self.fc_vowel1(x2)
#        x3 = self.fc_conso1(x3)

#        x1 = self.bn2_1(x1)
#        x2 = self.bn2_2(x2)
#        x3 = self.bn2_3(x3)

#        x1 = F.relu(x1)
#        x2 = F.relu(x2)
#        x3 = F.relu(x3)

#        x1 = self.fc_graph2(x1)
#        x2 = self.fc_vowel2(x2)
#        x3 = self.fc_conso2(x3)

#        x1 = self.bn3_1(x1)
#        x2 = self.bn3_2(x2)
#        x3 = self.bn3_3(x3)

#        x1 = F.relu(x1)
#        x2 = F.relu(x2)
#        x3 = F.relu(x3)

        fc_graph = self.fc_graph3(x)
        fc_vowel = self.fc_vowel3(x)
        fc_conso = self.fc_conso3(x)

#        fc_graph = self.backbone._swish(self.fc_graph3(x1))
#        fc_vowel = self.backbone._swish(self.fc_vowel3(x2))
#        fc_conso = self.backbone._swish(self.fc_conso3(x3))

#        fc_graph = self.fc_graph(x)
#        fc_vowel = self.fc_vowel(x)
#        fc_conso = self.fc_conso(x)

        return fc_graph, fc_vowel, fc_conso
