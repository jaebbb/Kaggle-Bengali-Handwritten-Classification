import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet
from models.senet import se_resnext101_32x4d_init
from models.ghost_net import ghost_net


#from Mish import Mish

class BengaliEfficientNet(torch.nn.Module):

    def __init__(self, pretrain=True, weight_init = True):
        super().__init__()
        self.backbone = None
        if pretrain:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b5',advprop=True)
            if weight_init:
              #  self.backbone._blocks.apply(self.weights_init)
                self.backbone._conv_head.apply(self.weights_init)
        else:
            self.backbone = EfficientNet.from_name('efficientnet-b5')
        #self.backbone.load_state_dict(torch.load(state_dict))
        in_features = self.backbone._fc.in_features

        self.bn1_1 = nn.BatchNorm1d(num_features=in_features)
        self.bn1_2 = nn.BatchNorm1d(num_features=in_features)
        self.bn1_3 = nn.BatchNorm1d(num_features=in_features)

        self.bn2_1 = nn.BatchNorm1d(num_features=1000)
        self.bn2_2 = nn.BatchNorm1d(num_features=1000)
        self.bn2_3 = nn.BatchNorm1d(num_features=1000)

        self.bn3_1 = nn.BatchNorm1d(num_features=1000)
        self.bn3_2 = nn.BatchNorm1d(num_features=1000)
        self.bn3_3 = nn.BatchNorm1d(num_features=1000)

        self.fc_graph1 = torch.nn.Linear(in_features, 1000)
        self.fc_vowel1 = torch.nn.Linear(in_features, 1000)
        self.fc_conso1 = torch.nn.Linear(in_features, 1000)

        self.fc_graph2 = torch.nn.Linear(1000, 1000)
        self.fc_vowel2 = torch.nn.Linear(1000, 1000)
        self.fc_conso2 = torch.nn.Linear(1000, 1000)

        self.fc_graph3 = torch.nn.Linear(1000, 168)
        self.fc_vowel3 = torch.nn.Linear(1000, 11)
        self.fc_conso3 = torch.nn.Linear(1000, 7)


        # self._init_weights()

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self.backbone._swish(self.backbone._bn1(self.backbone._conv_head(x)))

        return x
    
    
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



    def forward(self, inputs):      

        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self.backbone._avg_pooling(x)
        x = x.view(bs, -1)
        x = self.backbone._dropout(x)

        x1 = self.bn1_1(x)
        x2 = self.bn1_2(x)
        x3 = self.bn1_3(x)

        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)

        x1 = self.fc_graph1(x1)
        x2 = self.fc_vowel1(x2)
        x3 = self.fc_conso1(x3)

        x1 = self.bn2_1(x1)
        x2 = self.bn2_2(x2)
        x3 = self.bn2_3(x3)

        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)

        x1 = self.fc_graph2(x1)
        x2 = self.fc_vowel2(x2)
        x3 = self.fc_conso2(x3)

        x1 = self.bn3_1(x1)
        x2 = self.bn3_2(x2)
        x3 = self.bn3_3(x3)

        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)

        fc_graph = self.fc_graph3(x1)
        fc_vowel = self.fc_vowel3(x2)
        fc_conso = self.fc_conso3(x3)

        return fc_graph, fc_vowel, fc_conso


class BengaliSeNet(torch.nn.Module):

    def __init__(self, init=True):
        super().__init__()
        if init:
            self.backbone = se_resnext101_32x4d_init()
        else:
            self.backbone = se_resnext101_32x4d()
        #self.backbone.load_state_dict(torch.load(state_dict))
        in_features = self.backbone.last_linear.in_features
        in_features = 8192
        print('in_features:', in_features)
        self.bn1_1 = nn.BatchNorm1d(num_features=in_features)
        self.bn1_2 = nn.BatchNorm1d(num_features=in_features)
        self.bn1_3 = nn.BatchNorm1d(num_features=in_features)

        self.bn2_1 = nn.BatchNorm1d(num_features=1000)
        self.bn2_2 = nn.BatchNorm1d(num_features=1000)
        self.bn2_3 = nn.BatchNorm1d(num_features=1000)

        self.bn3_1 = nn.BatchNorm1d(num_features=1000)
        self.bn3_2 = nn.BatchNorm1d(num_features=1000)
        self.bn3_3 = nn.BatchNorm1d(num_features=1000)

        self.fc_graph1 = torch.nn.Linear(in_features, 1000)
        self.fc_vowel1 = torch.nn.Linear(in_features, 1000)
        self.fc_conso1 = torch.nn.Linear(in_features, 1000)

        self.fc_graph2 = torch.nn.Linear(1000, 1000)
        self.fc_vowel2 = torch.nn.Linear(1000, 1000)
        self.fc_conso2 = torch.nn.Linear(1000, 1000)

        self.fc_graph3 = torch.nn.Linear(1000, 168)
        self.fc_vowel3 = torch.nn.Linear(1000, 11)
        self.fc_conso3 = torch.nn.Linear(1000, 7)


        # self._init_weights()

    def forward(self, inputs):      

        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.backbone.features(inputs)

        # Pooling and final linear layer
        x = self.backbone.avg_pool(x)
        # x = self.backbone.dropout(x)
        x = x.view(bs, -1)
        
        x1 = self.bn1_1(x)
        x2 = self.bn1_2(x)
        x3 = self.bn1_3(x)

        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)

        x1 = self.fc_graph1(x1)
        x2 = self.fc_vowel1(x2)
        x3 = self.fc_conso1(x3)

        x1 = self.bn2_1(x1)
        x2 = self.bn2_2(x2)
        x3 = self.bn2_3(x3)

        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)

        x1 = self.fc_graph2(x1)
        x2 = self.fc_vowel2(x2)
        x3 = self.fc_conso2(x3)

        x1 = self.bn3_1(x1)
        x2 = self.bn3_2(x2)
        x3 = self.bn3_3(x3)

        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)

        fc_graph = self.fc_graph3(x1)
        fc_vowel = self.fc_vowel3(x2)
        fc_conso = self.fc_conso3(x3)

        return fc_graph, fc_vowel, fc_conso

    
    
    
    
    
class BengaliGhostNet(torch.nn.Module):

    def __init__(self, pretrain=True):
        super().__init__()
        self.backbone = ghost_net(width_mult=1.0)

        in_features = 160
        
        self.avg_pool = nn.AvgPool2d((7,7), stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
        self.bn1_1 = nn.BatchNorm1d(num_features=in_features)
        self.bn1_2 = nn.BatchNorm1d(num_features=in_features)
        self.bn1_3 = nn.BatchNorm1d(num_features=in_features)

        self.bn2_1 = nn.BatchNorm1d(num_features=160)
        self.bn2_2 = nn.BatchNorm1d(num_features=160)
        self.bn2_3 = nn.BatchNorm1d(num_features=160)

        self.bn3_1 = nn.BatchNorm1d(num_features=160)
        self.bn3_2 = nn.BatchNorm1d(num_features=160)
        self.bn3_3 = nn.BatchNorm1d(num_features=160)

        self.fc_graph1 = torch.nn.Linear(in_features, 160)
        self.fc_vowel1 = torch.nn.Linear(in_features, 160)
        self.fc_conso1 = torch.nn.Linear(in_features, 160)

        self.fc_graph2 = torch.nn.Linear(160, 160)
        self.fc_vowel2 = torch.nn.Linear(160, 160)
        self.fc_conso2 = torch.nn.Linear(160, 160)

        self.fc_graph3 = torch.nn.Linear(160, 168)
        self.fc_vowel3 = torch.nn.Linear(160, 11)
        self.fc_conso3 = torch.nn.Linear(160, 7)


        # self._init_weights()

    def forward(self, inputs):      

        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.backbone.features(inputs)

        # Pooling and final linear layer
        x = self.avg_pool(x)
    
        x = x.view(bs, -1)




        x1 = self.bn1_1(x)
        x2 = self.bn1_2(x)
        x3 = self.bn1_3(x)

        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)

        x1 = self.fc_graph1(x1)
        x2 = self.fc_vowel1(x2)
        x3 = self.fc_conso1(x3)

        x1 = self.bn2_1(x1)
        x2 = self.bn2_2(x2)
        x3 = self.bn2_3(x3)

        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)

        x1 = self.fc_graph2(x1)
        x2 = self.fc_vowel2(x2)
        x3 = self.fc_conso2(x3)

        x1 = self.bn3_1(x1)
        x2 = self.bn3_2(x2)
        x3 = self.bn3_3(x3)

        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)

        fc_graph = self.fc_graph3(x1)
        fc_vowel = self.fc_vowel3(x2)
        fc_conso = self.fc_conso3(x3)

        return fc_graph, fc_vowel, fc_conso

