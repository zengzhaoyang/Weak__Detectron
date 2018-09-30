import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
import nn as mynn
import utils.net as net_utils
from utils.resnet_weights_helper import convert_state_dict



def VGG16_conv5_body():
    return VGG_conv5_body()

class VGG_conv5_body(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        #self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
  
        self.dim_out = 512
        self.spatial_scale = 1. / 8
        self._init_modules()

    def _init_modules(self):
        freeze_params(self.conv1_1)
        freeze_params(self.conv1_2)
        freeze_params(self.conv2_1)
        freeze_params(self.conv2_2)
        pass


    def detectron_weight_mapping(self):
        mapping_to_detectron = {
            'conv1_1.weight': 'conv1_1_w',
            'conv1_1.bias': 'conv1_1_b',
            'conv1_2.weight': 'conv1_2_w',
            'conv1_2.bias': 'conv1_2_b',
            'conv2_1.weight': 'conv2_1_w',
            'conv2_1.bias': 'conv2_1_b',
            'conv2_2.weight': 'conv2_2_w',
            'conv2_2.bias': 'conv2_2_b',
            'conv3_1.weight': 'conv3_1_w',
            'conv3_1.bias': 'conv3_1_b',
            'conv3_2.weight': 'conv3_2_w',
            'conv3_2.bias': 'conv3_2_b',
            'conv3_3.weight': 'conv3_3_w',
            'conv3_3.bias': 'conv3_3_b',
            'conv4_1.weight': 'conv4_1_w',
            'conv4_1.bias': 'conv4_1_b',
            'conv4_2.weight': 'conv4_2_w',
            'conv4_2.bias': 'conv4_2_b',
            'conv4_3.weight': 'conv4_3_w',
            'conv4_3.bias': 'conv4_3_b',
            'conv5_1.weight': 'conv5_1_w',
            'conv5_1.bias': 'conv5_1_b',
            'conv5_2.weight': 'conv5_2_w',
            'conv5_2.bias': 'conv5_2_b',
            'conv5_3.weight': 'conv5_3_w',
            'conv5_3.bias': 'conv5_3_b',
        }
        orphan_in_detectron = ['fc1000_w', 'fc1000_b']
        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        # Override
        self.training = mode
        self.conv1_1.train(mode)
        self.conv1_2.train(mode)
        self.conv2_1.train(mode)
        self.conv2_2.train(mode)
        self.conv3_1.train(mode)
        self.conv3_2.train(mode)
        self.conv3_3.train(mode)
        self.conv4_1.train(mode)
        self.conv4_2.train(mode)
        self.conv4_3.train(mode)
        self.conv5_1.train(mode)
        self.conv5_2.train(mode)
        self.conv5_3.train(mode)

    def forward(self, x):
        x = F.relu(self.conv1_1(x), inplace=True)
        x = F.relu(self.conv1_2(x), inplace=True)
        x = self.pool1(x)
        x = F.relu(self.conv2_1(x), inplace=True)
        x = F.relu(self.conv2_2(x), inplace=True)
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x), inplace=True)
        x = F.relu(self.conv3_2(x), inplace=True)
        x = F.relu(self.conv3_3(x), inplace=True)
        x = self.pool3(x)
        x = F.relu(self.conv4_1(x), inplace=True)
        x = F.relu(self.conv4_2(x), inplace=True)
        x = F.relu(self.conv4_3(x), inplace=True)
        #x = self.pool4(x)
        x = F.relu(self.conv5_1(x), inplace=True)
        x = F.relu(self.conv5_2(x), inplace=True)
        x = F.relu(self.conv5_3(x), inplace=True)
        return x
       
class VGG_roi_fc_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = 4096
        self.roi_size = 7

        #roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * self.roi_size**2, hidden_dim)
        self.dropout1 = nn.Dropout(0.5, inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.5, inplace=True)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def train(self, mode=True):
        # Override
        self.training = mode
        self.dropout1.train(mode)
        self.dropout2.train(mode)

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=self.roi_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.dropout2(x)

        return x


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False
