"""
Helper functions for converting resnet pretrained weights from other formats
"""
import os
import pickle

import torch

import nn as mynn
import re
from core.config import cfg


def load_pretrained_imagenet_weights(model):
    """Load pretrained weights
    Args:
        num_layers: 50 for res50 and so on.
        model: the generalized rcnnn module
    """
    _, ext = os.path.splitext(cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS)
    if ext == '.pkl':
        with open(cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS, 'rb') as fp:
            src_blobs = pickle.load(fp, encoding='latin1')
        if 'blobs' in src_blobs:
            src_blobs = src_blobs['blobs']
        pretrianed_state_dict = src_blobs
    else:
        weights_file = os.path.join(cfg.ROOT_DIR, cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS)
        pretrianed_state_dict = convert_state_dict(torch.load(weights_file))

        # Convert batchnorm weights


    model_state_dict = model.state_dict()

    #pattern = re.compile(r"conv\d_\d_[w|b]")
    pattern = [
        'conv1_1_w', 'conv1_1_b',
        'conv1_2_w', 'conv1_2_b',
        'conv2_1_w', 'conv2_1_b',
        'conv2_2_w', 'conv2_2_b',
        'conv3_1_w', 'conv3_1_b',
        'conv3_2_w', 'conv3_2_b',
        'conv3_3_w', 'conv3_3_b',
        'conv4_1_w', 'conv4_1_b',
        'conv4_2_w', 'conv4_2_b',
        'conv4_3_w', 'conv4_3_b',
        'conv5_1_w', 'conv5_1_b',
        'conv5_2_w', 'conv5_2_b',
        'conv5_3_w', 'conv5_3_b',
        'fc6_w', 'fc6_b',
        'fc7_w', 'fc7_b',
    ] 

    name_mapping, _ = model.detectron_weight_mapping

    for k, v in name_mapping.items():
        #print(k, v)
        if isinstance(v, str):  # maybe a str, None or True
            #if pattern.match(v):
            if v in pattern:
                pretrianed_key = v
                if ext == '.pkl':
                    model_state_dict[k].copy_(torch.Tensor(pretrianed_state_dict[v]))
                else:
                    model_state_dict[k].copy_(pretrianed_state_dict[pretrianed_key])


def convert_state_dict(src_dict):
    """Return the correct mapping of tensor name and value

    Mapping from the names of torchvision model to our resnet conv_body and box_head.
    """
    dst_dict = {}
    key_map = {
        'features.0.weight': 'conv1_1_w',
        'features.0.bias': 'conv1_1_b',
        'features.2.weight': 'conv1_2_w',
        'features.2.bias': 'conv1_2_b',
        'features.5.weight': 'conv2_1_w',
        'features.5.bias': 'conv2_1_b',
        'features.7.weight': 'conv2_2_w',
        'features.7.bias': 'conv2_2_b',
        'features.10.weight': 'conv3_1_w',
        'features.10.bias': 'conv3_1_b',
        'features.12.weight': 'conv3_2_w',
        'features.12.bias': 'conv3_2_b',
        'features.14.weight': 'conv3_3_w',
        'features.14.bias': 'conv3_3_b',
        'features.17.weight': 'conv4_1_w',
        'features.17.bias': 'conv4_1_b',
        'features.19.weight': 'conv4_2_w',
        'features.19.bias': 'conv4_2_b',
        'features.21.weight': 'conv4_3_w',
        'features.21.bias': 'conv4_3_b',
        'features.24.weight': 'conv5_1_w',
        'features.24.bias': 'conv5_1_b',
        'features.26.weight': 'conv5_2_w',
        'features.26.bias': 'conv5_2_b',
        'features.28.weight': 'conv5_3_w',
        'features.28.bias': 'conv5_3_b',
        'classifier.0.weight': 'fc6_w',
        'classifier.0.bias': 'fc6_b',
        'classifier.3.weight': 'fc7_w',
        'classifier.3.bias': 'fc7_b'
    }
    for k, v in src_dict.items():
        if k in key_map: 
            name = key_map[k]
            dst_dict[name] = v
    return dst_dict
