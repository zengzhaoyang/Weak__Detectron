import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils

class wsddn_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        self.bbox_pred = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'bbox_pred.weight': 'bbox_pred_w',
            'bbox_pred.bias': 'bbox_pred_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = self.cls_score(x)
        cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)
        bbox_pred = F.softmax(bbox_pred, dim=0)

        bbox_mul = cls_score * bbox_pred

        if self.training:
            y = bbox_mul.sum(dim=0)
            return y
        else:
            return bbox_mul

def wsddn_losses(y, label_int32):
    device_id = y.get_device()
    zeros = torch.zeros(label.shape[0], cfg.MODEL.NUM_CLASSES).cuda(device_id)
    label = label[1:]
    y = y[1:]

    cls_loss = -label * torch.log(y + 1e-6) - (1-label) * torch.log(1 - y + 1e-6)
    cls_loss = cls_loss.sum()

    return cls_loss

