import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

from core.config import cfg
import nn as mynn
import utils.net as net_utils

from iou_cal import assign_iou, assign_label
#from featuredist_cal import assign_featuredist
from iou_cal import bbox_overlaps

def _get_highest_score_proposals(boxes, cls_prob, im_labels):
    num_classes = im_labels.shape[0]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in range(num_classes):
        if im_labels[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            max_index = np.argmax(cls_prob_tmp)

            gt_boxes = np.vstack((gt_boxes, boxes[max_index, :].reshape(1, -1)))
            gt_classes = np.vstack((gt_classes, (i+1) * np.ones((1, 1), dtype=np.int32)))
            gt_scores = np.vstack((gt_scores, cls_prob_tmp[max_index] * np.ones((1, 1), dtype=np.float32)))
            cls_prob[max_index, :] = 0

    proposals = {'gt_boxes': gt_boxes, 'gt_classes': gt_classes, 'gt_scores': gt_scores}
    return proposals

def _sample_rois(all_rois, proposals):
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = bbox_overlaps(np.ascontiguousarray(all_rois, dtype=np.float),
                             np.ascontiguousarray(gt_boxes, dtype=np.float))
    gt_assigment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assigment, 0]
    cls_loss_weights = gt_scores[gt_assigment, 0]

    fg_inds = np.where(max_overlaps >= 0.5)[0]
    bg_inds = np.where(max_overlaps < 0.5)[0]

    labels[bg_inds] = 0
    return labels, cls_loss_weights

class WSDDN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, label, y):

        loss = (-label * torch.log(y) - (1.-label) * torch.log(1.-y)).sum()
        ctx.save_for_backward(label, y)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        label, y = ctx.saved_tensors

        grad_input = (y - label) / (y.shape[0] * y * (1.-y))
        grad_input = grad_input * grad_output

        return None, grad_input


class OICR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rois_npy, cls_prob, img_label, cls_refine):

        proposals = _get_highest_score_proposals(rois_npy, cls_prob, img_label)
        labels, cls_loss_weights = _sample_rois(rois_npy, proposals)

        newlabels = np.eye(21)[labels]
        cls_loss_weights_reshape = np.reshape(cls_loss_weights, (cls_loss_weights.shape[0], 1))

        newlabels_tensor = Variable(torch.from_numpy(newlabels)).cuda().float()
        cls_loss_weights_tensor = Variable(torch.from_numpy(cls_loss_weights_reshape)).cuda()

        newlabels = newlabels * cls_loss_weights_reshape
        newlabels = Variable(torch.from_numpy(newlabels)).cuda().float()
        count = torch.clamp(torch.sum(newlabels > 1e-12).float(), min=1.)
        loss = torch.sum(-newlabels * torch.log(cls_refine)) / count

        ctx.save_for_backward(newlabels_tensor, cls_loss_weights_tensor, count , cls_refine)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        labels, cls_loss_weights, count, cls_refine = ctx.saved_tensors

        grad_input = cls_refine.clone()
        grad_input -= labels
        grad_input *= cls_loss_weights

        grad_input = grad_input * grad_output / count

        return None, None, None, grad_input


class oicr_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.cls_score0 = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES-1)
        self.cls_score1 = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES-1)


        self.cls_refine1 = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        self.cls_refine2 = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        self.cls_refine3 = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score0.weight, std=0.01)
        init.constant_(self.cls_score0.bias, 0)
        init.normal_(self.cls_score1.weight, std=0.01)
        init.constant_(self.cls_score1.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)
        init.normal_(self.cls_refine1.weight, std=0.01)
        init.constant_(self.cls_refine1.bias, 0)
        init.normal_(self.cls_refine2.weight, std=0.01)
        init.constant_(self.cls_refine2.bias, 0)
        init.normal_(self.cls_refine3.weight, std=0.01)
        init.constant_(self.cls_refine3.bias, 0)


    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score0.weight': 'cls_score0_w',
            'cls_score0.bias': 'cls_score0_b',
            'cls_score1.weight': 'cls_score1_w',
            'cls_score1.bias': 'cls_score1_b',
            'bbox_pred.weight': 'bbox_pred_w',
            'bbox_pred.bias': 'bbox_pred_b',
            'cls_refine1.weight': 'cls_refine1_w',
            'cls_refine1.bias': 'cls_refine1_b',
            'cls_refine2.weight': 'cls_refine2_w',
            'cls_refine2.bias': 'cls_refine2_b',
            'cls_refine3.weight': 'cls_refine3_w',
            'cls_refine3.bias': 'cls_refine3_b'

        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)

        cls_score0 = self.cls_score0(x)
        cls_score0 = F.softmax(cls_score0, dim=1)
        cls_score1 = self.cls_score1(x)
        cls_score1 = F.softmax(cls_score1, dim=0)

        bbox_mul = cls_score0 * cls_score1

        cls_refine1 = self.cls_refine1(x)
        cls_refine1 = F.softmax(cls_refine1, dim=1)
        cls_refine2 = self.cls_refine2(x)
        cls_refine2 = F.softmax(cls_refine2, dim=1)
        cls_refine3 = self.cls_refine3(x)
        cls_refine3 = F.softmax(cls_refine3, dim=1)


        if self.training:
            return bbox_mul, cls_refine1, cls_refine2, cls_refine3, 0.
        else:
            return cls_refine1 + cls_refine2 + cls_refine3



def oicr_losses(rois, bbox_mul, label_int32, cls_refine1, cls_refine2, cls_refine3, bbox_pred):

    y = bbox_mul.sum(dim=0)
    y = torch.clamp(y, 1e-10, 1-1e-10)

    device_id = y.get_device()
    label = label_int32.long()[:, None]
    zeros = torch.zeros(label.shape[0], cfg.MODEL.NUM_CLASSES).cuda(device_id)
    label = zeros.scatter_(1, label, 1) # one_hot
    
    label, _ = label.max(dim=0)
    label = label[1:]
    img_label = label.cpu().numpy().astype(np.float)

    wsddn = WSDDN.apply
    cls_loss = wsddn(label, y)

    rois_npy = rois.cpu().numpy()[:, 1:]

    oicr1 = OICR.apply
    oicr2 = OICR.apply
    oicr3 = OICR.apply

    refine_loss1 = oicr1(rois_npy, bbox_mul.detach().cpu().numpy(), img_label, cls_refine1)
    refine_loss2 = oicr1(rois_npy, cls_refine1[:, 1:].detach().cpu().numpy(), img_label, cls_refine2)
    refine_loss3 = oicr1(rois_npy, cls_refine2[:, 1:].detach().cpu().numpy(), img_label, cls_refine3)

    return cls_loss, refine_loss1, refine_loss2, refine_loss3, 0.


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

class roi_2mlp_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        #self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        self.dim_out = hidden_dim = 4096

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.5)

        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.dropout1(F.relu(self.fc1(x.view(batch_size, -1)), inplace=True))
        x = self.dropout2(F.relu(self.fc2(x), inplace=True))

        return x

