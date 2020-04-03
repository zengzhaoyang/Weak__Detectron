import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

from core.config import cfg
import nn as mynn
import utils.net as net_utils

from iou_cal import bbox_overlaps
import utils.net as net_utils
import utils.boxes as box_utils

class oicr_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.cls_score0 = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES-1)
        self.cls_score1 = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES-1)

        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            self.reg_num = cfg.MODEL.NUM_CLASSES
        else:
            self.reg_num = 2 
        self.bbox_pred = nn.Linear(dim_in, 4 * self.reg_num)

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

        bbox_pred = self.bbox_pred(x)

        bbox_mul = cls_score0 * cls_score1

        cls_refine1 = self.cls_refine1(x)
        cls_refine1 = F.softmax(cls_refine1, dim=1)
        cls_refine2 = self.cls_refine2(x)
        cls_refine2 = F.softmax(cls_refine2, dim=1)
        cls_refine3 = self.cls_refine3(x)
        cls_refine3 = F.softmax(cls_refine3, dim=1)

        if self.training:
            return bbox_mul, cls_refine1, cls_refine2, cls_refine3, bbox_pred
        else:
            return cls_refine1 + cls_refine2 + cls_refine3, bbox_pred


def oicr_losses(rois, bbox_mul, label_int32, cls_refine1, cls_refine2, cls_refine3, bbox_pred):

    y = bbox_mul.sum(dim=0)

    device_id = y.get_device()
    #label = label_int32.long()[:, None]
    #zeros = torch.zeros(label.shape[0], cfg.MODEL.NUM_CLASSES).cuda(device_id)
    #label = zeros.scatter_(1, label, 1) # one_hot
    #label, _ = label.max(dim=0)
    label = label[1:]
    img_label = label.cpu().numpy().astype(np.float)

    y = torch.clamp(y, 1e-6, 1-1e-6)
    cls_loss = -label * torch.log(y) - (1-label) * torch.log(1 - y)
    cls_loss = cls_loss.sum()

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

    def _sample_rois(all_rois, proposals, with_bbox=False):
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
        newlabels = np.eye(21)[labels]
        cls_loss_weights = np.reshape(cls_loss_weights, (cls_loss_weights.shape[0], 1))
        newlabels = newlabels * cls_loss_weights

        if with_bbox:
            gt_boxes = gt_boxes[gt_assigment, :]

            bbox_target_data = box_utils.bbox_transform_inv(all_rois, gt_boxes, cfg.MODEL.BBOX_REG_WEIGHTS)
            bbox_targets = np.zeros((all_rois.shape[0], 4 * 2))
            bbox_targets[fg_inds, 4:8] = bbox_target_data[fg_inds, :]

            bbox_inside_weights = np.zeros((all_rois.shape[0], 4 * 2))
            bbox_inside_weights[fg_inds, 4:8] = cls_loss_weights[fg_inds]
            bbox_outside_weights = (bbox_inside_weights > 0).astype(np.float32)

            return newlabels, gt_boxes, bbox_targets, bbox_inside_weights, bbox_outside_weights

        else:
            return newlabels

    rois_npy = rois.cpu().numpy()[:, 1:]
    proposals1 = _get_highest_score_proposals(rois_npy, bbox_mul.detach().cpu().numpy(), img_label)
    label1 = _sample_rois(rois_npy, proposals1)
    label1 = Variable(torch.from_numpy(label1)).cuda().float()
    cls_refine1 = torch.clamp(cls_refine1, 1e-6, 1-1e-6)


    refine_loss1 = torch.sum(torch.sum(-label1 * torch.log(cls_refine1), dim=1), dim=0) / torch.clamp(torch.sum(label1 > 1e-12).float(), 1., 9999999999.)

    proposals2 = _get_highest_score_proposals(rois_npy, cls_refine1[:, 1:].detach().cpu().numpy(), img_label)
    label2 = _sample_rois(rois_npy, proposals2)
    label2 = Variable(torch.from_numpy(label2)).cuda().float()
    cls_refine2 = torch.clamp(cls_refine2, 1e-6, 1-1e-6)
    refine_loss2 = torch.sum(torch.sum(-label2 * torch.log(cls_refine2), dim=1), dim=0) / torch.clamp(torch.sum(label2 > 1e-12).float(), 1., 999999999.)

    proposals3 = _get_highest_score_proposals(rois_npy, cls_refine2[:, 1:].detach().cpu().numpy(), img_label)
    label3, _, bbox_targets, bbox_inside_weights, bbox_outside_weights = _sample_rois(rois_npy, proposals3, with_bbox=True)
    label3 = Variable(torch.from_numpy(label3)).cuda().float()
    cls_refine3 = torch.clamp(cls_refine3, 1e-6, 1-1e-6)

    refine_loss3 = torch.sum(torch.sum(-label3 * torch.log(cls_refine3), dim=1), dim=0) / torch.clamp(torch.sum(label3 > 1e-12).float(), 1., 999999999.)


    bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda().float()
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda().float()
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda().float()
    bbox_loss = 30. * net_utils.smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    return cls_loss, refine_loss1, refine_loss2, refine_loss3, bbox_loss
