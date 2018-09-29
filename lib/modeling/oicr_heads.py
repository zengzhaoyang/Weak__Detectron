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

class oicr_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES-1)
        self.bbox_pred = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES-1)

        self.cls_refine1 = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        self.cls_refine2 = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        self.cls_refine3 = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)
        init.normal_(self.cls_refine1.weight, std=0.001)
        init.constant_(self.cls_refine1.bias, 0)
        init.normal_(self.cls_refine2.weight, std=0.001)
        init.constant_(self.cls_refine2.bias, 0)
        init.normal_(self.cls_refine3.weight, std=0.001)
        init.constant_(self.cls_refine3.bias, 0)


    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
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
        cls_score = self.cls_score(x)
        cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)
        bbox_pred = F.softmax(bbox_pred, dim=0)

        bbox_mul = cls_score * bbox_pred

        cls_refine1 = self.cls_refine1(x)
        cls_refine1 = F.softmax(cls_refine1, dim=1)
        cls_refine2 = self.cls_refine2(x)
        cls_refine2 = F.softmax(cls_refine2, dim=1)
        cls_refine3 = self.cls_refine3(x)
        cls_refine3 = F.softmax(cls_refine3, dim=1)



        if self.training:
            return bbox_mul, cls_refine1, cls_refine2, cls_refine3
        else:
            return (cls_refine1 + cls_refine2 + cls_refine3) / 3.
            #return cls_refine1



def oicr_losses(rois, bbox_mul, label_int32, cls_refine1, cls_refine2, cls_refine3):

    y = bbox_mul.sum(dim=0)

    device_id = y.get_device()
    label = label_int32.long()[:, None]
    zeros = torch.zeros(label.shape[0], cfg.MODEL.NUM_CLASSES).cuda(device_id)
    label = zeros.scatter_(1, label, 1) # one_hot
    
    label, _ = label.max(dim=0)
    label = label[1:]
    img_label = label.cpu().numpy().astype(np.float)

    cls_loss = -label * torch.log(y + 1e-6) - (1-label) * torch.log(1 - y + 1e-6)
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
        newlabels = np.eye(21)[labels]
        cls_loss_weights = np.reshape(cls_loss_weights, (cls_loss_weights.shape[0], 1))
        newlabels = newlabels * cls_loss_weights
        return newlabels

    rois_npy = rois.cpu().numpy()[:, 1:]
    proposals1 = _get_highest_score_proposals(rois_npy, bbox_mul.detach().cpu().numpy(), img_label)
    label1 = _sample_rois(rois_npy, proposals1)
    #if device_id == 0:
    #    print('refine1', bbox_mul.max().detach().cpu().numpy())
    label1 = Variable(torch.from_numpy(label1)).cuda().float() # r * 21
    refine_loss1 = torch.sum(torch.sum(-label1 * torch.log(cls_refine1 + 1e-6), dim=1), dim=0) / (torch.sum(label1 > 1e-12).float() + 1e-8)

    proposals2 = _get_highest_score_proposals(rois_npy, cls_refine1[:, 1:].detach().cpu().numpy(), img_label)
    label2 = _sample_rois(rois_npy, proposals2)
    #if device_id == 0:
    #    print('refine2', cls_refine1[:, 1:].max().detach().cpu().numpy(), cls_refine1.argmax(dim=1).sum().detach().cpu().numpy())
    label2 = Variable(torch.from_numpy(label2)).cuda().float() # r * 21
    refine_loss2 = torch.sum(torch.sum(-label2 * torch.log(cls_refine2 + 1e-6), dim=1), dim=0) / (torch.sum(label2 > 1e-12).float() + 1e-8)

    proposals3 = _get_highest_score_proposals(rois_npy, cls_refine2[:, 1:].detach().cpu().numpy(), img_label)
    label3 = _sample_rois(rois_npy, proposals3)
    #if device_id == 0:
    #    print('refine3', cls_refine2[:, 1:].max().detach().cpu().numpy(), cls_refine2.argmax(dim=1).sum().detach().cpu().numpy())
    label3 = Variable(torch.from_numpy(label3)).cuda().float() # r * 21
    refine_loss3 = torch.sum(torch.sum(-label3 * torch.log(cls_refine3 + 1e-6), dim=1), dim=0) / (torch.sum(label3 > 1e-12).float() + 1e-8)

    #idx = torch.argmax(bbox_mul, dim=0).cpu().numpy().astype(np.int) # 20
    #maxprob, _ = torch.max(bbox_mul, dim=0) #20
    #maxprob = maxprob.detach().cpu().numpy().astype(np.float)
    #if device_id == 1:
    #    print('bbox_mul', maxprob.sum(), maxprob.max(), maxprob.min())

    #rois = rois.cpu().numpy().astype(np.float)[:, 1:]
    #rois_take = rois[idx, :]
    #print(rois_take)
    #dim_r = rois.shape[0]
    #dim_c = rois_take.shape[0]

    #ious = np.zeros((dim_r, dim_c), dtype=np.float) #20
    #assign_iou(rois, rois_take, ious)
    #label1 = np.zeros((dim_r, cfg.MODEL.NUM_CLASSES), dtype=np.float) # r * 21
    #assign_label(img_label, label1, ious, maxprob)
    #label1 = Variable(torch.from_numpy(label1)).cuda().float() # r * 21
    #refine_loss1 = torch.sum(torch.sum(-label1 * torch.log(cls_refine1 + 1e-8), dim=1), dim=0) / torch.sum(label1 > 1e-12).float()

    #idx = torch.argmax(cls_refine1[:, 1:], dim=0).cpu().numpy().astype(np.int32)
    #maxprob, _ = torch.max(cls_refine1[:, 1:], dim=0)
    #maxprob = maxprob.detach().cpu().numpy().astype(np.float)
    #if device_id == 1:
    #    print((label1 != 0).sum(), (label1[:, 1:] != 0).sum())
    #    print('refine1', maxprob.sum(), maxprob.max(), maxprob.min(), cls_refine1.argmax(dim=1).sum().detach().cpu().numpy())
    #rois_take = rois[idx, :]
    #ious = np.zeros((dim_r, dim_c), dtype=np.float)
    #assign_iou(rois, rois_take, ious)
    #label2 = np.zeros((dim_r, cfg.MODEL.NUM_CLASSES), dtype=np.float)
    #assign_label(img_label, label2, ious, maxprob)
    #label2 = Variable(torch.from_numpy(label2)).cuda().float()
    #refine_loss2 = torch.sum(torch.sum(-label2 * torch.log(cls_refine2 + 1e-8), dim=1), dim=0) / torch.sum(label2 > 1e-12).float()


    #idx = torch.argmax(cls_refine2[:, 1:], dim=0).cpu().numpy().astype(np.int32)
    #maxprob, _ = torch.max(cls_refine2[:, 1:], dim=0)
    #maxprob = maxprob.detach().cpu().numpy().astype(np.float)
    #if device_id == 1:
    #    print('refine2', maxprob.sum(), maxprob.max(), maxprob.min(), cls_refine2.argmax(dim=1).sum().detach().cpu().numpy())
    #rois_take = rois[idx, :]
    #ious = np.zeros((dim_r, dim_c), dtype=np.float)
    #assign_iou(rois, rois_take, ious)
    #label3 = np.zeros((dim_r, cfg.MODEL.NUM_CLASSES), dtype=np.float)
    #assign_label(img_label, label3, ious, maxprob)
    #label3 = Variable(torch.from_numpy(label3)).cuda().float()
    #refine_loss3 = torch.sum(torch.sum(-label3 * torch.log(cls_refine3 + 1e-8), dim=1), dim=0) / torch.sum(label3 > 1e-12).float()

    return cls_loss, refine_loss1, refine_loss2, refine_loss3


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


class roi_Xconv1fc_head(nn.Module):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*2): 'head_conv%d_w' % (i+1),
                'convs.%d.bias' % (i*2): 'head_conv%d_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

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
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x


class roi_Xconv1fc_gn_head(nn.Module):
    """Add a X conv + 1fc head, with GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                             eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*3): 'head_conv%d_w' % (i+1),
                'convs.%d.weight' % (i*3+1): 'head_conv%d_gn_s' % (i+1),
                'convs.%d.bias' % (i*3+1): 'head_conv%d_gn_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

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
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x
