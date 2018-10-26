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
import utils.net as net_utils
import utils.boxes as box_utils
#from sklearn.cluster import KMeans
from kmeans import KMeans

def _get_top_ranking_proposals(probs):
    if probs.shape[0] < 3:
        index = np.array([np.argmax(probs)])
        return 
    #kmeans = KMeans(n_clusters=3, random_state=2).fit(probs)
    #high_score_label = np.argmax(kmeans.cluster_centers_)
    #index = np.where(kmeans.labels_ == high_score_label)[0]
    centers, labels = KMeans(probs.astype(np.float))
    high_score_label = np.argmax(centers)
    index = np.where(labels == high_score_label)[0] 

    if len(index) == 0:
        index = np.array([np.argmax(probs)])
    return index

def _build_graph(boxes, iou_threshold):
    overlaps = bbox_overlaps(np.ascontiguousarray(boxes, dtype=np.float),
                             np.ascontiguousarray(boxes, dtype=np.float))
    return (overlaps > iou_threshold).astype(np.float32)

def _get_graph_centers(boxes, cls_prob, im_labels):
    num_classes = im_labels.shape[0]
    im_labels_tmp = im_labels.copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in range(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            idxs = np.where(cls_prob_tmp >= 0)[0]
            idxs_tmp = _get_top_ranking_proposals(cls_prob_tmp[idxs].reshape(-1,))
            idxs = idxs[idxs_tmp]
            boxes_tmp = boxes[idxs, :].copy()
            cls_prob_tmp = cls_prob_tmp[idxs]

            graph = _build_graph(boxes_tmp, 0.4)

            keep_idxs = []
            gt_scores_tmp = []
            count = cls_prob_tmp.size
            while True:
                order = np.sum(graph, axis=1).argsort()[::-1]
                tmp = order[0]
                keep_idxs.append(tmp)
                inds = np.where(graph[tmp, :] > 0)[0]
                gt_scores_tmp.append(np.max(cls_prob_tmp[inds]))

                graph[:, inds] = 0
                graph[inds, :] = 0
                count = count - len(inds)
                if count <= 5:
                    break

            gt_boxes_tmp = boxes_tmp[keep_idxs, :].copy()
            gt_scores_tmp = np.array(gt_scores_tmp).copy()

            keep_idxs_new = np.argsort(gt_scores_tmp)\
                [-1:(-1-min(len(gt_scores_tmp), 5)):-1]


            gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp[keep_idxs_new, :]))
            gt_scores = np.vstack((gt_scores, gt_scores_tmp[keep_idxs_new].reshape(-1, 1)))
            gt_classes = np.vstack((gt_classes, (i+1)*np.ones((len(keep_idxs_new), 1), dtype=np.int32)))

            cls_prob = np.delete(cls_prob.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)
            boxes = np.delete(boxes.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)

    proposals = {'gt_boxes': gt_boxes, 'gt_classes': gt_classes, 'gt_scores': gt_scores}
    return proposals

def _get_proposal_clusters(all_rois, proposals, im_labels, cls_prob, debug=False):
    num_classes = im_labels.shape[0]
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']

    overlaps = bbox_overlaps(np.ascontiguousarray(all_rois, dtype=np.float),
                             np.ascontiguousarray(gt_boxes, dtype=np.float))
    gt_assigment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assigment, 0]
    cls_loss_weights = gt_scores[gt_assigment, 0]

    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

    labels[bg_inds] = 0
    gt_assigment[bg_inds] = -1

    refine_loss = np.zeros(21, dtype=np.float32)
    refine_loss[0] = -np.sum(cls_loss_weights[bg_inds] * np.log(cls_prob[bg_inds, 0]))

    img_cls_loss_weights = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_probs = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_labels = np.zeros(gt_boxes.shape[0], dtype=np.int32)
    pc_count = np.zeros(gt_boxes.shape[0], dtype=np.float32) 

    for i in range(gt_boxes.shape[0]):
        po_index = np.where(gt_assigment == i)[0]
        if len(po_index) == 0:
            continue
        img_cls_loss_weights[i] = np.sum(cls_loss_weights[po_index])
        pc_labels[i] = gt_labels[i, 0]
        pc_count[i] = len(po_index)
        pc_probs[i] = np.average(cls_prob[po_index, pc_labels[i]])
        refine_loss[gt_labels[i, 0]] = -img_cls_loss_weights[i] * np.log(pc_probs[i])

    return refine_loss / cls_prob.shape[0], labels, cls_loss_weights, gt_assigment, pc_labels, pc_probs, pc_count, img_cls_loss_weights

class PCL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rois_npy, cls_prob, img_label, cls_refine):

        proposals = _get_graph_centers(rois_npy, cls_prob.detach().cpu().numpy(), img_label)
        loss, labels, cls_loss_weights, gt_assigment, pc_labels, pc_probs, pc_count, img_cls_loss_weights = _get_proposal_clusters(rois_npy, proposals, img_label, cls_refine.detach().cpu().numpy())

        loss = Variable(torch.from_numpy(loss)).cuda()
        labels = Variable(torch.from_numpy(labels)).cuda().long()
        cls_loss_weights = Variable(torch.from_numpy(cls_loss_weights)).cuda()
        gt_assigment = Variable(torch.from_numpy(gt_assigment)).cuda()
        pc_labels = Variable(torch.from_numpy(pc_labels)).cuda()
        pc_probs = Variable(torch.from_numpy(pc_probs)).cuda()
        pc_count = Variable(torch.from_numpy(pc_count)).cuda()
        img_cls_loss_weights = Variable(torch.from_numpy(img_cls_loss_weights)).cuda()
        gt_boxes = Variable(torch.from_numpy(proposals['gt_boxes'])).cuda()

        ctx.save_for_backward(labels, cls_loss_weights, gt_assigment, pc_labels, pc_probs, pc_count, img_cls_loss_weights, cls_refine, gt_boxes)

        return loss.float().sum()

    @staticmethod
    def backward(ctx, grad_output):

        labels, cls_loss_weights, gt_assigment, pc_labels, pc_probs, pc_count, img_cls_loss_weights, cls_refine, gt_boxes = ctx.saved_tensors

        grad_input = torch.zeros(cls_refine.shape, dtype=torch.float32).cuda()

        gt_assigment = gt_assigment.cpu().numpy()
        labels_npy = labels.cpu().numpy()

        bg_idx = np.where(labels_npy==0)[0]
        grad_input[bg_idx, 0] = -cls_loss_weights[bg_idx] / cls_refine[bg_idx, 0]
        for i in range(gt_boxes.shape[0]):
            fg_idx = np.where(gt_assigment==i)[0]
            if len(fg_idx) == 0:
                continue
            grad_input[fg_idx, labels[fg_idx]] = -img_cls_loss_weights[i] / (pc_count[i] *  pc_probs[i])

        grad_input = grad_input * grad_output / labels_npy.shape[0]

        return None, None, None, grad_input

class pcl_outputs(nn.Module):
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
            #x = cls_refine1 + cls_refine2 + cls_refine3
            return cls_refine1 + cls_refine2 + cls_refine3, bbox_pred



def pcl_losses(rois, bbox_mul, label_int32, cls_refine1, cls_refine2, cls_refine3, bbox_pred):

    y = bbox_mul.sum(dim=0)

    device_id = y.get_device()
    label = label_int32.long()[:, None]
    zeros = torch.zeros(label.shape[0], cfg.MODEL.NUM_CLASSES).cuda(device_id)
    label = zeros.scatter_(1, label, 1) # one_hot
    
    label, _ = label.max(dim=0)
    label = label[1:]
    img_label = label.cpu().numpy().astype(np.float)

    y = torch.clamp(y, 1e-6, 1-1e-6)
    cls_loss = -label * torch.log(y) - (1-label) * torch.log(1 - y)
    cls_loss = cls_loss.sum()


    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        reg_num = cfg.MODEL.NUM_CLASSES
    else:
        reg_num = 2

    pcl1 = PCL.apply
    pcl2 = PCL.apply
    pcl3 = PCL.apply

    rois_npy = rois.cpu().numpy()[:, 1:]
    cls_refine1 = torch.clamp(cls_refine1, 1e-6, 1-1e-6)
    cls_refine2 = torch.clamp(cls_refine2, 1e-6, 1-1e-6)
    cls_refine3 = torch.clamp(cls_refine3, 1e-6, 1-1e-6)

    refine_loss1 = pcl1(rois_npy, bbox_mul, img_label, cls_refine1)
    refine_loss2 = pcl1(rois_npy, cls_refine1[:, 1:], img_label, cls_refine2)
    refine_loss3 = pcl1(rois_npy, cls_refine2[:, 1:], img_label, cls_refine3)

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

