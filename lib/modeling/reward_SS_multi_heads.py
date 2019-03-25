import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import cv2

from core.config import cfg
import nn as mynn
import utils.net as net_utils

#from iou_cal import assign_iou, assign_label
#from featuredist_cal import assign_featuredist
from iou_cal import bbox_overlaps
import utils.net as net_utils
import utils.boxes as box_utils

def cal_reward(image, superpixels, im_info, roi, reward_type, theta=0):

    if reward_type == 'CC':
        #image = LAB(image)
        roi = roi / im_info
        roi = roi.astype(np.int32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        h, w = image.shape[:2] 
        roiw = roi[:, 2] - roi[:, 0] + 1
        roih = roi[:, 3] - roi[:, 1] + 1
        roix = (roi[:, 2] + roi[:, 0]) * 0.5
        roiy = (roi[:, 3] + roi[:, 1]) * 0.5
        neww = roiw * theta
        newh = roih * theta
        newroi = roi.copy()
        newroi[:, 0] = np.clip(roix - neww * 0.5, 0, w-1)
        newroi[:, 1] = np.clip(roiy - newh * 0.5, 0, h-1)
        newroi[:, 2] = np.clip(roix + neww * 0.5, 0, w-1)
        newroi[:, 3] = np.clip(roiy + newh * 0.5, 0, h-1)

        area = roiw * roih * 1.0
        newarea = (newroi[:, 2] - newroi[:, 0] + 1) * (newroi[:, 3] - newroi[:, 1] + 1)
         
        reward = []
        eps = 1e-6
        for i in range(roi.shape[0]):
            crop_image = [image[roi[i][1]:roi[i][3], roi[i][0]:roi[i][2], :]]
            hist_L = cv2.calcHist(crop_image, [0], None, [8], [0., 255.]) / area[i]
            hist_A = cv2.calcHist(crop_image, [1], None, [16], [0., 255.]) / area[i]
            hist_B = cv2.calcHist(crop_image, [2], None, [16], [0., 255.]) / area[i]

            crop_image_new = [image[newroi[i][1]:newroi[i][3], newroi[i][0]:newroi[i][2], :]]
            hist2_L = cv2.calcHist(crop_image_new, [0], None, [8], [0., 255.]) / newarea[i]
            hist2_A = cv2.calcHist(crop_image_new, [1], None, [16], [0., 255.]) / newarea[i]
            hist2_B = cv2.calcHist(crop_image_new, [2], None, [16], [0., 255.]) / newarea[i]

            hist = np.vstack((hist_L, hist_A, hist_B))
            hist2 = np.vstack((hist2_L, hist2_A, hist2_B))

            dist = ((hist-hist2)**2 / (hist+hist2 + eps)).sum() / 1.
            reward.append(dist)

        return np.array(reward).reshape((-1, 1))

    elif reward_type == 'ED':

        roi = roi / im_info
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2] 
        image = cv2.resize(image, (200, 200))
        ratioh = 200. / h
        ratiow = 200. / w

        image = cv2.GaussianBlur(image, (3, 3), 0)
        canny = cv2.Canny(image, 50, 150) / 255

        roi[:,0] *= ratiow
        roi[:,1] *= ratioh
        roi[:,2] *= ratiow
        roi[:,3] *= ratioh

        roi = roi.astype(np.int32)

        roiw = roi[:, 2] - roi[:, 0] + 1
        roih = roi[:, 3] - roi[:, 1] + 1
        roix = (roi[:, 2] + roi[:, 0]) * 0.5
        roiy = (roi[:, 3] + roi[:, 1]) * 0.5
        neww = roiw / theta
        newh = roih / theta
        newroi = roi.copy()
        newroi[:, 0] = np.clip(roix - neww * 0.5, 0, w-1)
        newroi[:, 1] = np.clip(roiy - newh * 0.5, 0, h-1)
        newroi[:, 2] = np.clip(roix + neww * 0.5, 0, w-1)
        newroi[:, 3] = np.clip(roiy + newh * 0.5, 0, h-1)

        perimeter = ((newroi[:, 2] - newroi[:, 0] + 1) + (newroi[:, 3] - newroi[:, 1] + 1)) * 2

        reward = []
        for i in range(roi.shape[0]):
            crop_image = canny[newroi[i][1]:newroi[i][3], newroi[i][0]:newroi[i][2]]
            reward.append(crop_image.sum() / perimeter[i])

        return np.array(reward).reshape((-1, 1))

    elif reward_type == 'MS':

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        scales = [16, 24, 32, 48, 64]
        tot = len(scales)
        mags = []
        ratios = []
        eps = 1e-6
        for s in scales:
            resize_img = cv2.resize(image, (s, s))
            c = cv2.dft(np.float32(resize_img), flags=cv2.DFT_COMPLEX_OUTPUT)
            mag = np.sqrt(c[:, :, 0]**2 + c[:, :, 1]**2)
            spectralResidual = np.exp(np.log(mag+eps) - cv2.boxFilter(np.log(mag+eps), -1, (3,3)))
            
            c[:, :, 0] = c[:, :, 0] * spectralResidual / (mag + eps)
            c[:, :, 1] = c[:, :, 1] * spectralResidual / (mag + eps)
            c = cv2.dft(c, flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))
            mag = c[:, :, 0]**2 + c[:, :, 1]**2
            cv2.normalize(cv2.GaussianBlur(mag, (9,9),3,3), mag, 0., 1., cv2.NORM_MINMAX)

            #mag = (mag > theta).astype(np.uint8)
            mags.append(mag)
            ratioh = s * 1.0 / h
            ratiow = s * 1.0 / w
            ratios.append((ratioh, ratiow))
        roi = roi / im_info
        roi_scales = []
        for i in range(tot):
            r = roi.copy()
            r[:,0] *= ratios[i][1]
            r[:,1] *= ratios[i][0]
            r[:,2] *= ratios[i][1]
            r[:,3] *= ratios[i][0]
            r = r.astype(np.int32)
            roi_scales.append(r)

        reward = []
        for i in range(roi.shape[0]):
            sum_reward = 0.
            for j in range(tot):
                xmin = roi_scales[j][i, 0]
                ymin = roi_scales[j][i, 1]
                xmax = roi_scales[j][i, 2]
                ymax = roi_scales[j][i, 3]
                area = (xmax - xmin + 1) * (ymax - ymin + 1)
                sum_reward += mags[j][ymin:ymax, xmin:xmax].sum() / area
            sum_reward /= tot
            reward.append(sum_reward) 

        return np.array(reward).reshape((-1, 1))

    elif reward_type == 'SS':
        roi = roi / im_info
        roi = roi.astype(np.int32)
        h, w = image.shape[:2] 
        area = (roi[:, 2] - roi[:, 0] + 1) * (roi[:, 3] - roi[:, 1] + 1)

        superpixels = superpixels.astype(np.int64)
        superpixels = superpixels[:, :, 0] * 256 * 256 + superpixels[:, :, 1] * 256 + superpixels[:, :, 2]

        sp_flatten = superpixels.flatten()
        sp_unique, sp_idx = np.unique(sp_flatten, return_inverse=True)
        tot = sp_idx.max() + 1
        sp_bin = np.bincount(sp_idx, minlength=tot)

        sp_idx = sp_idx.reshape((h, w))

        reward = []

        for i in range(roi.shape[0]):
            crop_sp = sp_idx[roi[i,1]:roi[i,3], roi[i,0]:roi[i,2]]
            crop_sp_flatten = crop_sp.flatten()

            crop_bin = np.bincount(crop_sp_flatten, minlength=tot)
            crop_unique = np.unique(crop_sp_flatten)

            sp_bin_sel = sp_bin[crop_unique]
            crop_bin_sel = crop_bin[crop_unique]

            crop_diff = sp_bin_sel - crop_bin_sel
            crop_diff = np.minimum(crop_bin_sel, crop_diff)

            summ = crop_diff.sum() / area[i]

            reward.append(1 - summ)
        return np.array(reward).reshape(-1)


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

def _sample_rois(images, superpixels, im_info, all_rois, proposals, with_bbox=False):
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

    reward = cal_reward(images, superpixels, im_info, all_rois[fg_inds, :], 'SS')

    cls_loss_weights[fg_inds] = reward

    if with_bbox:
        gt_boxes = gt_boxes[gt_assigment, :]

        bbox_target_data = box_utils.bbox_transform_inv(all_rois, gt_boxes, cfg.MODEL.BBOX_REG_WEIGHTS)
        bbox_targets = np.zeros((all_rois.shape[0], 4 * 2))
        bbox_targets[fg_inds, 4:8] = bbox_target_data[fg_inds, :]

        bbox_inside_weights = np.zeros((all_rois.shape[0], 4 * 2))

        #weights = np.reshape(cls_loss_weights, (-1, 1))
        #bbox_inside_weights[fg_inds, 4:8] = weights[fg_inds]
        bbox_inside_weights[fg_inds, 4:8] = reward.reshape((-1, 1))

        bbox_outside_weights = (bbox_inside_weights > 0).astype(np.float32)

        return labels, cls_loss_weights, bbox_targets, bbox_inside_weights, bbox_outside_weights

    else:
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
    def forward(ctx, images, superpixels, im_info, rois_npy, cls_prob, img_label, cls_refine, with_bbox=False, bboxs=[]):


        proposals = _get_highest_score_proposals(rois_npy, cls_prob, img_label)
        if with_bbox:
            labels, cls_loss_weights, bbox_targets, bbox_inside_weights, bbox_outside_weights = _sample_rois(images, superpixels, im_info, rois_npy, proposals, with_bbox)
            bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda().float()
            bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda().float()
            bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda().float()
            bboxs.append(bbox_targets)
            bboxs.append(bbox_inside_weights)
            bboxs.append(bbox_outside_weights)

        else:
            labels, cls_loss_weights = _sample_rois(images, superpixels, im_info, rois_npy, proposals)

        newlabels = np.eye(21)[labels]
        cls_loss_weights_reshape = np.reshape(cls_loss_weights, (-1, 1))
        newlabels_tensor = Variable(torch.from_numpy(newlabels)).cuda().float()
        cls_loss_weights_tensor = Variable(torch.from_numpy(cls_loss_weights_reshape)).cuda()

        newlabels = newlabels * cls_loss_weights_reshape
        newlabels = Variable(torch.from_numpy(newlabels)).cuda().float()
        count = torch.clamp(torch.sum(newlabels > 1e-12).float(), min=1.)
        cls_refine_softmax = F.softmax(cls_refine, dim=1)
        cls_refine_softmax = torch.clamp(cls_refine_softmax, 1e-6, 1-1e-6)
        loss = torch.sum(-newlabels * torch.log(cls_refine_softmax)) / count

        ctx.save_for_backward(newlabels_tensor, cls_loss_weights_tensor, count, cls_refine_softmax)
        return loss

    @staticmethod
    def backward(ctx, grad_output):

        labels, cls_loss_weights, count, cls_refine = ctx.saved_tensors

        grad_input = cls_refine.clone()
        grad_input -= labels
        grad_input *= cls_loss_weights
        grad_input = grad_input * grad_output / count

        return None, None, None, None, None, None, grad_input, None, None



class reward_outputs(nn.Module):
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
        cls_refine2 = self.cls_refine2(x)
        cls_refine3 = self.cls_refine3(x)



        if self.training:
            return bbox_mul, cls_refine1, cls_refine2, cls_refine3, bbox_pred
        else:
            #x = cls_refine1 + cls_refine2 + cls_refine3

            cls_refine1 = F.softmax(cls_refine1, dim=1)
            cls_refine2 = F.softmax(cls_refine2, dim=1)
            cls_refine3 = F.softmax(cls_refine3, dim=1)
            return cls_refine1 + cls_refine2 + cls_refine3, bbox_pred



def reward_losses(rois, images, superpixels, im_info, bbox_mul, label_int32, cls_refine1, cls_refine2, cls_refine3, bbox_pred):


    y = bbox_mul.sum(dim=0)
    y = torch.clamp(y, 1e-6, 1-1e-6)

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

    cls_refine1_softmax = F.softmax(cls_refine1, dim=1)
    cls_refine2_softmax = F.softmax(cls_refine2, dim=1)

    bboxs = []

    refine_loss1 = oicr1(images, superpixels, im_info, rois_npy, bbox_mul.detach().cpu().numpy(), img_label, cls_refine1, False)
    refine_loss2 = oicr2(images, superpixels, im_info, rois_npy, cls_refine1_softmax[:, 1:].detach().cpu().numpy(), img_label, cls_refine2, False)
    refine_loss3 = oicr3(images, superpixels, im_info, rois_npy, cls_refine2_softmax[:, 1:].detach().cpu().numpy(), img_label, cls_refine3, True, bboxs)


    bbox_targets = bboxs[0]
    bbox_inside_weights = bboxs[1]
    bbox_outside_weights = bboxs[2]
    loss_bbox = net_utils.smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    return cls_loss, refine_loss1, refine_loss2, refine_loss3, loss_bbox


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
