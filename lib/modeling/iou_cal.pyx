# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------
"""
Proposal Operator transform anchor coordinates into ROI coordinates with prediction results on
classification probability and bounding box prediction results, and image size and scale information.
"""

import numpy as np
cimport numpy as np

np.import_array()
DTYPE = np.float
ctypedef np.float_t DTYPE_t

#def caliou(np.ndarray[DTYPE_t, ndim=1] bbox1, np.ndarray[DTYPE_t, ndim=1] bbox2):
#
#    cdef DTYPE_t area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
#    cdef DTYPE_t area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
#        
#    cdef DTYPE_t xx1 = max(bbox1[0], bbox2[0])
#    cdef DTYPE_t yy1 = max(bbox1[1], bbox2[1])
#    cdef DTYPE_t xx2 = min(bbox1[2], bbox2[2])
#    cdef DTYPE_t yy2 = min(bbox1[3], bbox2[3])
#
#    cdef DTYPE_t w = max(0., xx2 - xx1 + 1)
#    cdef DTYPE_t h = max(0., yy2 - yy1 + 1)
#
#    cdef DTYPE_t inter = w * h
#    cdef DTYPE_t iou = inter / (area1 + area2 - inter)
#    return iou

def assign_iou(np.ndarray[DTYPE_t, ndim=2] bbox_r, np.ndarray[DTYPE_t, ndim=2] bbox_c, np.ndarray[DTYPE_t, ndim=2] ious):
    cdef int i, j, k
    cdef int dim_r = bbox_r.shape[0]
    cdef int dim_c = bbox_c.shape[0]
    cdef DTYPE_t iw, ih, area_r, iou, ua
    for i in range(dim_r):

        area_r = (
            (bbox_r[i, 2] - bbox_r[i, 0] + 1) * 
            (bbox_r[i, 3] - bbox_r[i, 1] + 1)
        )

        for j in range(dim_c):
            iou = 0.
            iw = (
                min(bbox_c[j, 2], bbox_r[i, 2]) -
                max(bbox_c[j, 0], bbox_r[i, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(bbox_c[j, 3], bbox_r[i, 3]) - 
                    max(bbox_c[j, 1], bbox_r[i, 1]) + 1
                )
                if ih > 0:
                    ua = (bbox_c[j, 2] - bbox_c[j, 0] + 1) * (bbox_c[j, 3] - bbox_c[j, 1] + 1) + area_r - iw * ih
                    iou = iw * ih / ua

            #iou = caliou(bbox_r[i], bbox_c[j])
            ious[i, j] = iou

def assign_label(np.ndarray[DTYPE_t, ndim=1] img_label, np.ndarray[DTYPE_t, ndim=2] labels, np.ndarray[DTYPE_t, ndim=2] ious, np.ndarray[DTYPE_t, ndim=1] maxprob):
    cdef int dim_r = labels.shape[0]
    cdef int dim_c = ious.shape[1]
    cdef int i, j, k, maxidx
    for i in range(dim_r):
        maxiou = -1.
        maxidx = 0
        for j in range(dim_c):
            if ious[i, j] >= maxiou and img_label[j] == 1.:
               maxiou = ious[i, j]
               maxidx = j

        if maxiou > 0.5:
            labels[i, maxidx + 1] = maxprob[maxidx]
        else:
            labels[i, 0] = maxprob[maxidx]

def bbox_overlaps(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps
