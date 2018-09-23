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

def caliou(np.ndarray[DTYPE_t, ndim=1] bbox1, np.ndarray[DTYPE_t, ndim=1] bbox2):

    cdef DTYPE_t area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    cdef DTYPE_t area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
        
    cdef DTYPE_t xx1 = max(bbox1[0], bbox2[0])
    cdef DTYPE_t yy1 = max(bbox1[1], bbox2[1])
    cdef DTYPE_t xx2 = min(bbox1[2], bbox2[2])
    cdef DTYPE_t yy2 = min(bbox1[3], bbox2[3])

    cdef DTYPE_t w = max(0., xx2 - xx1 + 1)
    cdef DTYPE_t h = max(0., yy2 - yy1 + 1)

    cdef DTYPE_t inter = w * h
    cdef DTYPE_t iou = inter / (area1 + area2 - inter)
    return iou

def assign_iou(np.ndarray[DTYPE_t, ndim=2] bbox_r, np.ndarray[DTYPE_t, ndim=2] bbox_c, np.ndarray[DTYPE_t, ndim=2] ious):
    cdef int i, j, k
    cdef int dim_r = bbox_r.shape[0]
    cdef int dim_c = bbox_c.shape[0]
    for i in range(dim_r):
        for j in range(dim_c):
            iou = caliou(bbox_r[i], bbox_c[j])
            ious[i, j] = iou > 0.6

def assign_label(np.ndarray[DTYPE_t, ndim=1] img_label, np.ndarray[DTYPE_t, ndim=2] labels, np.ndarray[DTYPE_t, ndim=2] ious):
    cdef int dim_r = labels.shape[0]
    cdef int dim_c = ious.shape[1]
    cdef int i, j, k
    for i in range(dim_r):
        for j in range(dim_c):
            if ious[i, j] > 0.5 and img_label[j] == 1.:
               labels[i, j + 1] = 1
