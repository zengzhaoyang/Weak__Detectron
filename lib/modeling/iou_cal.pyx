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

def find_inout(
        np.ndarray[DTYPE_t, ndim=2] cls_prob,
        np.ndarray[DTYPE_t, ndim=2] boxes,
        unsigned int label,
        np.ndarray[np.int_t, ndim=1] marks):

    cdef unsigned int N = cls_prob.shape[0]
    cdef DTYPE_t max_prob = -1.
    cdef unsigned int i, j
    cdef unsigned int max_index
    for i in range(N):
        if cls_prob[i, label] > max_prob:
            max_prob = cls_prob[i, label]        
            max_index = i

    cdef DTYPE_t in_max_prob = -1.
    cdef DTYPE_t out_max_prob = -1.
    cdef unsigned int in_max_index = 999999
    cdef unsigned int out_max_index = 999999
    cdef DTYPE_t iou = 0.
    for i in range(N):
        if i != max_index and marks[i] != 1:
            iou = 0
            box_area = (
                (boxes[i, 2] - boxes[i, 0] + 1) *
                (boxes[i, 3] - boxes[i, 1] + 1)
            )
            iw = (
                min(boxes[i, 2], boxes[max_index, 2]) -
                max(boxes[i, 0], boxes[max_index, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[i, 3], boxes[max_index, 3]) - 
                    max(boxes[i, 1], boxes[max_index, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[max_index, 2] - boxes[max_index, 0] + 1) *
                        (boxes[max_index, 3] - boxes[max_index, 1] + 1) +
                        box_area - iw * ih
                    )
                    iou = iw * ih / ua

            if iou < 0.3:
                # inside
                if boxes[i, 2] >= boxes[max_index, 2] \
                    and boxes[i, 0] <= boxes[max_index, 0] \
                    and boxes[i, 3] >= boxes[max_index, 3] \
                    and boxes[i, 1] <= boxes[max_index, 1]:
                    if cls_prob[i, label] > in_max_prob:
                        in_max_prob = cls_prob[i, label]
                        in_max_index = i 
                # outside
                if boxes[i, 2] <= boxes[max_index, 2] \
                    and boxes[i, 0] >= boxes[max_index, 0] \
                    and boxes[i, 3] <= boxes[max_index, 3] \
                    and boxes[i, 1] >= boxes[max_index, 1]:
                    if cls_prob[i, label] > in_max_prob:
                        out_max_prob = cls_prob[i, label]
                        out_max_index = i 

    indexs = [max_index]
    if in_max_index != 999999:
        indexs.append(in_max_index)
    if out_max_index != 999999:
        indexs.append(out_max_index)
    return indexs
