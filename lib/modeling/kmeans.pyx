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

def KMeans(np.ndarray[DTYPE_t, ndim=1] probs):

    cdef unsigned int N = probs.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] labels = np.zeros((N, ), dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] old_labels = np.zeros((N, ), dtype=np.int)
    cdef unsigned int i, j, n
    cdef unsigned int flag

    cdef np.ndarray[DTYPE_t, ndim=1] centers = np.random.random_sample((3,))
    cdef np.ndarray[np.int_t, ndim=1] nums = np.zeros((3, ), dtype=np.int)

    for n in range(300):
        flag = 0
        for i in range(N):
            dis0 = abs(probs[i] - centers[0]) 
            dis1 = abs(probs[i] - centers[1]) 
            dis2 = abs(probs[i] - centers[2]) 
            if dis0 <= dis1 and dis0 <= dis2:
                labels[i] = 0
            elif dis1 <= dis0 and dis1 <= dis2:
                labels[i] = 1
            elif dis2 <= dis0 and dis2 <= dis1:
                labels[i] = 2
            if labels[i] != old_labels[i]:
                flag = 1
        if flag == 0:
            break
        old_labels = labels
        centers[0] = 0.
        centers[1] = 0.
        centers[2] = 0. 
        nums[0] = 0
        nums[1] = 0
        nums[2] = 0
        for i in range(N):
            centers[labels[i]] += probs[i]
            nums[labels[i]] += 1

        centers[0] /= nums[0]
        centers[1] /= nums[1]
        centers[2] /= nums[2]
        
    return centers, labels

