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

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def caldist(np.ndarray[DTYPE_t, ndim=1] feature1, np.ndarray[DTYPE_t, ndim=1] feature2):
    diff = feature1 - feature2
    summ = np.sum(diff ** 2)
    return summ


def assign_dist(np.ndarray[DTYPE_t, ndim=2] feature_r, np.ndarray[DTYPE_t, ndim=2] feature_c, np.ndarray[DTYPE_t, ndim=2] dists):
    cdef int i, j
    cdef int dim_r = feature_r.shape[0]
    cdef int dim_c = feature_c.shape[0]

    for i in range(dim_r):
        for j in range(dim_c):
            dists[i, j] = caldist(feature_r[i], feature_c[j])

