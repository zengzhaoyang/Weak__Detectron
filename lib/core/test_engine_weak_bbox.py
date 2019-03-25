# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml
import heapq

import torch

from core.config import cfg
# from core.rpn_generator import generate_rpn_on_dataset  #TODO: for rpn only case
# from core.rpn_generator import generate_rpn_on_range
from core.test_weak_bbox import im_detect_all
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_weak_bbox_builder as model_weak_builder
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils
import utils.boxes as box_utils
from utils.io import save_object
from utils.timer import Timer

logger = logging.getLogger(__name__)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    if cfg.MODEL.RPN_ONLY:
        raise NotImplementedError
        # child_func = generate_rpn_on_range
        # parent_func = generate_rpn_on_dataset
    else:
        # Generic case that handles all network types other than RPN-only nets
        # and RetinaNet
        child_func = test_net
        parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]

    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert is_parent or len(cfg.TEST.PROPOSAL_FILES) == 1, \
            'The child inference process can only work on a single proposal file'
        assert len(cfg.TEST.PROPOSAL_FILES) == len(cfg.TEST.DATASETS), \
            'If proposals are used, one proposal file must be specified for ' \
            'each dataset'
        proposal_file = cfg.TEST.PROPOSAL_FILES[index]
    else:
        proposal_file = None

    return dataset_name, proposal_file


def apply_nms(all_boxes, thresh):
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]

    for cls_ind in range(1, num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            if cfg.TEST.SOFT_NMS.ENABLED:
                nms_dets_tmp, _ = box_utils.soft_nms(
                    dets,
                    sigma=cfg.TEST.SOFT_NMS.SIGMA,
                    overlap_thresh=cfg.TEST.NMS,
                    score_thresh=0.0001,
                    method=cfg.TEST.SOFT_NMS.METHOD
                )
            else:


                keep = box_utils.nms(dets, thresh)
                if len(keep) == 0:
                    continue

                nms_dets_tmp = dets[keep, :]
            if cfg.TEST.BBOX_VOTE.ENABLED:
                nms_dets_tmp = box_utils.box_voting(
                    nms_dets_tmp,
                    dets,
                    cfg.TEST.BBOX_VOTE.VOTE_TH,
                    scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
                )

            #nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
            nms_boxes[cls_ind][im_ind] = nms_dets_tmp.copy()

    return nms_boxes

def run_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            all_results = {}
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = args.output_dir
                results = parent_func(
                    args,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing
                )
                all_results.update(results)

            return all_results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()
    if check_expected_results and is_parent:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)

    return all_results


def test_net_on_dataset(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""
    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb())
        all_boxes, all_segms, all_keyps = multi_gpu_test_net_on_dataset(
            args, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        all_boxes, all_segms, all_keyps = test_net(
            args, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    results = task_evaluation.evaluate_all(
        dataset, all_boxes, all_segms, all_keyps, output_dir
    )
    return results


def multi_gpu_test_net_on_dataset(
        args, dataset_name, proposal_file, num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset and proposal file (if any) via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
    if proposal_file:
        opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'detection', num_images, binary, output_dir,
        args.load_ckpt, args.load_detectron, opts
    )

    # Collate the results from each subprocess
    all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    for det_data in outputs:
        all_boxes_batch = det_data['all_boxes']
        all_segms_batch = det_data['all_segms']
        all_keyps_batch = det_data['all_keyps']
        for cls_idx in range(1, cfg.MODEL.NUM_CLASSES):
            all_boxes[cls_idx] += all_boxes_batch[cls_idx]
            all_segms[cls_idx] += all_segms_batch[cls_idx]
            all_keyps[cls_idx] += all_keyps_batch[cls_idx]
    det_file = os.path.join(output_dir, 'detections.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return all_boxes, all_segms, all_keyps


def test_net(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range
    )
    model = initialize_model_from_cfg(args, gpu_id=gpu_id)
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
    timers = defaultdict(Timer)

    thresh = -np.inf * np.ones(num_classes-1)
    top_scores = [[] for _ in range(num_classes-1)]
    max_per_set = 40 * num_images
    max_per_image = 100

    print(dataset_name)

    if 'test' in dataset_name:
        for i, entry in enumerate(roidb):
            im = cv2.imread(entry['image'])
            #cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(model, im, box_proposals, timers)
            box_proposals = entry['boxes'][entry['gt_classes'] == 0]
            scores, boxes = im_detect_all(model, im, box_proposals, timers)

            for j in range(0, num_classes-1):
                inds = np.where((scores[:, j] > thresh[j]))[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j*4:(j+1)*4]
                top_inds = np.argsort(-cls_scores)[:max_per_image]
                cls_scores = cls_scores[top_inds]
                cls_boxes = cls_boxes[top_inds, :]
                for val in cls_scores:
                    heapq.heappush(top_scores[j], val)
                if len(top_scores[j]) > max_per_set:
                    while len(top_scores[j]) > max_per_set:
                        heapq.heappop(top_scores[j])
                    thresh[j] = top_scores[j][0]

                all_boxes[j+1][i] = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        
            #extend_results(i, all_boxes, cls_boxes_i)

            if i % 10 == 0:  # Reduce log file size
                ave_total_time = np.sum([t.average_time for t in timers.values()])
                eta_seconds = ave_total_time * (num_images - i - 1)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                det_time = (
                    timers['im_detect_bbox'].average_time +
                    timers['im_detect_mask'].average_time +
                    timers['im_detect_keypoints'].average_time
                )
                misc_time = (
                    timers['misc_bbox'].average_time +
                    timers['misc_mask'].average_time +
                    timers['misc_keypoints'].average_time
                )
                logger.info(
                    (
                        'im_detect: range [{:d}, {:d}] of {:d}: '
                        '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                    ).format(
                        start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                        start_ind + num_images, det_time, misc_time, eta
                    )
                )
        
        for j in range(num_classes-1):
            for i in range(num_images):
                inds = np.where(all_boxes[j+1][i][:, -1] > thresh[j])[0]
                all_boxes[j+1][i] = all_boxes[j+1][i][inds, :]

        all_boxes = apply_nms(all_boxes, cfg.TEST.NMS)

        cfg_yaml = yaml.dump(cfg)
        if ind_range is not None:
            det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
        else:
            det_name = 'detections.pkl'
        det_file = os.path.join(output_dir, det_name)
        save_object(
            dict(
                all_boxes=all_boxes,
                all_segms=all_segms,
                all_keyps=all_keyps,
                cfg=cfg_yaml
            ), det_file
        )
        logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
        return all_boxes, all_segms, all_keyps
    else:
        for i, entry in enumerate(roidb):
            im = cv2.imread(entry['image'])
            #cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(model, im, box_proposals, timers)
            box_proposals = entry['boxes'][entry['gt_classes'] == 0]
            scores, boxes = im_detect_all(model, im, box_proposals, timers)

            for j in range(0, num_classes-1):
                index = np.argmax(scores[:, j])
                cls_boxes = boxes[index, j*4:(j+1)*4].reshape(1, -1)
                all_boxes[j+1][i] = np.hstack((cls_boxes, np.array([[scores[index, j]]]))).astype(np.float32, copy=False)
            #extend_results(i, all_boxes, cls_boxes_i)

            if i % 10 == 0:  # Reduce log file size
                ave_total_time = np.sum([t.average_time for t in timers.values()])
                eta_seconds = ave_total_time * (num_images - i - 1)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                det_time = (
                    timers['im_detect_bbox'].average_time +
                    timers['im_detect_mask'].average_time +
                    timers['im_detect_keypoints'].average_time
                )
                misc_time = (
                    timers['misc_bbox'].average_time +
                    timers['misc_mask'].average_time +
                    timers['misc_keypoints'].average_time
                )
                logger.info(
                    (
                        'im_detect: range [{:d}, {:d}] of {:d}: '
                        '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                    ).format(
                        start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                        start_ind + num_images, det_time, misc_time, eta
                    )
                )

        #gt_tmp = {
        #    'aeroplane': np.empty((0, 4), dtype=np.float32),
        #    'bicycle': np.empty((0, 4), dtype=np.float32),
        #    'bird': np.empty((0, 4), dtype=np.float32),
        #    'boat': np.empty((0, 4), dtype=np.float32),
        #    'bottle': np.empty((0, 4), dtype=np.float32),
        #    'bus': np.empty((0, 4), dtype=np.float32),
        #    'car': np.empty((0, 4), dtype=np.float32),
        #    'cat': np.empty((0, 4), dtype=np.float32),
        #    'chair': np.empty((0, 4), dtype=np.float32),
        #    'cow': np.empty((0, 4), dtype=np.float32),
        #    'diningtable': np.empty((0, 4), dtype=np.float32),
        #    'dog': np.empty((0, 4), dtype=np.float32),
        #    'horse': np.empty((0, 4), dtype=np.float32),
        #    'motorbike': np.empty((0, 4), dtype=np.float32),
        #    'person': np.empty((0, 4), dtype=np.float32),
        #    'pottedplant': np.empty((0, 4), dtype=np.float32),
        #    'sheep': np.empty((0, 4), dtype=np.float32),
        #    'sofa': np.empty((0, 4), dtype=np.float32),
        #    'train': np.empty((0, 4), dtype=np.float32),
        #    'tvmonitor': np.empty((0, 4), dtype=np.float32),
        #}
        #tmp_idx = np.where(roidb[i]['labels'][0][:20])[0]
        #
        #for j in xrange(len(tmp_idx)):
        #    idx_real = np.argmax(scores[:, tmp_idx[j]])
        #    gt_tmp[imdb.classes[tmp_idx[j]]] = np.array([boxes[idx_real, tmp_idx[j]*4+1],
        #                                                 boxes[idx_real, tmp_idx[j]*4],
        #                                                 boxes[idx_real, tmp_idx[j]*4+3],
        #                                                 boxes[idx_real, tmp_idx[j]*4+2]], dtype=np.float32)
        #    gt_tmp[imdb.classes[tmp_idx[j]]] += 1

        #gt[i] = {'gt': gt_tmp}

        cfg_yaml = yaml.dump(cfg)
        if ind_range is not None:
            det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
        else:
            det_name = 'detections.pkl'
        det_file = os.path.join(output_dir, det_name)
        save_object(
            dict(
                all_boxes=all_boxes,
                all_segms=all_segms,
                all_keyps=all_keyps,
                cfg=cfg_yaml
            ), det_file
        )
        logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
        return all_boxes, all_segms, all_keyps



def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = model_weak_builder.Generalized_RCNN()
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert proposal_file, 'No proposal file given'
        roidb = dataset.get_roidb(
            proposal_file=proposal_file,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )
    else:
        roidb = dataset.get_roidb()

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]
