import logging
import numpy as np

from torch import nn

from core.config import cfg
import utils.boxes as box_utils

logger = logging.getLogger(__name__)


class GenerateSSWProposalsOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, roidb):
        print(roidb)
        rois = np.zeros((2000, 5), dtype=np.float32)
        return rois

def _filter_boxes(boxes, min_size, im_info):
    """Only keep boxes with both sides >= min_size and center within the image.
  """
    # Scale min_size to match image scale
    min_size *= im_info[2]
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    x_ctr = boxes[:, 0] + ws / 2.
    y_ctr = boxes[:, 1] + hs / 2.
    keep = np.where((ws >= min_size) & (hs >= min_size) &
                    (x_ctr < im_info[1]) & (y_ctr < im_info[0]))[0]
    return keep
