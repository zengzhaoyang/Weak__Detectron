from torch import nn
from torch.nn import init
import torch.nn.functional as F

from core.config import cfg
from modeling.generate_ssw_proposals import GenerateSSWProposalsOp
from modeling.generate_proposal_labels import GenerateProposalLabelsOp
import utils.net as net_utils


# ---------------------------------------------------------------------------- #
# RPN and Faster R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def generic_ssw_outputs():
    """Add RPN outputs (objectness classification and bounding box regression)
    to an RPN model. Abstracts away the use of FPN.
    """
    return single_scale_ssw_outputs()


class single_scale_ssw_outputs(nn.Module):
    """Add RPN outputs to a single scale model (i.e., no FPN)."""
    def __init__(self):
        super().__init__()

        self.SSW_GenerateProposals = GenerateSSWProposalsOp()

    def forward(self, roidb=None):
        """
        x: feature maps from the backbone network. (Variable)
        im_info: (CPU Variable)
        roidb: (list of ndarray)
        """

        ssw_roi = self.SSW_GenerateProposals(roidb)

        return_dict = {
            'rois': ssw_roi
        }

        return return_dict
