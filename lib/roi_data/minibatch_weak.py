import numpy as np
import cv2

from core.config import cfg
import utils.blob as blob_utils
import roi_data.wsddn


def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    blob_names += roi_data.wsddn.get_wsddn_blob_names(
        is_training=is_training
    )
    return blob_names


def get_minibatch(roidb, preloads=None):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_scales, ori_ims = _get_image_blob(roidb, preloads)
    blobs['data'] = im_blob
    blobs['ori_data'] = ori_ims
    blobs['im_info'] = im_scales
    valid = roi_data.wsddn.add_wsddn_blobs(blobs, im_scales, roidb)
    return blobs, valid


def _get_image_blob(roidb, preloads=None):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    ori_ims = []
    im_scales = []
    for i in range(num_images):
        if preloads is None:
            im = cv2.imread(roidb[i]['image'])
        else:
            im = preloads[roidb[i]['image'].split('/')[-1].split('.')[0]].copy()
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])

        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        ori_ims.append(im)
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales, ori_ims
