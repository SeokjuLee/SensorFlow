from __future__ import division
import torch
import random
import numpy as np
from scipy.misc import imresize
import pdb

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics, poses):
        for t in self.transforms:
            images, intrinsics, poses = t(images, intrinsics, poses)
        return images, intrinsics, poses


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics, poses):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, intrinsics, poses


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, intrinsics, poses):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors, intrinsics, poses


def HorizontalFlipPose(pose):
    '''
        Input: 4x4 np array
        (x)-----> (X-axis)
         |
         |
         v
        (Y-axis)
    '''
    output_pose = np.copy(pose)

    """ Rotation part """
    output_pose[0,1] = -output_pose[0,1]
    output_pose[0,2] = -output_pose[0,2]
    output_pose[1,0] = -output_pose[1,0]
    output_pose[2,0] = -output_pose[2,0]

    """ Translation part """
    output_pose[0,3] = -output_pose[0,3]

    return output_pose


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics, poses):
        assert intrinsics is not None
        assert poses is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
            output_poses = [HorizontalFlipPose(pose) for pose in poses]
            # pdb.set_trace()
        else:
            output_images = images
            output_intrinsics = intrinsics
            output_poses = poses
        return output_images, output_intrinsics, output_poses


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, intrinsics, poses):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h,
                             offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        return cropped_images, output_intrinsics, poses
