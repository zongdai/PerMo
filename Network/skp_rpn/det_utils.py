import math

import torch
from torch.jit.annotations import List, Tuple
from torch import Tensor
import torchvision

# TODO: https://github.com/pytorch/pytorch/issues/26727
def zeros_like(tensor, dtype):
    # type: (Tensor, int) -> Tensor
    return torch.zeros_like(tensor, dtype=dtype, layout=tensor.layout,
                            device=tensor.device, pin_memory=tensor.is_pinned())

class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float)
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor])
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = torch.tensor(1, dtype=torch.uint8)
            neg_idx_per_image_mask[neg_idx_per_image] = torch.tensor(1, dtype=torch.uint8)

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


def encode_positions(reference_positions, proposals):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
    """

    # perform some unpacking to make it JIT-fusion friendly

    proposals_x = proposals[:, 0].unsqueeze(1)
    proposals_y = proposals[:, 1].unsqueeze(1)
    proposals_z = proposals[:, 2].unsqueeze(1)
    proposals_r = proposals[:, 3].unsqueeze(1)
    proposals_w = proposals[:, 4].unsqueeze(1)
    proposals_h = proposals[:, 5].unsqueeze(1)
    proposals_l = proposals[:, 6].unsqueeze(1)

    reference_x = reference_positions[:, 0].unsqueeze(1)
    reference_y = reference_positions[:, 1].unsqueeze(1)
    reference_z = reference_positions[:, 2].unsqueeze(1)
    reference_r = reference_positions[:, 3].unsqueeze(1)
    reference_w = reference_positions[:, 4].unsqueeze(1)
    reference_h = reference_positions[:, 5].unsqueeze(1)
    reference_l = reference_positions[:, 6].unsqueeze(1)

    # implementation starts here
    # ex_widths = proposals_x2 - proposals_x1
    # ex_heights = proposals_y2 - proposals_y1
    # ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    # ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    # gt_widths = reference_boxes_x2 - reference_boxes_x1
    # gt_heights = reference_boxes_y2 - reference_boxes_y1
    # gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    # gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    # targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    # targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    # targets_dw = ww * torch.log(gt_widths / ex_widths)
    # targets_dh = wh * torch.log(gt_heights / ex_heights)
    target_dx = proposals_x - reference_x
    target_dy = proposals_y - reference_y
    target_dz = proposals_z - reference_z
    target_dr = proposals_r - reference_r
    target_dw = proposals_w - reference_w
    target_dh = proposals_h - reference_h
    target_dl = proposals_l - reference_l

    position_targets = torch.cat((target_dx, target_dy, target_dz, target_dr), dim=1)
    size_targets = torch.cat((target_dw, target_dh, target_dl), dim=1)
    return position_targets, size_targets


class PositionCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, min_x, max_x, min_z, max_z, step, device='cuda'):
        """
        Arguments:
            position_xform_clip (float)
        """
        self.step = step
        self.min_x = min_x
        self.max_x = max_x
        self.min_z = min_z
        self.max_z = max_z
        self.anchor_positions = []
        for x_i in range(self.min_x, self.max_x, self.step):
                for z_i in range(self.min_z, self.max_z, self.step):
                    self.anchor_positions.append(torch.tensor([x_i, 0, z_i, 0], device=device))
        self.anchor_positions = torch.cat(self.anchor_positions).reshape(-1, 4)
    def encode(self, gt_position):
        # type: (List[dic])
        position_off_sets = []
        for pos in gt_position:
            for x_i in range(self.min_x, self.max_x, self.step):
                for z_i in range(self.min_z, self.max_z, self.step):
                    position_off_sets.append([pos[0]-x_i, pos[1], pos[2]-z_i, pos[3]])
        position_off_sets = torch.tensor(position_off_sets, device=gt_position.device)
        return position_off_sets

   

    def decode(self, objectness, pre_position_offset, pre_size_offset, top_n=1):
        '''
        Arguments:
            objectness (Tensor) , shape = [bs, num_anchor]
            pred_position_deltas (Tensor) , shape = [bs*num_anchor, 4]
            pred_size_deltas (Tensor), shape = [bs, 3]
        '''
        num_anchor = objectness.shape[1]
        values, indices = torch.sort(objectness, dim=1, descending=True) # indices.shape = [bs, num_anchor]
        # print(values)
        res_scores = []
        res_postions = []
        res_sizes = [size for size in pre_size_offset]
        for bs_index, indice in enumerate(indices):
            scores = objectness[bs_index, indice[0:top_n]]
            # print(scores)
            matched_pre_position_offset = pre_position_offset[indice[0:top_n]+(bs_index*num_anchor), :] # top_n * 4
            # print(matched_pre_position_offset)
            # print(self.anchor_positions.shape)
            anchor_position = self.anchor_positions[indice[0:top_n], :]
            pre_postion = matched_pre_position_offset + anchor_position
            res_scores.append(scores)
            res_postions.append(pre_postion)
        return res_scores, res_postions, res_sizes