import torch
import torch.utils.data
from torch import nn
import torchvision
from torch.nn import functional as F
from torch.jit.annotations import List, Optional, Dict, Tuple
from torch import Tensor
from . import det_utils
import torch.nn.functional as F



class AnchorGenerator(nn.Module):
    

    def __init__(
        self,
        step=2,
        min_z=0,
        max_z=50,
        min_x=-25,
        max_x=25
    ):
        super(AnchorGenerator, self).__init__()

        self.step = step
        self.min_x = min_x
        self.max_x = max_x
        self.min_z = min_z
        self.max_z = max_z
        self.cell_anchors = None
        self._cache = {}
    
    def generate_anchors(self, device):
        # type: (List[int], List[float], int, Device)  # noqa: F821
        base_anchors = []
        for x in range(self.min_x, self.max_x, self.step):
            for z in range(self.min_z, self.max_z, self.step): 
                base_anchors.append(torch.tensor([x, 1.5, z, 0, 1.5, 1.5, 4]))
        base_anchors = torch.stack(base_anchors, dim=0)
        base_anchors.to(device)
        return base_anchors.round()

    def forward(self, batch_size, device):
        anchors = []
        for i in range(batch_size):
            anchors_per_structure = self.generate_anchors(device)
            anchors.append(anchors_per_structure)
        # anchors = [torch.cat(anchors_per_structure) for anchors_per_structure in anchors]
        # print(len(anchors)) 16
        # print(anchors[0].shape) 7 * 625
        return anchors

class StructureKeypointsRPNHead(nn.Module):
    def __init__(self, inchannels, num_anchors):
        super(StructureKeypointsRPNHead, self).__init__()
        self.num_anchors = num_anchors
        # self.class_pred_group = nn.ModuleList([nn.Linear(inchannels, 1) for i in range(num_anchors)])
        self.anchor_pred = nn.Linear(inchannels, num_anchors)
        self.position_pred_group = nn.ModuleList([nn.Linear(inchannels, 4) for i in range(num_anchors)]) # x, y, z, r
        # self.size_pred_group = nn.ModuleList([nn.Linear(inchannels, 3) for i in range(num_anchors)]) # width, height, length
        self.size_pred = nn.Linear(inchannels, 3)
        
    def forward(self, input_feature):
        # type: (List[Tensor])
        t = F.relu(input_feature)
        logits = self.anchor_pred(t)
        position_reg = ([layer(t) for layer in self.position_pred_group])
        size_reg = self.size_pred(t)   
        position_reg_ = []     
        for i in range(input_feature.shape[0]):
            for j in range(self.num_anchors):
                position_reg_.append(position_reg[j][i, :].reshape(1,4))
        position_reg_ = torch.cat(position_reg_)
        # print(position_reg_.shape)
        return logits, position_reg_, size_reg

class StructureKeypointsRegionProposalNetwork(torch.nn.Module):
    def __init__(self,
                 min_x, max_x, min_z, max_z, step,
                 head,
                 #
                 fg_dis_thresh, bg_dis_thresh,
                 batch_size_per_image, positive_fraction,
                 #
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super(StructureKeypointsRegionProposalNetwork, self).__init__()
        self.step = step
        self.min_x = min_x
        self.max_x = max_x
        self.min_z = min_z
        self.max_z = max_z
        self.head = head
        self.fg_dis_thresh = fg_dis_thresh
        self.bg_dis_thresh = bg_dis_thresh
        self.position_coder = det_utils.PositionCoder(min_x, max_x, min_z, max_z, step)

        # self.proposal_matcher = det_utils.Matcher(
        #     fg_iou_thresh,
        #     bg_iou_thresh,
        #     allow_low_quality_matches=True,
        # )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]])
        labels = []
        matched_gt_target = []
        device = anchors[0].device
        labels_per_structure = torch.zeros((anchors[0].shape[0],), dtype=torch.float32, device=device)
        matched_gt_target_per_structure = torch.zeros(anchors[0].shape, dtype=torch.float32, device=device)
        for target_per_structure in targets:
            gt_pos_mat = (target_per_structure[0:3].reshape(1,3)).expand(anchors[0].shape[0], 3)
            dis_flag = torch.sqrt(torch.sum((gt_pos_mat-anchors[0][:,0:3])*(gt_pos_mat-anchors[0][:,0:3]), dim=1))
            # print(torch.argmin(dis_flag))
            labels_per_structure[dis_flag < self.fg_dis_thresh] = torch.tensor(1.0)
            labels_per_structure[dis_flag > self.bg_dis_thresh] = torch.tensor(0.0)
            labels_per_structure[(dis_flag < self.bg_dis_thresh) & (dis_flag > self.fg_dis_thresh)] = torch.tensor(-1.0)
            for i in range(anchors[0].shape[0]):
                matched_gt_target_per_structure[i, :] = target_per_structure 
            labels.append(labels_per_structure)
            matched_gt_target.append(matched_gt_target_per_structure)

        return labels, matched_gt_target
    



    def compute_loss(self, objectness, pred_position_deltas, pred_size_deltas, gt_anchor_vec, position_off_sets, gt_size):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor])
        """
        Arguments:
            objectness (Tensor) , shape = [bs, num_anchor]
            pred_position_deltas (Tensor) , shape = [bs*num_anchor, 4]
            pred_size_deltas (Tensor), shape = [bs, 3]

            gt_anchor_vec (Tensor) gt_anchor_vec.shape = [bs, num_anchor, 1]
            position_off_sets (Tensor) position_off_set.shape = [bs*num_anchor, 4]
            gt_size (Tensor) gt_size.shape = [bs, 3]
        Returns:
            objectness_loss (Tensor)
            position_loss (Tensor)
        print(objectness.shape)
        print(pred_position_deltas.shape)
        print(pred_size_deltas.shape)
        print(gt_anchor_vec.shape)
        print(position_off_sets.shape)
        print(gt_size.shape)
        print('................')
        """
        
        
        gt_anchor_vec = torch.squeeze(gt_anchor_vec) 
        pos_weight = torch.full([gt_anchor_vec.shape[1]], 30).to(objectness.device)

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness, gt_anchor_vec,pos_weight=pos_weight
        )
        
        gt_pos_mask = ((gt_anchor_vec.reshape(-1, 1)) > 0).expand(position_off_sets.shape)

        position_loss = F.l1_loss(
            pred_position_deltas[gt_pos_mask],
            position_off_sets[gt_pos_mask],
            reduction="sum",
        ) / (torch.sum(gt_pos_mask))

        size_loss = F.l1_loss(
            pred_size_deltas,
            gt_size,
            reduction="sum",
        ) / (objectness.shape[0])
        

        return objectness_loss, position_loss, size_loss

    def forward(self, features, gt_position=None, gt_size=None, gt_anchor_vec=None):
        
        objectness, position, size = self.head(features)
       
        ## evaluate 
        pre_scores, pre_positions, pre_size = self.position_coder.decode(objectness.detach(), position.detach(), size.detach())
        # print(pre_positions)
        # print(gt_position)
        ## 
        losses = {}
        if self.training:
            assert gt_position is not None
            # objectness = objectness.reshape(bs*num_anchors, 1)
            # position = position.reshape(bs*num_anchors, 4)
            # size = size.reshape(bs*num_anchors, 3)
            # labels, matched_gt_target = self.assign_targets_to_anchors(anchors, targets)
            position_off_sets = self.position_coder.encode(gt_position) # position_off_set.shape = [bs*num_anchors, 3]

            # gt_size = [t["states"] for t in targets]
            # gt_anchor_vec = [t["anchor_vec"] for t in targets]
            # gt_size = torch.cat(gt_size)
            # gt_anchor_vec = torch.cat(gt_anchor_vec)
            loss_objectness, loss_position, loss_size = self.compute_loss(
                objectness, position, size, gt_anchor_vec, position_off_sets, gt_size)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_position": loss_position,
                "loss_size": loss_size
            }
        
        return losses, pre_scores, pre_positions, pre_size