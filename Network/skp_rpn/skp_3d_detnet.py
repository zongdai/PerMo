import torch
import torch.utils.data
from torch import nn
import torchvision
from torch.nn import functional as F
from torch.jit.annotations import List, Optional, Dict, Tuple
from torch import Tensor
from .skp_rpn import AnchorGenerator, StructureKeypointsRegionProposalNetwork, StructureKeypointsRPNHead

class SKP3DDetNet(nn.Module):
    def __init__(
        self,
        min_z=0,
        max_z=35,
        min_x=-20,
        max_x=20,
        step=5,
        fg_dis=8
    ):
        super(SKP3DDetNet, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False, num_classes=512)
        anchor_generator = AnchorGenerator(step=step, min_z=min_z, max_z=max_z, min_x=min_x, max_x=max_x)
        rpn_head = StructureKeypointsRPNHead(512, ((max_x-min_x)*(max_z-min_z))//(step*step))
        self.skp_rpn = StructureKeypointsRegionProposalNetwork(min_x, max_x, min_z, max_z, step, rpn_head, fg_dis, 5, 16, 0.5, 20, 20, 0.8)
    
    def forward(self, skp, gt_position=None, gt_size=None, gt_anchor_vec=None):
        feature = self.backbone(skp)
        feature = F.relu(feature)
        losses, pre_scores, pre_positions, pre_size = self.skp_rpn(feature, gt_position, gt_size, gt_anchor_vec) 
        return losses, pre_scores, pre_positions, pre_size