import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmrnext.model.RAFT.modules.corr import CorrBlock
from cmrnext.model.RAFT.modules.extractor import BasicEncoder, SmallEncoder
from cmrnext.model.RAFT.modules.update import BasicUpdateBlock, SmallUpdateBlock
from cmrnext.model.RAFT.utils.utils import coords_grid, upflow2, upflow8
from torch.cuda import amp


class RAFT_cmrnext(nn.Module):

    def __init__(self,
                 args,
                 use_reflectance=False,
                 with_uncertainty=False,
                 fourier_levels=-1,
                 unc_type="DER",
                 unc_freeze=False,
                 context_encoder='rgb'):
        super(RAFT_cmrnext, self).__init__()
        self.args = args
        self.with_uncertainty = with_uncertainty
        self.use_reflectance = use_reflectance
        self.unc_type = unc_type
        self.unc_freeze = unc_freeze
        self.context_encoder = context_encoder

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in args._get_kwargs():
            args.dropout = 0

        self.elu = nn.ELU()

        lidar_feat = 1
        if fourier_levels >= 0:
            lidar_feat = 2 * fourier_levels
            lidar_feat = lidar_feat + 1
        if use_reflectance:
            lidar_feat = lidar_feat + 1
        # feature network, context network, and update block
        if args.small:
            self.fnet_img = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.fnet_lidar = SmallEncoder(output_dim=128,
                                           norm_fn='instance',
                                           dropout=args.dropout,
                                           in_feat=lidar_feat)
            if self.context_encoder == 'rgb':
                self.cnet = SmallEncoder(output_dim=hdim + cdim,
                                         norm_fn='none',
                                         dropout=args.dropout)
            else:
                self.cnet = SmallEncoder(output_dim=hdim + cdim,
                                         norm_fn='none',
                                         dropout=args.dropout,
                                         in_feat=lidar_feat)
            if not self.unc_freeze:
                self.update_block = SmallUpdateBlock(self.args,
                                                     hidden_dim=hdim,
                                                     with_uncertainty=with_uncertainty,
                                                     unc_type=unc_type)
            else:
                self.update_block = SmallUpdateBlock(self.args,
                                                     hidden_dim=hdim,
                                                     with_uncertainty=False)
                if unc_type == "NLL":
                    self.update_block_unc = SmallUpdateBlock(self.args,
                                                             hidden_dim=hdim,
                                                             force_out_dim=2)
                elif unc_type == "DER":
                    self.update_block_unc = SmallUpdateBlock(self.args,
                                                             hidden_dim=hdim,
                                                             force_out_dim=6)

        else:
            self.fnet_img = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.fnet_lidar = BasicEncoder(output_dim=256,
                                           norm_fn='instance',
                                           dropout=args.dropout,
                                           in_feat=lidar_feat)
            if self.context_encoder == 'rgb':
                self.cnet = BasicEncoder(output_dim=hdim + cdim,
                                         norm_fn='batch',
                                         dropout=args.dropout)
            else:
                self.cnet = BasicEncoder(output_dim=hdim + cdim,
                                         norm_fn='batch',
                                         dropout=args.dropout,
                                         in_feat=lidar_feat)
            if not self.unc_freeze:
                self.update_block = BasicUpdateBlock(self.args,
                                                     hidden_dim=hdim,
                                                     with_uncertainty=with_uncertainty,
                                                     unc_type=unc_type)
            else:
                self.update_block = BasicUpdateBlock(self.args,
                                                     hidden_dim=hdim,
                                                     with_uncertainty=False)
                if unc_type == "NLL":
                    self.update_block_unc = BasicUpdateBlock(self.args,
                                                             hidden_dim=hdim,
                                                             force_out_dim=2)
                elif unc_type == "DER":
                    self.update_block_unc = BasicUpdateBlock(self.args,
                                                             hidden_dim=hdim,
                                                             force_out_dim=6)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, dtype=torch.float32):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device).type(dtype)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device).type(dtype)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask, uncertainty=None):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        up_uncertainty = None
        if uncertainty is not None:
            if self.unc_type == "NLL":
                uncertainty = self.elu(uncertainty) + 2
                up_uncertainty = F.unfold(uncertainty, [3, 3], padding=1)
                up_uncertainty = up_uncertainty.view(N, 2, 9, 1, 1, H, W)
                up_uncertainty = torch.sum(mask * up_uncertainty, dim=2)
                up_uncertainty = up_uncertainty.permute(0, 1, 4, 2, 5, 3)
                up_uncertainty = up_uncertainty.reshape(N, 2, 8 * H, 8 * W)

            elif self.unc_type == "DER":
                up_uncertainty = F.unfold(uncertainty, [3, 3], padding=1)
                up_uncertainty = up_uncertainty.view(N, 6, 9, 1, 1, H, W)
                up_uncertainty = torch.sum(mask * up_uncertainty, dim=2)
                up_uncertainty = up_uncertainty.permute(0, 1, 4, 2, 5, 3)
                up_uncertainty = up_uncertainty.reshape(N, 6, 8 * H, 8 * W)

                up_uncertainty = F.softplus(up_uncertainty)  # v

        return up_flow.reshape(N, 2, 8 * H, 8 * W), up_uncertainty

    def forward(self,
                image1,
                lidar_image,
                iters=12,
                flow_init=None,
                upsample=True,
                test_mode=False):
        """ Estimate optical flow between pair of frames """

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1 = self.fnet_lidar(lidar_image)
        fmap2 = self.fnet_img(image1)
        
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        if self.context_encoder == 'rgb':
            cnet = self.cnet(image1)
        else:
            cnet = self.cnet(lidar_image)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        if self.unc_freeze:
            net_unc = torch.tanh(net)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1, dtype=fmap1.dtype)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        flow_uncertainties = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            if self.unc_freeze:
                net_unc, up_mask_unc, delta_flow_unc = self.update_block_unc(
                    net_unc, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow[:, :2]
            uncertainty = None
            if self.with_uncertainty:
                if self.unc_freeze:
                    uncertainty = delta_flow_unc
                else:
                    uncertainty = delta_flow[:, 2:]

            uncertainty_up = None
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up, uncertainty_up = self.upsample_flow(coords1 - coords0, up_mask,
                                                             uncertainty)

            flow_predictions.append(flow_up)
            flow_uncertainties.append(uncertainty_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions, flow_uncertainties
