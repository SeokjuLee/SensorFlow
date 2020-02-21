from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp2, inverse_warp3, compute_rigid_flow, fwd_warp_depth
import math
from matplotlib import pyplot as plt
import numpy as np
from flow_utils import vis_flow
from torch_sparse import coalesce

import pdb

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pixel_uv = None


def set_id_grid(flow):
    global pixel_uv
    b, _, h, w = flow.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(flow)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(flow)  # [1, H, W]

    pixel_uv = torch.stack((j_range, i_range), dim=1)  # [1, 2, H, W]


def inv_warp(img, flo):
    '''
    img: b x 3 x h x w
    flo: b x 2 x h x w
    '''
    bs, ch, gh, gw = img.size()
    mgrid_np = np.expand_dims(np.mgrid[0:gw,0:gh].transpose(0,2,1).astype(np.float32),0).repeat(bs, axis=0)
    mgrid = torch.from_numpy(mgrid_np).type_as(flo)
    grid = mgrid.add(flo).permute(0,2,3,1)   # b x 2 x gh x gw

    grid[:,:,:,0] = grid[:,:,:,0].sub(gw/2).div(gw/2)
    grid[:,:,:,1] = grid[:,:,:,1].sub(gh/2).div(gh/2)
    
    if np.array(torch.__version__[:3]).astype(float) >= 1.3:
        img_w = F.grid_sample(img, grid, align_corners=True)
    else:
        img_w = F.grid_sample(img, grid)

    valid = (grid.abs().max(dim=-1)[0] <= 1).unsqueeze(1).float()

    img_w[(valid==0).repeat(1,ch,1,1)] = 0

    return img_w, valid


def fwd_warp(img, flo, upscale=2):
    '''
    img: b x 3 x h x w
    flo: b x 2 x h x w
    '''
    img = F.interpolate(img, scale_factor=upscale)
    flo = F.interpolate(flo, scale_factor=upscale) * upscale
    bs, ch, gh, gw = img.size()
    mgrid_np = np.expand_dims(np.mgrid[0:gw,0:gh].transpose(0,2,1).astype(np.float32),0).repeat(bs, axis=0)
    mgrid = torch.from_numpy(mgrid_np).type_as(flo)
    grid = mgrid.add(flo)   # b x 2 x gh x gw

    hh, ww = int(gh/upscale), int(gw/upscale)

    image_w, valid_w = [], []
    for bb in range(bs):
        coo = grid[bb].reshape(2,-1)
        rgb = img[bb].reshape(3,-1).permute(1,0)
        idx = coo.long()[[1,0]]
        idx[0][idx[0]<0] = gh
        idx[0][idx[0]>gh-1] = gh
        idx[1][idx[1]<0] = gw
        idx[1][idx[1]>gw-1] = gw
        idx = idx / upscale
        _idx, _val = coalesce(idx, rgb, m=hh+1, n=ww+1, op='mean')
        image_w.append( torch.sparse.FloatTensor(_idx, _val, torch.Size([hh+1,ww+1,3])).to_dense()[:-1,:-1].permute(2,0,1) )
        valid_w.append( (1- (torch.sparse.FloatTensor(_idx, _val, torch.Size([hh+1,ww+1,3])).to_dense()[:-1,:-1]==0).float()).prod(dim=2).unsqueeze(0) )
    image_w = torch.stack(image_w, dim=0)
    valid_w = torch.stack(valid_w, dim=0)

    image_w[(valid_w==0).repeat(1,ch,1,1)] = 0

    return image_w, valid_w


def L2_norm(x, dim=1, keepdim=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset, dim=dim, keepdim=True)
    return l2_norm


"""
def compute_photo_loss(tgt_img, ref_imgs, r2t_flows, t2r_flows, args, tgt_noccs=[[None], [None]], ref_noccs=[[None], [None]]):
    photo_loss = 0

    num_scales = min(len(r2t_flows[0]), args.num_scales)
    for ref_img, r2t_flow, t2r_flow, tgt_nocc, ref_nocc in zip(ref_imgs, r2t_flows, t2r_flows, tgt_noccs, ref_noccs):
        for s in range(num_scales):
            b, _, h, w = r2t_flow[s].size()

            tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
            ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area')
            # pdb.set_trace()

            photo_loss1 = compute_pairwise_warp_loss(ref_img_scaled, tgt_img_scaled, r2t_flow[s], args, tgt_nocc[s])
            photo_loss2 = compute_pairwise_warp_loss(tgt_img_scaled, ref_img_scaled, t2r_flow[s], args, ref_nocc[s])
            # pdb.set_trace()

            photo_loss += (photo_loss1 + photo_loss2)

    return photo_loss
"""

def compute_photo_loss(curr_imgs, pred_imgs, inv_flows, fwd_flows, args, pred_noccs=[[None], [None]], curr_noccs=[[None], [None]]):
    photo_loss = 0

    num_scales = min(len(inv_flows[0]), args.num_scales)
    for curr_img, pred_img, inv_flow, fwd_flow, pred_nocc, curr_nocc in zip(curr_imgs, pred_imgs, inv_flows, fwd_flows, pred_noccs, curr_noccs):
        for s in range(num_scales):
            b, _, h, w = inv_flow[s].size()

            curr_img_scaled = F.interpolate(curr_img, (h, w), mode='area')
            pred_img_scaled = F.interpolate(pred_img, (h, w), mode='area')

            photo_loss1 = compute_pairwise_warp_loss(curr_img_scaled, pred_img_scaled, inv_flow[s], args, pred_nocc[s])
            photo_loss2 = compute_pairwise_warp_loss(pred_img_scaled, curr_img_scaled, fwd_flow[s], args, curr_nocc[s])
            # pdb.set_trace()

            photo_loss += (photo_loss1 + photo_loss2)

    return photo_loss


def compute_pairwise_warp_loss(ref_img, tgt_img, flow, args, tgt_nocc):
    global pixel_uv
    b, _, h, w = flow.size()
    if (pixel_uv is None) or pixel_uv.size(2) != h:
        set_id_grid(flow)

    flow_uv = pixel_uv.expand(b,2,h,w) + flow

    u_norm = 2*flow_uv[:,0]/(w-1) - 1
    v_norm = 2*flow_uv[:,1]/(h-1) - 1

    """ Masking flied-out regions """
    u_mask = ((u_norm > 1)+(u_norm < -1)).detach()
    u_norm[u_mask] = 2
    v_mask = ((v_norm > 1)+(v_norm < -1)).detach()
    v_norm[v_mask] = 2

    flow_grid = torch.stack([u_norm, v_norm], dim=-1)    # [B, H, W, 2]

    if np.array(torch.__version__[:3]).astype(float) >= 1.3:
        r2t_img = F.grid_sample(ref_img, flow_grid, align_corners=True)
    else:
        r2t_img = F.grid_sample(ref_img, flow_grid)

    valid_grid = flow_grid.abs().max(dim=-1)[0] <= 1
    valid_grid = valid_grid.unsqueeze(1).float()
    # pdb.set_trace()
    '''
        bb = 0
        plt.close('all');
        plt.figure(1); plt.imshow(valid_grid[bb,0].detach().cpu().numpy()); plt.colorbar(); plt.ion(); plt.show();
        plt.figure(2); plt.imshow(tgt_nocc[bb,0].detach().cpu().numpy()); plt.colorbar(); plt.ion(); plt.show();
    
    '''

    diff_img = (tgt_img - r2t_img).abs()

    if args.with_ssim:
        ssim_map = (0.5*(1-ssim(tgt_img, r2t_img))).clamp(0, 1)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    if tgt_nocc is not None:
        valid_grid = valid_grid * tgt_nocc

    reconstruction_loss = mean_on_mask(diff_img, valid_grid)

    return reconstruction_loss


"""
def compute_flow_consistency_loss(r2t_flows, t2r_flows, args, alpha1=0.01, alpha2=0.5):
    loss = 0
    noc_masks_tgt, noc_masks_ref = [], []

    num_scales = min(len(r2t_flows[0]), args.num_scales)
    for r2t_flow, t2r_flow in zip(r2t_flows, t2r_flows):
        noc_masks_tgt_scale, noc_masks_ref_scale = [], []
        for s in range(num_scales):
            tr2rt_flow = inv_warp(t2r_flow[s], r2t_flow[s])
            rt2tr_flow = inv_warp(r2t_flow[s], t2r_flow[s])

            r2t_flow_diff = torch.abs(tr2rt_flow + r2t_flow[s])
            t2r_flow_diff = torch.abs(rt2tr_flow + t2r_flow[s])

            r2t_consist_bound = alpha1 * (L2_norm(r2t_flow[s]) + L2_norm(tr2rt_flow)) + alpha2
            t2r_consist_bound = alpha1 * (L2_norm(t2r_flow[s]) + L2_norm(rt2tr_flow)) + alpha2

            noc_mask_ref = (L2_norm(t2r_flow_diff) < t2r_consist_bound).type(torch.FloatTensor).cuda()
            noc_mask_tgt = (L2_norm(r2t_flow_diff) < r2t_consist_bound).type(torch.FloatTensor).cuda()

            noc_masks_tgt_scale.append(noc_mask_tgt)
            noc_masks_ref_scale.append(noc_mask_ref)

            loss += 1/2 * \
                    ( (r2t_flow_diff.mean(dim=1, keepdim=True) * noc_mask_tgt).sum() / torch.clamp(noc_mask_tgt.sum(), min=1e-10) + \
                      (t2r_flow_diff.mean(dim=1, keepdim=True) * noc_mask_ref).sum() / torch.clamp(noc_mask_ref.sum(), min=1e-10) )

        noc_masks_tgt.append(noc_masks_tgt_scale)
        noc_masks_ref.append(noc_masks_ref_scale)

    return loss, noc_masks_tgt, noc_masks_ref
"""


"""
def compute_flow_consistency_loss(curr_imgs, pred_imgs, inv_flows, fwd_flows, args, alpha1=0.01, alpha2=0.5):
    loss = 0
    pred_noccs, curr_noccs = [], []

    num_scales = min(len(inv_flows[0]), args.num_scales)
    for curr_img, pred_img, inv_flow, fwd_flow in zip(curr_imgs, pred_imgs, inv_flows, fwd_flows):
        pred_noccs_scale, curr_noccs_scale = [], []
        for s in range(num_scales):
            b, _, h, w = inv_flow[s].size()
            curr_img_scaled = F.interpolate(curr_img, (h, w), mode='area')
            pred_img_scaled = F.interpolate(pred_img, (h, w), mode='area')

            pred_nocc = fwd_warp(curr_img_scaled, fwd_flow[s], upscale=1)[1]    # ref input -> nocc mask on tgt
            curr_nocc = fwd_warp(pred_img_scaled, inv_flow[s], upscale=1)[1]    # ref input -> nocc mask on ref

            inv2fwd_flow, inv2fwd_val = inv_warp(inv_flow[s], fwd_flow[s])
            fwd2inv_flow, fwd2inv_val = inv_warp(fwd_flow[s], inv_flow[s])

            fwd_flow_diff = L2_norm(torch.abs(inv2fwd_flow + fwd_flow[s])) * inv2fwd_val
            inv_flow_diff = L2_norm(torch.abs(fwd2inv_flow + inv_flow[s])) * fwd2inv_val
            # pdb.set_trace()
            '''
                plt.close('all')
                bb = 1
                plt.figure(1); plt.imshow(pred_nocc[bb,0].detach().cpu()); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.ion(); plt.show();
                plt.figure(2); plt.imshow(curr_nocc[bb,0].detach().cpu()); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.ion(); plt.show();
                plt.figure(3); plt.imshow(fwd_flow_diff[bb,0].detach().cpu()); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.ion(); plt.show();
                plt.figure(4); plt.imshow(inv_flow_diff[bb,0].detach().cpu()); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.ion(); plt.show();
                plt.figure(5); plt.imshow((curr_nocc*fwd_flow_diff)[bb,0].detach().cpu()); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.ion(); plt.show();
                plt.figure(6); plt.imshow((pred_nocc*inv_flow_diff)[bb,0].detach().cpu()); plt.colorbar(); plt.grid(linestyle=':', linewidth=0.4); plt.ion(); plt.show();

            '''

            pred_noccs_scale.append(pred_nocc)
            curr_noccs_scale.append(curr_nocc)

            loss += 1/2 * (fwd_flow_diff*curr_nocc).mean() + (inv_flow_diff*pred_nocc).mean()

        pred_noccs.append(pred_noccs_scale)
        curr_noccs.append(curr_noccs_scale)

    return loss, pred_noccs, curr_noccs
"""


"""
def compute_flow_consistency_loss_debug(curr_imgs, pred_imgs, inv_flows, fwd_flows, args):
    loss = 0
    pred_noccs, curr_noccs = [], []

    num_scales = min(len(inv_flows[0]), args.num_scales)
    for curr_img, pred_img, inv_flow, fwd_flow in zip(curr_imgs, pred_imgs, inv_flows, fwd_flows):
        pred_noccs_scale, curr_noccs_scale = [], []
        for s in range(num_scales):
            b, _, h, w = inv_flow[s].size()
            curr_img_scaled = F.interpolate(curr_img, (h, w), mode='area')
            pred_img_scaled = F.interpolate(pred_img, (h, w), mode='area')

            inv2fwd_flow, inv2fwd_val = inv_warp(inv_flow[s], fwd_flow[s])
            fwd2inv_flow, fwd2inv_val = inv_warp(fwd_flow[s], inv_flow[s])

            fwd_flow_diff_norm = ( L2_norm(torch.abs(fwd_flow[s] + inv2fwd_flow)) / L2_norm(torch.abs(fwd_flow[s] - inv2fwd_flow)).clamp(min=1e-6) ).clamp(0,1)
            inv_flow_diff_norm = ( L2_norm(torch.abs(inv_flow[s] + fwd2inv_flow)) / L2_norm(torch.abs(fwd_flow[s] - inv2fwd_flow)).clamp(min=1e-6) ).clamp(0,1)

            fwd_weight_mask = (1 - fwd_flow_diff_norm) * inv2fwd_val
            inv_weight_mask = (1 - inv_flow_diff_norm) * fwd2inv_val

            pred_noccs_scale.append(inv_weight_mask)
            curr_noccs_scale.append(fwd_weight_mask)

            # pdb.set_trace()
            '''
                # pred_nocc = fwd_warp(curr_img_scaled, fwd_flow[s], upscale=1)[1]    # ref input -> nocc mask on tgt
                # curr_nocc = fwd_warp(pred_img_scaled, inv_flow[s], upscale=1)[1]    # ref input -> nocc mask on ref

                alpha1=0.04, alpha2=1.0

                fwd_flow_diff = L2_norm(torch.abs(fwd_flow[s] + inv2fwd_flow))# * inv2fwd_val
                inv_flow_diff = L2_norm(torch.abs(inv_flow[s] + fwd2inv_flow))# * fwd2inv_val

                fwd_consist_bound = alpha1 * (L2_norm(fwd_flow[s]) + L2_norm(inv2fwd_flow)) + alpha2
                inv_consist_bound = alpha1 * (L2_norm(inv_flow[s]) + L2_norm(fwd2inv_flow)) + alpha2

                fwd_noc_mask = (L2_norm(fwd_flow_diff) < fwd_consist_bound).type(torch.FloatTensor).cuda()
                inv_noc_mask = (L2_norm(inv_flow_diff) < inv_consist_bound).type(torch.FloatTensor).cuda()

                bb = 0
                plt.close('all')
                fig = plt.figure(1, figsize=(24, 8))
                ea1 = 2; ea2 = 6; ii = 1;
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(curr_img[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "curr_img", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(fwd_flow_diff[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "fwd_flow_diff", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(fwd_consist_bound[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "fwd_consist_bound", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(fwd_noc_mask[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "fwd_noc_mask", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow((fwd_noc_mask*fwd_flow_diff)[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "fwd_noc_mask*fwd_flow_diff", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(fwd_weight_mask[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "fwd_weight_mask", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(pred_img[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "pred_img", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(inv_flow_diff[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "inv_flow_diff", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(inv_consist_bound[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "inv_consist_bound", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(inv_noc_mask[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "inv_noc_mask", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow((inv_noc_mask*inv_flow_diff)[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "fwd_noc_mask*fwd_flow_diff", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(inv_weight_mask[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "inv_weight_mask", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                plt.tight_layout(); plt.ion(); plt.show();
            
            '''

        pred_noccs.append(pred_noccs_scale)
        curr_noccs.append(curr_noccs_scale)

    return pred_noccs, curr_noccs
"""


def compute_flow_consistency_loss(inv_flows, fwd_flows, args):
    loss = 0
    pred_noccs, curr_noccs = [], []

    num_scales = min(len(inv_flows[0]), args.num_scales)
    for inv_flow, fwd_flow in zip(inv_flows, fwd_flows):
        pred_noccs_scale, curr_noccs_scale = [], []
        for s in range(num_scales):
            inv2fwd_flow, inv2fwd_val = inv_warp(inv_flow[s], fwd_flow[s])
            fwd2inv_flow, fwd2inv_val = inv_warp(fwd_flow[s], inv_flow[s])

            fwd_flow_diff_norm = ( L2_norm(torch.abs(fwd_flow[s] + inv2fwd_flow)) / L2_norm(torch.abs(fwd_flow[s]) + torch.abs(inv2fwd_flow)).clamp(min=1e-6) ).clamp(0,1)
            inv_flow_diff_norm = ( L2_norm(torch.abs(inv_flow[s] + fwd2inv_flow)) / L2_norm(torch.abs(fwd_flow[s]) + torch.abs(inv2fwd_flow)).clamp(min=1e-6) ).clamp(0,1)

            fwd_weight_mask = (1 - fwd_flow_diff_norm) * inv2fwd_val
            inv_weight_mask = (1 - inv_flow_diff_norm) * fwd2inv_val

            pred_noccs_scale.append(inv_weight_mask)
            curr_noccs_scale.append(fwd_weight_mask)

            loss += 1/2 * (mean_on_mask(fwd_flow_diff_norm, inv2fwd_val) + mean_on_mask(inv_flow_diff_norm, fwd2inv_val))
            # pdb.set_trace()
            '''
                bb = 0
                plt.close('all')
                fig = plt.figure(1, figsize=(24, 8))
                ea1 = 2; ea2 = 4; ii = 1;
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(fwd_flow_diff_norm[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "fwd_flow_diff_norm", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(inv2fwd_val[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "inv2fwd_val", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(inv_flow_diff_norm[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "inv_flow_diff_norm", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(fwd2inv_val[bb,0].detach().cpu()); plt.colorbar(); plt.text(0, 20, "fwd2inv_val", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                plt.tight_layout(); plt.ion(); plt.show();

            '''

        pred_noccs.append(pred_noccs_scale)
        curr_noccs.append(curr_noccs_scale)

    return loss, pred_noccs, curr_noccs



def compute_occ_guide_loss(curr_imgs, pred_imgs, inv_flows, fwd_flows, pred_noccs, curr_noccs, args):
    loss = 0

    num_scales = min(len(inv_flows[0]), args.num_scales)
    for curr_img, pred_img, inv_flow, fwd_flow, pred_nocc, curr_nocc in zip(curr_imgs, pred_imgs, inv_flows, fwd_flows, pred_noccs, curr_noccs):
        for s in range(num_scales):
            b, _, h, w = inv_flow[s].size()
            curr_img_scaled = F.interpolate(curr_img, (h, w), mode='area')
            pred_img_scaled = F.interpolate(pred_img, (h, w), mode='area')

            pred_nocc_gt = fwd_warp(curr_img_scaled, fwd_flow[s], upscale=2)[1].detach()
            curr_nocc_gt = fwd_warp(pred_img_scaled, inv_flow[s], upscale=2)[1].detach()

            cost1 = torch.norm(pred_nocc_gt-pred_nocc[s], 1, dim=1, keepdim=True).mean()
            cost2 = torch.norm(curr_nocc_gt-curr_nocc[s], 1, dim=1, keepdim=True).mean()
            # pdb.set_trace()
            '''
                bb = 0
                plt.close('all')
                plt.figure(1), plt.imshow(pred_nocc_gt[bb,0].detach().cpu()), plt.colorbar(), plt.ion(), plt.show()
                plt.figure(2), plt.imshow(pred_nocc[s][bb,0].detach().cpu()), plt.colorbar(), plt.ion(), plt.show()
                plt.figure(3), plt.imshow(curr_nocc_gt[bb,0].detach().cpu()), plt.colorbar(), plt.ion(), plt.show()
                plt.figure(4), plt.imshow(curr_nocc[s][bb,0].detach().cpu()), plt.colorbar(), plt.ion(), plt.show()
            
            '''
            loss += cost1 + cost2

    return loss


def compute_rigid_flow_loss(tgt_img, ref_imgs, r2t_flows, t2r_flows, tgt_depth, ref_depths, r2t_poses, t2r_poses, intrinsics, args):
    loss = 0

    num_scales = min(len(r2t_flows[0]), args.num_scales)
    for ref_img, r2t_flow, t2r_flow, ref_depth, r2t_pose, t2r_pose in zip(ref_imgs, r2t_flows, t2r_flows, ref_depths, r2t_poses, t2r_poses):
        for s in range(num_scales):
            b, _, h, w = r2t_flow[s].size()
            downscale = tgt_img.size(2)/h

            tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area').detach()
            ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area').detach()
            tgt_depth_scaled = F.interpolate(tgt_depth, (h, w), mode='area').detach()
            ref_depth_scaled = F.interpolate(ref_depth, (h, w), mode='area').detach()
            intrinsic_scaled = torch.cat( (intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1 ).detach()

            r2t_rig_flow, r2t_val_mask = compute_rigid_flow(tgt_depth, r2t_pose, intrinsic_scaled)
            t2r_rig_flow, t2r_val_mask = compute_rigid_flow(ref_depth, t2r_pose, intrinsic_scaled)
            # pdb.set_trace()
            '''
                ### dpoint ###
                
                plt.close('all'); 
                bb = 0
                aaa = ref_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
                bbb = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
                ccc = vis_flow(r2t_rig_flow[bb].detach().cpu().numpy().transpose(1,2,0))
                ddd = vis_flow(r2t_flow[s][bb].detach().cpu().numpy().transpose(1,2,0))
                eee = inv_warp(ref_img, r2t_rig_flow)[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
                fff = inv_warp(ref_img, r2t_flow[s])[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
                ggg = r2t_rig_flow[bb,0].detach().cpu().numpy()
                hhh = r2t_flow[s][bb,0].detach().cpu().numpy()
                iii = np.abs(bbb-eee)
                jjj = np.abs(bbb-fff)
                kkk = r2t_rig_flow[bb,1].detach().cpu().numpy()
                lll = r2t_flow[s][bb,1].detach().cpu().numpy()
                mmm = 1/tgt_depth[bb,0].detach().cpu().numpy()
                fig = plt.figure(1, figsize=(20, 12))
                ea1 = 3; ea2 = 4; ii = 1;
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(aaa); plt.colorbar(); plt.text(0, 20, "ref_img", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(bbb); plt.colorbar(); plt.text(0, 20, "tgt_img", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ccc); plt.colorbar(); plt.text(0, 20, "r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ddd); plt.colorbar(); plt.text(0, 20, "r2t_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(eee); plt.colorbar(); plt.text(0, 20, "r2t_rig_img", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(fff); plt.colorbar(); plt.text(0, 20, "r2t_img", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ggg); plt.colorbar(); plt.text(0, 20, "u_r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(hhh, vmin=ggg.min(), vmax=ggg.max()); plt.colorbar(); plt.text(0, 20, "u_r2t_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(iii); plt.colorbar(); plt.text(0, 20, "tgt_rig_err", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(jjj, vmin=iii.min(), vmax=iii.max()); plt.colorbar(); plt.text(0, 20, "tgt_err", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(kkk); plt.colorbar(); plt.text(0, 20, "v_r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(lll, vmin=kkk.min(), vmax=kkk.max()); plt.colorbar(); plt.text(0, 20, "v_r2t_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                plt.tight_layout(); plt.ion(); plt.show();
                

                #plt.close('all'); 
                #bb = 1
                aaa = ref_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
                bbb = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
                ccc = vis_flow(r2t_rig_flow[bb].detach().cpu().numpy().transpose(1,2,0))
                ddd = vis_flow(t2r_rig_flow[bb].detach().cpu().numpy().transpose(1,2,0))
                eee = r2t_rig_flow[bb,0].detach().cpu().numpy()
                fff = t2r_rig_flow[bb,0].detach().cpu().numpy()
                ggg = r2t_rig_flow[bb,1].detach().cpu().numpy()
                hhh = t2r_rig_flow[bb,1].detach().cpu().numpy()
                fig = plt.figure(2, figsize=(12, 13))
                ea1 = 5; ea2 = 2; ii = 1;
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(aaa); plt.colorbar(); plt.text(0, 20, "ref", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(bbb); plt.colorbar(); plt.text(0, 20, "tgt", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ccc); plt.colorbar(); plt.text(0, 20, "r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ddd); plt.colorbar(); plt.text(0, 20, "t2r_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(eee); plt.colorbar(); plt.text(0, 20, "X_r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(fff); plt.colorbar(); plt.text(0, 20, "X_t2r_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ggg); plt.colorbar(); plt.text(0, 20, "Y_r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(hhh); plt.colorbar(); plt.text(0, 20, "Y_t2r_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                plt.tight_layout(); plt.ion(); plt.show();


            '''
            # loss1 = F.smooth_l1_loss(r2t_rig_flow[r2t_val_mask.expand(b,2,h,w) > 0], r2t_flow[s][r2t_val_mask.expand(b,2,h,w) > 0])
            # loss2 = F.smooth_l1_loss(t2r_rig_flow[t2r_val_mask.expand(b,2,h,w) > 0], t2r_flow[s][t2r_val_mask.expand(b,2,h,w) > 0])

            loss1 = torch.norm(r2t_rig_flow-r2t_flow[s], 2, 1, keepdim=True)[r2t_val_mask>0].mean()
            loss2 = torch.norm(t2r_rig_flow-t2r_flow[s], 2, 1, keepdim=True)[t2r_val_mask>0].mean()

            loss += (loss1 + loss2)

    return loss


def compute_fwd_rigid_flow_loss(tgt_img, ref_imgs, r2t_flows, t2r_flows, tgt_depth, ref_depths, r2t_poses, t2r_poses, intrinsics, args):
    loss = 0

    num_scales = min(len(r2t_flows[0]), args.num_scales)
    for ref_img, r2t_flow, t2r_flow, ref_depth, r2t_pose, t2r_pose in zip(ref_imgs, r2t_flows, t2r_flows, ref_depths, r2t_poses, t2r_poses):
        for s in range(num_scales):
            b, _, h, w = r2t_flow[s].size()
            downscale = tgt_img.size(2)/h

            tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area').detach()
            ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area').detach()
            tgt_depth_scaled = F.interpolate(tgt_depth, (h, w), mode='area').detach()
            ref_depth_scaled = F.interpolate(ref_depth, (h, w), mode='area').detach()
            intrinsic_scaled = torch.cat( (intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1 ).detach()

            r2t_rig_flow, r2t_val_mask = compute_fwd_rigid_flow(tgt_depth, r2t_pose, intrinsic_scaled)
            t2r_rig_flow, t2r_val_mask = compute_fwd_rigid_flow(ref_depth, t2r_pose, intrinsic_scaled)

            t2r_depth, t2r_val_mask = fwd_warp_depth(tgt_depth[:,0], r2t_pose, intrinsic_scaled)
            r2t_depth, r2t_val_mask = fwd_warp_depth(ref_depth[:,0], t2r_pose, intrinsic_scaled)
            # pdb.set_trace()
            '''
                
                plt.close('all'); bb = 3
                aaa = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
                bbb = 1/tgt_depth[bb,0].detach().cpu().numpy()
                ccc = vis_flow(r2t_rig_flow[bb].detach().cpu().numpy().transpose(1,2,0))
                ddd = vis_flow(r2t_flow[s][bb].detach().cpu().numpy().transpose(1,2,0))
                eee = r2t_rig_flow[bb,0].detach().cpu().numpy()
                fff = r2t_flow[s][bb,0].detach().cpu().numpy()
                ggg = r2t_rig_flow[bb,1].detach().cpu().numpy()
                hhh = r2t_flow[s][bb,1].detach().cpu().numpy()
                fig = plt.figure(1, figsize=(11, 11))
                ea1 = 4; ea2 = 2; ii = 1;
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(aaa); plt.colorbar(); plt.text(0, 20, "tgt", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(bbb); plt.colorbar(); plt.text(0, 20, "tgt_depth", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ccc); plt.colorbar(); plt.text(0, 20, "r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ddd); plt.colorbar(); plt.text(0, 20, "r2t_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(eee); plt.colorbar(); plt.text(0, 20, "X_r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(fff, vmin=eee.min(), vmax=eee.max()); plt.colorbar(); plt.text(0, 20, "X_r2t_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ggg); plt.colorbar(); plt.text(0, 20, "Y_r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(hhh, vmin=ggg.min(), vmax=ggg.max()); plt.colorbar(); plt.text(0, 20, "Y_r2t_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                plt.tight_layout(); plt.ion(); plt.show();
                

                plt.close('all'); bb = 1
                aaa = ref_imgs[0][bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
                bbb = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
                ccc = vis_flow(r2t_rig_flow[bb].detach().cpu().numpy().transpose(1,2,0))
                ddd = vis_flow(t2r_rig_flow[bb].detach().cpu().numpy().transpose(1,2,0))
                eee = r2t_rig_flow[bb,0].detach().cpu().numpy()
                fff = t2r_rig_flow[bb,0].detach().cpu().numpy()
                ggg = r2t_rig_flow[bb,1].detach().cpu().numpy()
                hhh = t2r_rig_flow[bb,1].detach().cpu().numpy()
                fig = plt.figure(1, figsize=(12, 13))
                ea1 = 5; ea2 = 2; ii = 1;
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(aaa); plt.colorbar(); plt.text(0, 20, "ref", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(bbb); plt.colorbar(); plt.text(0, 20, "tgt", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ccc); plt.colorbar(); plt.text(0, 20, "r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ddd); plt.colorbar(); plt.text(0, 20, "t2r_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(eee); plt.colorbar(); plt.text(0, 20, "X_r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(fff); plt.colorbar(); plt.text(0, 20, "X_t2r_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ggg); plt.colorbar(); plt.text(0, 20, "Y_r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(hhh); plt.colorbar(); plt.text(0, 20, "Y_t2r_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                plt.tight_layout(); plt.ion(); plt.show();
                
                
                
                

                plt.close('all'); bb = 1
                aaa = ref_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
                bbb = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
                ccc = ref_depth[bb,0].detach().cpu().numpy()
                ddd = tgt_depth[bb,0].detach().cpu().numpy()
                eee = t2r_depth[bb].detach().cpu().numpy()
                fff = t2r_val_mask[bb].detach().cpu().numpy()
                fig = plt.figure(1, figsize=(12, 13))
                ea1 = 3; ea2 = 2; ii = 1;
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(aaa); plt.colorbar(); plt.text(0, 20, "ref", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(bbb); plt.colorbar(); plt.text(0, 20, "tgt", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ccc); plt.colorbar(); plt.text(0, 20, "r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(ddd); plt.colorbar(); plt.text(0, 20, "t2r_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(eee); plt.colorbar(); plt.text(0, 20, "X_r2t_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                fig.add_subplot(ea1,ea2,ii); ii += 1;
                plt.imshow(fff); plt.colorbar(); plt.text(0, 20, "X_t2r_rig_flow", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
                plt.tight_layout(); plt.ion(); plt.show();

            '''

    return loss



#############################################################################################################################################################

# compute photometric loss (with ssim) and geometry consistency loss
def compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, poses, poses_inv, args):

    photo_loss = 0
    geometry_loss = 0

    num_scales = min(len(tgt_depth), args.num_scales)
    for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
        for s in range(num_scales):
            b, _, h, w = tgt_depth[s].size()
            downscale = tgt_img.size(2)/h

            tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
            ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area')
            intrinsic_scaled = torch.cat( (intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1 )

            photo_loss1, geometry_loss1 = compute_pairwise_loss(tgt_img_scaled, ref_img_scaled, tgt_depth[s], ref_depth[s], pose, intrinsic_scaled, args)
            photo_loss2, geometry_loss2 = compute_pairwise_loss(ref_img_scaled, tgt_img_scaled, ref_depth[s], tgt_depth[s], pose_inv, intrinsic_scaled, args)

            photo_loss += (photo_loss1 + photo_loss2)
            geometry_loss += (geometry_loss1 + geometry_loss2)

    return photo_loss, geometry_loss


def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, args):

    # ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, ref_depth, pose, intrinsic, args.padding_mode)
    ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp3(ref_img, tgt_depth, ref_depth, pose, intrinsic, args.padding_mode)
    # pdb.set_trace()
    '''
        plt.close('all')
        bb = 6
        aaa = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
        bbb = ref_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
        ccc = ref_img_warped[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
        ddd = np.abs((tgt_img[bb] - ref_img_warped[bb]).detach().cpu().numpy().transpose(1,2,0))
        eee = tgt_depth[bb,0].detach().cpu().numpy()
        fig = plt.figure(1, figsize=(11, 11))
        ea1 = 4; ea2 = 2; ii = 1;
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(aaa); plt.colorbar(); plt.text(0, 20, "tgt", bbox={'facecolor': 'yellow', 'alpha': 0.5});
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(bbb); plt.colorbar(); plt.text(0, 20, "ref", bbox={'facecolor': 'yellow', 'alpha': 0.5});
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(ccc); plt.colorbar(); plt.text(0, 20, "ref_w", bbox={'facecolor': 'yellow', 'alpha': 0.5});
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(ddd); plt.colorbar(); plt.text(0, 20, "err", bbox={'facecolor': 'yellow', 'alpha': 0.5});
        fig.add_subplot(ea1,ea2,ii); ii += 1;
        plt.imshow(eee); plt.colorbar(); plt.text(0, 20, "d_tgt", bbox={'facecolor': 'yellow', 'alpha': 0.5});
        plt.tight_layout(); plt.ion(); plt.show();


    '''

    diff_img = (tgt_img - ref_img_warped).abs()

    diff_depth = ((computed_depth - projected_depth).abs() /
                  (computed_depth + projected_depth).abs()).clamp(0, 1)

    if args.with_ssim:
        ssim_map = (0.5*(1-ssim(tgt_img, ref_img_warped))).clamp(0, 1)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    if args.with_mask:
        weight_mask = (1 - diff_depth)
        diff_img = diff_img * weight_mask

    # compute loss
    reconstruction_loss = mean_on_mask(diff_img, valid_mask)
    geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)

    return reconstruction_loss, geometry_consistency_loss


# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / mask.sum()
    return mean_value


def edge_aware_smoothness_loss(pred_disp, img, max_scales):
    def gradient_x(img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(img):
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gy

    def get_edge_smoothness(img, pred):
        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)

        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x),
                                          1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y),
                                          1, keepdim=True))

        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.

    s = 0
    for scaled_disp in pred_disp:
        s += 1
        if s > max_scales:
            break

        b, _, h, w = scaled_disp.size()
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += get_edge_smoothness(scaled_img, scaled_disp) * weight
        weight /= 4.0

    return loss


def compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs, max_scales=1):
    loss = edge_aware_smoothness_loss(tgt_depth, tgt_img, max_scales)

    for ref_depth, ref_img in zip(ref_depths, ref_imgs):
        loss += edge_aware_smoothness_loss(ref_depth, ref_img, max_scales)

    return loss


def compute_flow_smooth_loss(r2t_flows, tgt_img, t2r_flows, ref_imgs, max_scales=1):
    loss = 0

    for r2t_flow in r2t_flows:
        loss += edge_aware_smoothness_loss(r2t_flow, tgt_img, max_scales)

    for t2r_flow, ref_img in zip(t2r_flows, ref_imgs):
        loss += edge_aware_smoothness_loss(t2r_flow, ref_img, max_scales)

    return loss


def create_gaussian_window(window_size, channel):
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window@(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(
        channel, 1, window_size, window_size).contiguous()
    return window


window_size = 5
gaussian_img_kernel = create_gaussian_window(window_size, 3).float().to(device)


def ssim(img1, img2):
    params = {'weight': gaussian_img_kernel,
              'groups': 3, 'padding': window_size//2}
    mu1 = F.conv2d(img1, **params)
    mu2 = F.conv2d(img2, **params)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, **params) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, **params) - mu2_sq
    sigma12 = F.conv2d(img1*img2, **params) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map


@torch.no_grad()
def compute_errors(gt, pred):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    crop_mask = gt[0] != gt[0]
    y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
    x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
    crop_mask[y1:y2, x1:x2] = 1
    max_depth = 80

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        valid_pred = valid_pred * \
            torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]
