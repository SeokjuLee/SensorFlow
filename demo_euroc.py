import argparse
import time
import csv
from path import Path
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms

import models
from utils import save_checkpoint
from loss_functions import compute_photo_loss, compute_flow_smooth_loss, compute_rigid_flow_loss, compute_errors
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter

from matplotlib import pyplot as plt
from inverse_warp import pose_vec2mat
# from viz_filter import viz_filter
from imageio import imread
import matplotlib.animation as animation
from torch_sparse import coalesce
from flow_utils import vis_flow

import pdb

parser = argparse.ArgumentParser(description='Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video (KITTI and CityScapes)',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N',
                    help='sequence length for training', default=3)
parser.add_argument('--demi-length', type=int, metavar='N',
                    help='demi length for training', default=1)
parser.add_argument('--max-demi', type=int, metavar='N',
                    help='demi length for training', default=1)
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                    ' zeros will null gradients outside target image.'
                    ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int,
                    metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4,
                    type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4,
                    type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float,
                    metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0,
                    type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--pretrained-sf', dest='pretrained_sf',
                    default=None, metavar='PATH', help='path to pre-trained sfnet model')
parser.add_argument('--pretrained-disp', dest='pretrained_disp',
                    default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None,
                    metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv',
                    metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv',
                    metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('--sfnet', dest='sfnet', type=str, default='SFResNet',
                    choices=['SFNet', 'SFResNet', 'SFResNet_v2'], help='depth network architecture.')
parser.add_argument('--num-scales', '--number-of-scales',
                    type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float,
                    help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float,
                    help='weight for flow smoothness loss', metavar='W', default=0.1)
parser.add_argument('-f', '--flow-loss-weight', type=float,
                    help='weight for flow loss', metavar='W', default=0.1)
parser.add_argument('-c', '--flow-consistency-weight', type=float,
                    help='weight for flow consistency loss', metavar='W', default=0.1)
parser.add_argument('-o', '--occlusion-guide-weight', type=float,
                    help='weight for occlusion guide loss', metavar='W', default=0.5)
parser.add_argument('--with-ssim', action='store_true', help='use ssim loss',)
parser.add_argument('--with-mask', action='store_true',
                    help='use the the mask for handling moving objects and occlusions')
parser.add_argument('--name', dest='name', type=str, required=True,
                    help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--debug-mode', action='store_true', help='debug mode or not')
parser.add_argument('--rotation-mode', dest='rotation_mode', type=str, default='quaternion', choices=['quaternion', 'euler', '6D'], help='encoding rotation mode')
parser.add_argument('--fwd-flow', action='store_true', help='forward-flow mode or not')
parser.add_argument('--two-way-flow', action='store_true', help='two-way-flow mode or not')
parser.add_argument('--check-flows', action='store_true', help='check-flows mode or not')


best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def save_animation(file_name, img, img_w, fx, fy, traj, cmap='inferno', gap=1):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    fx_max, fx_min = np.array(fx).max(), np.array(fx).min()
    fy_max, fy_min = np.array(fy).max(), np.array(fy).min()
    anims = []
    for idx in range(0, traj.shape[0], gap):
        anim1 = [ax1.imshow(img, animated=True), ax1.annotate("input image", (8,22), bbox={'facecolor': 'white', 'alpha': 0.5})]
        anim2 = [ax2.imshow(img_w[idx], animated=True), ax2.annotate("fwd warped, (tx,ty,tz) = ({:.2f},{:.2f},{:.2f}) [cm]".format(traj[idx,3],traj[idx,4],traj[idx,5]), (8,22), bbox={'facecolor': 'white', 'alpha': 0.5})]
        anim3 = [ax3.imshow(fx[idx], cmap=cmap, animated=True, vmax=fx_max, vmin=fx_min), ax3.annotate("Fx, (tx,ty,tz) = ({:.2f},{:.2f},{:.2f}) [cm]".format(traj[idx,3],traj[idx,4],traj[idx,5]), (8,22), bbox={'facecolor': 'white', 'alpha': 0.5})]
        anim4 = [ax4.imshow(fy[idx], cmap=cmap, animated=True, vmax=fy_max, vmin=fy_min), ax4.annotate("Fy, (tx,ty,tz) = ({:.2f},{:.2f},{:.2f}) [cm]".format(traj[idx,3],traj[idx,4],traj[idx,5]), (8,22), bbox={'facecolor': 'white', 'alpha': 0.5})]
        anims.append(anim1 + anim2 + anim3 + anim4)

    ani = animation.ArtistAnimation(fig, anims, interval=200, blit=False, repeat_delay=20)
    fig.tight_layout(); fig.colorbar(anim1[0], ax=ax1); fig.colorbar(anim2[0], ax=ax2); fig.colorbar(anim3[0], ax=ax3); fig.colorbar(anim4[0], ax=ax4);
    print("=> Save GIF");
    ani.save(file_name, writer='imagemagick', fps=10, dpi=100)
    plt.close('all')


def save_animation2(file_name, img, img_iw, img_fw, noccs, inv_flows, fwd_flows, traj, cmap='inferno', gap=1):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(2,3,1)
    ax2 = fig.add_subplot(2,3,2)
    ax3 = fig.add_subplot(2,3,3)
    ax4 = fig.add_subplot(2,3,4)
    ax5 = fig.add_subplot(2,3,5)
    ax6 = fig.add_subplot(2,3,6)
    ax1.grid(linestyle=':', linewidth=0.4)
    ax2.grid(linestyle=':', linewidth=0.4)
    ax3.grid(linestyle=':', linewidth=0.4)
    ax4.grid(linestyle=':', linewidth=0.4)
    ax5.grid(linestyle=':', linewidth=0.4)
    ax6.grid(linestyle=':', linewidth=0.4)
    anims = []
    for idx in range(0, traj.shape[0], gap):
        anim1 = [ax1.imshow(img, animated=True), ax1.annotate("input image", (8,22), bbox={'facecolor': 'white', 'alpha': 0.5})]
        anim2 = [ax2.imshow(img_iw[idx], animated=True), ax2.annotate("inv warped, (tx,ty,tz) = ({:.2f},{:.2f},{:.2f}) [cm]".format(traj[idx,3],traj[idx,4],traj[idx,5]), (8,22), bbox={'facecolor': 'white', 'alpha': 0.5})]
        anim3 = [ax3.imshow(img_fw[idx], animated=True), ax3.annotate("fwd warped, (tx,ty,tz) = ({:.2f},{:.2f},{:.2f}) [cm]".format(traj[idx,3],traj[idx,4],traj[idx,5]), (8,22), bbox={'facecolor': 'white', 'alpha': 0.5})]
        anim4 = [ax4.imshow(noccs[idx], cmap=cmap, animated=True, vmax=1, vmin=0), ax4.annotate("occlusion, (tx,ty,tz) = ({:.2f},{:.2f},{:.2f}) [cm]".format(traj[idx,3],traj[idx,4],traj[idx,5]), (8,22), bbox={'facecolor': 'white', 'alpha': 0.5})]
        anim5 = [ax5.imshow(vis_flow(inv_flows[idx]), animated=True), ax5.annotate("F_inv, (tx,ty,tz) = ({:.2f},{:.2f},{:.2f}) [cm]".format(traj[idx,3],traj[idx,4],traj[idx,5]), (8,22), bbox={'facecolor': 'white', 'alpha': 0.5})]
        anim6 = [ax6.imshow(vis_flow(fwd_flows[idx]), animated=True), ax6.annotate("F_fwd, (tx,ty,tz) = ({:.2f},{:.2f},{:.2f}) [cm]".format(traj[idx,3],traj[idx,4],traj[idx,5]), (8,22), bbox={'facecolor': 'white', 'alpha': 0.5})]
        anims.append(anim1 + anim2 + anim3 + anim4 + anim5 + anim6)

    ani = animation.ArtistAnimation(fig, anims, interval=200, blit=False, repeat_delay=20)
    fig.tight_layout(); 
    # fig.colorbar(anim1[0], ax=ax1); fig.colorbar(anim2[0], ax=ax2); fig.colorbar(anim3[0], ax=ax3); fig.colorbar(anim5[0], ax=ax5); fig.colorbar(anim6[0], ax=ax6);
    print("=> Save GIF");
    ani.save(file_name, writer='imagemagick', fps=10, dpi=100)
    plt.close('all')


def fwd_warp(im, fwd_flow, upscale=2):
    im = F.interpolate(im, scale_factor=upscale)
    fwd_flow = F.interpolate(fwd_flow, scale_factor=upscale) * upscale
    bb, _, hh, ww = fwd_flow.size()
    i_range = torch.arange(0, hh).view(1, hh, 1).expand(1, hh, ww).type_as(fwd_flow)  # [1, H, W]
    j_range = torch.arange(0, ww).view(1, 1, ww).expand(1, hh, ww).type_as(fwd_flow)  # [1, H, W]
    pixel_uv = torch.stack((j_range, i_range), dim=1)   # [1, 2, H, W]
    flow_uv = pixel_uv + fwd_flow

    coo = flow_uv.reshape(2,-1)
    v_im = im.reshape(3,-1).permute(1,0)
    idx = coo.long()[[1,0]]
    idx[0][idx[0]<0] = hh
    idx[0][idx[0]>hh-1] = hh
    idx[1][idx[1]<0] = ww
    idx[1][idx[1]>ww-1] = ww

    idx = idx / upscale
    hh = int(hh / upscale)
    ww = int(ww / upscale)

    _idx, _val = coalesce(idx, v_im, m=hh+1, n=ww+1, op='mean')
    w_rgb = torch.sparse.FloatTensor(_idx, _val, torch.Size([hh+1,ww+1,3])).to_dense()[:-1,:-1]
    w_val = (1- (torch.sparse.FloatTensor(_idx, _val, torch.Size([hh+1,ww+1,3])).to_dense()[:-1,:-1]==0).float()).prod(dim=2)
    # pdb.set_trace()

    return (w_rgb*w_val.unsqueeze(-1)).detach().cpu().numpy().clip(-1,1) * 0.5 + 0.5, w_val.detach().cpu().numpy()


def inv_warp(im, inv_flow):
    bb, _, hh, ww = inv_flow.size()
    i_range = torch.arange(0, hh).view(1, hh, 1).expand(1, hh, ww).type_as(inv_flow)  # [1, H, W]
    j_range = torch.arange(0, ww).view(1, 1, ww).expand(1, hh, ww).type_as(inv_flow)  # [1, H, W]
    pixel_uv = torch.stack((j_range, i_range), dim=1)   # [1, 2, H, W]
    flow_uv = (pixel_uv + inv_flow).permute(0,2,3,1)    # [1, H, W, 2]

    flow_uv[:,:,:,0] = flow_uv[:,:,:,0].sub(ww/2).div(ww/2)
    flow_uv[:,:,:,1] = flow_uv[:,:,:,1].sub(hh/2).div(hh/2)

    if np.array(torch.__version__[:3]).astype(float) >= 1.3:
        w_rgb = F.grid_sample(im, flow_uv, align_corners=True)
    else:
        w_rgb = F.grid_sample(im, flow_uv)

    w_rgb = w_rgb[0].permute(1,2,0)
    w_val = (flow_uv[0].abs().max(dim=-1)[0] <= 1).float()
    # pdb.set_trace()

    return (w_rgb*w_val.unsqueeze(-1)).detach().cpu().numpy().clip(-1,1) * 0.5 + 0.5, w_val.detach().cpu().numpy()


def flow_warp(img, flow):
    # img: b x c x h x w
    # flo: b x 2 x h x w
    bs, _, gh, gw = img.size()
    mgrid_np = np.expand_dims(np.mgrid[0:gw,0:gh].transpose(0,2,1).astype(np.float32),0).repeat(bs, axis=0)
    mgrid = torch.from_numpy(mgrid_np).cuda()
    grid = mgrid.add(flow).permute(0,2,3,1)

    # grid = mgrid.add(flo12_.div(self.downscale*(2**i))).transpose(1,2).transpose(2,3)
    #                     # bx2x80x160 -> bx80x2x160 -> bx80x160x2
    grid[:,:,:,0] = grid[:,:,:,0].sub(gw/2).div(gw/2)
    grid[:,:,:,1] = grid[:,:,:,1].sub(gh/2).div(gh/2)
    # pdb.set_trace()
    
    if np.array(torch.__version__[:3]).astype(float) >= 1.3:
        img_w = F.grid_sample(img, grid, align_corners=True)
    else:
        img_w = F.grid_sample(img, grid)

    # pdb.set_trace()
    return img_w


def L2_norm(x, dim=1, keepdim=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset, dim=dim, keepdim=True)
    return l2_norm



def main():
    global best_error, n_iter, device
    args = parser.parse_args()
    if args.dataset_format == 'stacked':
        from datasets.stacked_sequence_folders import SequenceFolder
    elif args.dataset_format == 'sequential':
        from datasets.sj_sequence_folders import SequenceFolder
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    args.save_path = 'demo_results'/Path(args.name)/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    # Data loading
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize
    ])
    valid_transform = custom_transforms.Compose(
        [custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        max_demi=args.max_demi
    )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    # create model
    print("=> creating model")

    if args.rotation_mode == 'quaternion':
        dim_motion = 7
    elif args.rotation_mode in ['euler', '6D']:
        dim_motion = 6

    if args.two_way_flow:
        ch_pred = 4
    else:
        ch_pred = 2

    sf_net = getattr(models, args.sfnet)(dim_motion=dim_motion, ch_pred=ch_pred).to(device)

    if args.pretrained_sf:
        print("=> using pre-trained weights for SFNet")
        weights = torch.load(args.pretrained_sf)
        sf_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        sf_net.init_weights()

    cudnn.benchmark = True
    sf_net = torch.nn.DataParallel(sf_net)

    _demo = demo(args, train_loader, sf_net)

@torch.no_grad()
def demo(args, train_loader, sf_net):
    global device
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    sf_net.eval()

    end = time.time()

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, r2t_poses, t2r_poses) in enumerate(train_loader):
        # if i > 5: break;

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        r2t_poses = [r2t_pose.to(device) for r2t_pose in r2t_poses]
        t2r_poses = [t2r_pose.to(device) for t2r_pose in t2r_poses]

        r2t_poses_c = convert_pose(r2t_poses, output_rot_mode=args.rotation_mode)
        t2r_poses_c = convert_pose(t2r_poses, output_rot_mode=args.rotation_mode)

        if args.check_flows:
            print("=> Start debugging"); pdb.set_trace();
            continue;
            
            bb = 0
            sq = 0
            alpha = 3.0; beta = 0.05;
            alpha1 = 0.01; alpha2 = 0.5;
            tgt = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            flows_on_tgt = sf_net(tgt_img, t2r_poses_c[sq])        # {t}->{t-1}
            inv_flow_t2r = flows_on_tgt[:,:2]
            fwd_flow_r2t = flows_on_tgt[:,2:]
            inv_flow_arr_t2r = inv_flow_t2r.detach().cpu().numpy()[bb].transpose(1,2,0)
            fwd_flow_arr_r2t = fwd_flow_r2t.detach().cpu().numpy()[bb].transpose(1,2,0)

            ref = ref_imgs[sq][bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            flows_on_ref = sf_net(ref_imgs[sq], r2t_poses_c[sq])   # {t}->{t-1}
            inv_flow_r2t = flows_on_ref[:,:2]
            fwd_flow_t2r = flows_on_ref[:,2:]
            inv_flow_arr_r2t = inv_flow_r2t.detach().cpu().numpy()[bb].transpose(1,2,0)
            fwd_flow_arr_t2r = fwd_flow_t2r.detach().cpu().numpy()[bb].transpose(1,2,0)
            
            ##################################
            """ 1) Occlusion by inverse flows """
            inv_tr2rt_flow = flow_warp(inv_flow_t2r, inv_flow_r2t)
            inv_rt2tr_flow = flow_warp(inv_flow_r2t, inv_flow_t2r)

            inv_r2t_flow_diff = torch.abs(inv_tr2rt_flow + inv_flow_r2t)
            inv_t2r_flow_diff = torch.abs(inv_rt2tr_flow + inv_flow_t2r)

            # inv_r2t_consist_bound = torch.max(beta * L2_norm(inv_flow_r2t), torch.Tensor([alpha]).cuda())
            # inv_t2r_consist_bound = torch.max(beta * L2_norm(inv_flow_t2r), torch.Tensor([alpha]).cuda())
            inv_r2t_consist_bound = alpha1 * (L2_norm(inv_flow_r2t) + L2_norm(inv_tr2rt_flow)) + alpha2
            inv_t2r_consist_bound = alpha1 * (L2_norm(inv_flow_t2r) + L2_norm(inv_rt2tr_flow)) + alpha2

            inv_noc_mask_ref = (L2_norm(inv_t2r_flow_diff) < inv_t2r_consist_bound).type(torch.FloatTensor).cuda()
            inv_noc_mask_tgt = (L2_norm(inv_r2t_flow_diff) < inv_r2t_consist_bound).type(torch.FloatTensor).cuda()

            inv_noc_mask_ref_arr = inv_noc_mask_ref[bb,0].detach().cpu().numpy()
            inv_noc_mask_tgt_arr = inv_noc_mask_tgt[bb,0].detach().cpu().numpy()

            ##################################
            """ 2) Occlusion by forward flows """
            fwd_tr2rt_flow = flow_warp(fwd_flow_t2r, fwd_flow_r2t)
            fwd_rt2tr_flow = flow_warp(fwd_flow_r2t, fwd_flow_t2r)

            fwd_r2t_flow_diff = torch.abs(fwd_tr2rt_flow + fwd_flow_r2t)
            fwd_t2r_flow_diff = torch.abs(fwd_rt2tr_flow + fwd_flow_t2r)

            # fwd_r2t_consist_bound = torch.max(beta * L2_norm(fwd_flow_r2t), torch.Tensor([alpha]).cuda())
            # fwd_t2r_consist_bound = torch.max(beta * L2_norm(fwd_flow_t2r), torch.Tensor([alpha]).cuda())
            fwd_r2t_consist_bound = alpha1 * (L2_norm(fwd_flow_r2t) + L2_norm(fwd_tr2rt_flow)) + alpha2
            fwd_t2r_consist_bound = alpha1 * (L2_norm(fwd_flow_t2r) + L2_norm(fwd_rt2tr_flow)) + alpha2

            fwd_noc_mask_ref = (L2_norm(fwd_t2r_flow_diff) < fwd_t2r_consist_bound).type(torch.FloatTensor).cuda()
            fwd_noc_mask_tgt = (L2_norm(fwd_r2t_flow_diff) < fwd_r2t_consist_bound).type(torch.FloatTensor).cuda()

            fwd_noc_mask_ref_arr = fwd_noc_mask_ref[bb,0].detach().cpu().numpy()
            fwd_noc_mask_tgt_arr = fwd_noc_mask_tgt[bb,0].detach().cpu().numpy()

            ##############################
            """ 3) Occlusion by tgt flows """
            tgt_tr2rt_flow = flow_warp(inv_flow_t2r, fwd_flow_r2t)
            tgt_rt2tr_flow = flow_warp(fwd_flow_r2t, inv_flow_t2r)

            tgt_r2t_flow_diff = torch.abs(tgt_tr2rt_flow + fwd_flow_r2t)
            tgt_t2r_flow_diff = torch.abs(tgt_rt2tr_flow + inv_flow_t2r)

            # tgt_r2t_consist_bound = torch.max(beta * L2_norm(fwd_flow_r2t), torch.Tensor([alpha]).cuda())
            # tgt_t2r_consist_bound = torch.max(beta * L2_norm(inv_flow_t2r), torch.Tensor([alpha]).cuda())
            tgt_r2t_consist_bound = alpha1 * (L2_norm(fwd_flow_r2t) + L2_norm(tgt_tr2rt_flow)) + alpha2
            tgt_t2r_consist_bound = alpha1 * (L2_norm(inv_flow_t2r) + L2_norm(tgt_rt2tr_flow)) + alpha2

            tgt_noc_mask_ref = (L2_norm(tgt_t2r_flow_diff) < tgt_t2r_consist_bound).type(torch.FloatTensor).cuda()
            tgt_noc_mask_tgt = (L2_norm(tgt_r2t_flow_diff) < tgt_r2t_consist_bound).type(torch.FloatTensor).cuda()

            tgt_noc_mask_ref_arr = tgt_noc_mask_ref[bb,0].detach().cpu().numpy()
            tgt_noc_mask_tgt_arr = tgt_noc_mask_tgt[bb,0].detach().cpu().numpy()

            ##############################
            """ 4) Occlusion by ref flows """
            ref_tr2rt_flow = flow_warp(fwd_flow_t2r, inv_flow_r2t)
            ref_rt2tr_flow = flow_warp(inv_flow_r2t, fwd_flow_t2r)

            ref_r2t_flow_diff = torch.abs(ref_tr2rt_flow + inv_flow_r2t)
            ref_t2r_flow_diff = torch.abs(ref_rt2tr_flow + fwd_flow_t2r)

            # ref_r2t_consist_bound = torch.max(beta * L2_norm(inv_flow_r2t), torch.Tensor([alpha]).cuda())
            # ref_t2r_consist_bound = torch.max(beta * L2_norm(fwd_flow_t2r), torch.Tensor([alpha]).cuda())
            ref_r2t_consist_bound = alpha1 * (L2_norm(inv_flow_r2t) + L2_norm(ref_tr2rt_flow)) + alpha2
            ref_t2r_consist_bound = alpha1 * (L2_norm(fwd_flow_t2r) + L2_norm(ref_rt2tr_flow)) + alpha2

            ref_noc_mask_ref = (L2_norm(ref_t2r_flow_diff) < ref_t2r_consist_bound).type(torch.FloatTensor).cuda()
            ref_noc_mask_tgt = (L2_norm(ref_r2t_flow_diff) < ref_r2t_consist_bound).type(torch.FloatTensor).cuda()

            ref_noc_mask_ref_arr = ref_noc_mask_ref[bb,0].detach().cpu().numpy()
            ref_noc_mask_tgt_arr = ref_noc_mask_tgt[bb,0].detach().cpu().numpy()


            plt.close('all');
            fig = plt.figure(1, figsize=(20, 13))
            ea1 = 5; ea2 = 4; ii = 1;
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(inv_flow_arr_t2r[:,:,0]); plt.colorbar(); plt.text(0, 20, "(tgt→) inv_flow_arr_t2r[0]", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(inv_flow_arr_t2r[:,:,1]); plt.colorbar(); plt.text(0, 20, "(tgt→) inv_flow_arr_t2r[1]", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(fwd_flow_arr_r2t[:,:,0]); plt.colorbar(); plt.text(0, 20, "(tgt→) fwd_flow_arr_r2t[0]", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(fwd_flow_arr_r2t[:,:,1]); plt.colorbar(); plt.text(0, 20, "(tgt→) fwd_flow_arr_r2t[1]", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(inv_flow_arr_r2t[:,:,0]); plt.colorbar(); plt.text(0, 20, "(ref→) inv_flow_arr_r2t[0]", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(inv_flow_arr_r2t[:,:,1]); plt.colorbar(); plt.text(0, 20, "(ref→) inv_flow_arr_r2t[1]", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(fwd_flow_arr_t2r[:,:,0]); plt.colorbar(); plt.text(0, 20, "(ref→) fwd_flow_arr_t2r[0]", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(fwd_flow_arr_t2r[:,:,1]); plt.colorbar(); plt.text(0, 20, "(ref→) fwd_flow_arr_t2r[1]", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(inv_noc_mask_tgt_arr); plt.colorbar(); plt.text(0, 20, "inv_noc_mask_tgt_arr", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(inv_noc_mask_ref_arr); plt.colorbar(); plt.text(0, 20, "inv_noc_mask_ref_arr", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(fwd_noc_mask_tgt_arr); plt.colorbar(); plt.text(0, 20, "fwd_noc_mask_tgt_arr", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(fwd_noc_mask_ref_arr); plt.colorbar(); plt.text(0, 20, "fwd_noc_mask_ref_arr", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(tgt_noc_mask_tgt_arr); plt.colorbar(); plt.text(0, 20, "tgt_noc_mask_tgt_arr", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(tgt_noc_mask_ref_arr); plt.colorbar(); plt.text(0, 20, "tgt_noc_mask_ref_arr", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(ref_noc_mask_tgt_arr); plt.colorbar(); plt.text(0, 20, "ref_noc_mask_tgt_arr", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(ref_noc_mask_ref_arr); plt.colorbar(); plt.text(0, 20, "ref_noc_mask_ref_arr", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(tgt); plt.colorbar(); plt.text(0, 20, "tgt", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(ref); plt.colorbar(); plt.text(0, 20, "ref", bbox={'facecolor': 'yellow', 'alpha': 0.5}); plt.grid(linestyle=':', linewidth=0.4);
            plt.tight_layout(); plt.ion(); plt.show();
            


            plt.figure(2); plt.imshow(L2_norm(tgt_r2t_flow_diff)[bb,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()
            plt.figure(3); plt.imshow(L2_norm(tgt_t2r_flow_diff)[bb,0].detach().cpu()); plt.colorbar(); plt.ion(); plt.show()

            # img_iw = inv_warp(tgt_img, inv_flow_t2r)[bb]
            # img_fw = inv_warp(tgt_img, fwd_flow_r2t)[bb]
            # f_inv_t2r = vis_flow(inv_flow_arr_t2r)
            # f_fwd_t2r = vis_flow(fwd_flow_arr_t2r)


        im = imread('/seokju/EuRoC_MAV_448/MH_01_0/1403636639213555456.jpg').astype(np.float32)
        # im = imread('/seokju/EuRoC_MAV_448/MH_01_1/1403636757863555584.jpg').astype(np.float32)
        # im = imread('/seokju/EuRoC_MAV_448/MH_01_1/1403636752863555584.jpg').astype(np.float32)
        # im = imread('/seokju/EuRoC_MAV_448/V1_01_0/1403715401762142976.jpg').astype(np.float32)
        # im = imread('/seokju/EuRoC_MAV_448/V1_01_0/1403715398812143104.jpg').astype(np.float32)
        im = np.transpose(im, (2, 0, 1))
        im = torch.from_numpy(im).float()/255
        im = im.sub(0.5).div(0.5).unsqueeze(0).to(device)

        img = im[0].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5

        """Generate virtual trajectory via 3D parametric curve"""

        tt = np.pi * np.arange(0, 2, 0.05)
        txs = 0.3 * np.sin(tt)
        tys = 0.2 * np.sin(tt)
        tzs = 0.3 * np.sin(tt)
        # traj = np.array([[0,0,0,tx,0,0] for tx in txs]).astype(np.float32)
        # traj = np.array([[0,0,0,0,ty,0] for ty in tys]).astype(np.float32)
        # traj = np.array([[0,0,0,0,0,tz] for tz in tzs]).astype(np.float32)
        # traj = np.array([[0,0,0,tx,ty,0] for tx, ty in zip(txs,tys)]).astype(np.float32)
        # traj = np.array([[0,0,0,0,0,tz] for tz in tzs] + [[0,0,0,tx,0,0] for tx in txs]).astype(np.float32)
        traj = np.array([[0,0,0,tx,0,0] for tx in txs] + [[0,0,0,0,ty,0] for ty in tys]).astype(np.float32)
        
        # traj = []
        # for th in np.linspace(0., 1 * np.pi, 30):
        #     tx = 0.15 * np.sin(1*th) * np.cos(4*th)
        #     ty = 0.15 * np.sin(1*th) * np.sin(4*th)
        #     tz = 0.25 * (-np.cos(th) + 1)
        #     traj.append([0, 0, 0, tx, ty, tz])
        # for th in np.linspace(1/2 * np.pi, 2 * np.pi, 20):
        #     tz = 0.5 * np.sin(th)
        #     traj.append([0, 0, 0, 0, 0, tz])
        # traj = np.array(traj).astype(np.float32)


        inv_fx, inv_fy = [], []
        fwd_fx, fwd_fy = [], []
        img_fw, img_iw = [], []
        inv_flows, fwd_flows = [], []
        noccs = []

        # alpha1, alpha2 = 0.02, 1.0
        # alpha1, alpha2 = 0.03, 0.5
        alpha1, alpha2 = 0.01, 0.5

        if args.fwd_flow:
            for pose in traj:
                pose = torch.from_numpy(pose).to(device).unsqueeze(0)
                fwd_flow = sf_net(im, pose)
                fwd_flow_arr = fwd_flow.detach().cpu().numpy()
                fwd_fx.append( fwd_flow_arr[0,0] )
                fwd_fy.append( fwd_flow_arr[0,1] )
                img_fw.append( fwd_warp(im, fwd_flow, upscale=3)[0] )
            save_animation('./temp_fwd.gif', img, img_fw, fwd_fx, fwd_fy, traj, cmap='inferno', gap=1)
        elif args.two_way_flow:
            for pose in traj:
                pose = torch.from_numpy(pose).to(device).unsqueeze(0)
                flows = sf_net(im, pose)
                inv_flow = flows[:,:2]
                fwd_flow = flows[:,2:]
                inv_flow_arr = inv_flow.detach().cpu().numpy()
                fwd_flow_arr = fwd_flow.detach().cpu().numpy()
                inv_fx.append( inv_flow_arr[0,0] )
                inv_fy.append( inv_flow_arr[0,1] )
                fwd_fx.append( fwd_flow_arr[0,0] )
                fwd_fy.append( fwd_flow_arr[0,1] )
                img_iw.append( inv_warp(im, inv_flow)[0] )
                img_fw.append( fwd_warp(im, fwd_flow, upscale=2)[0] )
                inv_flows.append( inv_flow_arr[0].transpose(1,2,0) )
                fwd_flows.append( fwd_flow_arr[0].transpose(1,2,0) )

                """ generate non-occluded mask """
                # flow_inv2fwd = flow_warp(inv_flow, fwd_flow)
                # flow_fwd2inv = flow_warp(fwd_flow, inv_flow)
                # flow_inv2fwd_diff = torch.abs(flow_inv2fwd + fwd_flow)
                # flow_fwd2inv_diff = torch.abs(flow_fwd2inv + inv_flow)
                # inv2fwd_consist_bound = alpha1 * (L2_norm(fwd_flow) + L2_norm(flow_inv2fwd)) + alpha2
                # fwd2inv_consist_bound = alpha1 * (L2_norm(inv_flow) + L2_norm(flow_fwd2inv)) + alpha2
                # pred_noc_mask = (L2_norm(flow_fwd2inv_diff) < fwd2inv_consist_bound).type(torch.FloatTensor).cuda()
                # curr_noc_mask = (L2_norm(flow_inv2fwd_diff) < inv2fwd_consist_bound).type(torch.FloatTensor).cuda()
                # noccs.append(pred_noc_mask[0,0].detach().cpu().numpy())

                """ generate flow_diff mask """
                inv2fwd_flow = flow_warp(inv_flow, fwd_flow)
                fwd2inv_flow = flow_warp(fwd_flow, inv_flow)
                fwd_flow_diff_norm = ( L2_norm(torch.abs(fwd_flow + inv2fwd_flow)) / L2_norm(torch.abs(fwd_flow) + torch.abs(inv2fwd_flow)).clamp(min=1e-6) ).clamp(0,1)
                inv_flow_diff_norm = ( L2_norm(torch.abs(inv_flow + fwd2inv_flow)) / L2_norm(torch.abs(fwd_flow) + torch.abs(inv2fwd_flow)).clamp(min=1e-6) ).clamp(0,1)
                noccs.append(inv_flow_diff_norm[0,0].detach().cpu().numpy())


            # save_animation('./temp_inv.gif', img, img_iw, inv_fx, inv_fy, traj, cmap='inferno', gap=1)
            # save_animation('./temp_fwd.gif', img, img_fw, fwd_fx, fwd_fy, traj, cmap='inferno', gap=1)
            save_animation2('./anim2.gif', img, img_iw, img_fw, noccs, inv_flows, fwd_flows, traj, cmap='bone', gap=1)

        pdb.set_trace()

    return 0



def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1/disp for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [1/disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def convert_pose(poses, output_rot_mode="quaternion"):
    '''
        input: pairwse "pose" [bs x 4 x 4 ]
    '''
    output_poses = []

    if output_rot_mode == "quaternion":
        for pose in poses:
            qw = torch.sqrt(1.0 + pose[:,0,0] + pose[:,1,1] + pose[:,2,2]) / 2.0
            qw = torch.max(qw , torch.zeros(pose.size(0)).cuda()+1e-8) #batch
            qx = (pose[:,2,1] - pose[:,1,2]) / (4.0 * qw)
            qy = (pose[:,0,2] - pose[:,2,0]) / (4.0 * qw)
            qz = (pose[:,1,0] - pose[:,0,1]) / (4.0 * qw)
            quat = torch.cat((qw.view(-1,1), qx.view(-1,1), qy.view(-1,1), qz.view(-1,1)), dim=1)
            output_pose = torch.cat((quat, pose[:,:3,3]), dim=1)
            output_poses.append(output_pose)

    if output_rot_mode == "euler":
        for pose in poses:

            sy = torch.sqrt(pose[:,0,0]*pose[:,0,0]+pose[:,1,0]*pose[:,1,0])
            singular= sy<1e-6
            singular=singular.float()
            
            x=torch.atan2(pose[:,2,1], pose[:,2,2])
            y=torch.atan2(-pose[:,2,0], sy)
            z=torch.atan2(pose[:,1,0],pose[:,0,0])
            
            xs=torch.atan2(-pose[:,1,2], pose[:,1,1])
            ys=torch.atan2(-pose[:,2,0], sy)
            zs=pose[:,1,0]*0
            
            out_euler=torch.zeros(pose.size(0),3).cuda()
            out_euler[:,0]=x*(1-singular)+xs*singular
            out_euler[:,1]=y*(1-singular)+ys*singular
            out_euler[:,2]=z*(1-singular)+zs*singular
            # out_euler = out_euler * 180 / np.pi

            output_pose = torch.cat((out_euler, pose[:,:3,3]), dim=1)
            output_poses.append(output_pose)
            
    return output_poses
    

def compute_inv_flow(sf_net, tgt_img, ref_imgs, r2t_poses, t2r_poses):
    r2t_flows = []
    for ref_img, r2t_pose in zip(ref_imgs, r2t_poses):
        r2t_flows.append( sf_net(ref_img, r2t_pose) )

    t2r_flows = []
    for t2r_pose in t2r_poses:
        t2r_flows.append( sf_net(tgt_img, t2r_pose) )

    return r2t_flows, t2r_flows


def compute_fwd_flow(sf_net, tgt_img, ref_imgs, r2t_poses, t2r_poses):
    t2r_flows = []
    for ref_img, r2t_pose in zip(ref_imgs, r2t_poses):
        t2r_flows.append( sf_net(ref_img, r2t_pose) )

    r2t_flows = []
    for t2r_pose in t2r_poses:
        r2t_flows.append( sf_net(tgt_img, t2r_pose) )

    return r2t_flows, t2r_flows


def compute_two_way_flow(sf_net, tgt_img, ref_imgs, r2t_poses, t2r_poses):
    """About two-way-flow"""
    '''
        Input: single frame -> 4ch output inv/fwd flows
        
        [1st two xy-channels] inv_r2t_flows: {(I_ref, P_t2r) input -> F_r2t} => {(I_ref, P_r2t) input -> F_r2t}
        [2nd two xy-channels] fwd_t2r_flows: {(I_ref, P_r2t) input -> F_t2r}
        
        [1st two xy-channels] inv_t2r_flows: {(I_tgt, P_r2t) input -> F_t2r} => {(I_tgt, P_t2r) input -> F_t2r}
        [2nd two xy-channels] fwd_r2t_flows: {(I_tgt, P_t2r) input -> F_r2t}
        
    '''
    inv_r2t_flows, fwd_t2r_flows = [], []
    inv_t2r_flows, fwd_r2t_flows = [], []

    for ref_img, r2t_pose in zip(ref_imgs, r2t_poses):
        outputs = sf_net(ref_img, r2t_pose)
        inv_flows = [output[:,:2] for output in outputs]
        fwd_flows = [output[:,2:] for output in outputs]
        inv_r2t_flows.append( inv_flows )
        fwd_t2r_flows.append( fwd_flows )

    for t2r_pose in t2r_poses:
        outputs = sf_net(tgt_img, t2r_pose)
        inv_flows = [output[:,:2] for output in outputs]
        fwd_flows = [output[:,2:] for output in outputs]
        inv_t2r_flows.append( inv_flows )
        fwd_r2t_flows.append( fwd_flows )

    return inv_r2t_flows, fwd_t2r_flows, inv_t2r_flows, fwd_r2t_flows


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv


def compute_pose(pose_net, tgt_img, ref_imgs):
    poses = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))

    return poses


if __name__ == '__main__':
    main()
