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
                    choices=['SFNet', 'SFResNet'], help='depth network architecture.')
parser.add_argument('--num-scales', '--number-of-scales',
                    type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float,
                    help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float,
                    help='weight for flow smoothness loss', metavar='W', default=0.1)
parser.add_argument('-f', '--flow-loss-weight', type=float,
                    help='weight for flow loss', metavar='W', default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float,
                    help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('--with-ssim', action='store_true', help='use ssim loss',)
parser.add_argument('--with-mask', action='store_true',
                    help='use the the mask for handling moving objects and occlusions')
parser.add_argument('--name', dest='name', type=str, required=True,
                    help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--debug-mode', action='store_true', help='debug mode or not')
parser.add_argument('--rotation-mode', dest='rotation_mode', type=str, default='quaternion', choices=['quaternion', 'euler', '6D'], help='encoding rotation mode')
parser.add_argument('--fwd-flow', action='store_true', help='forward-flow mode or not')
parser.add_argument('--two-way-flow', action='store_true', help='two-way-flow mode or not')


best_error = -1
n_iter = 0
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


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
    ani.save(file_name, writer='imagemagick', fps=10, dpi=100)
    plt.close('all')


def fwd_warp(im, fwd_flow, upscale=2):
    im = F.interpolate(im, scale_factor=upscale)
    fwd_flow = F.interpolate(fwd_flow, scale_factor=upscale) * upscale
    bb, _, hh, ww = fwd_flow.size()
    i_range = torch.arange(0, hh).view(1, hh, 1).expand(1, hh, ww).type_as(fwd_flow)  # [1, H, W]
    j_range = torch.arange(0, ww).view(1, 1, ww).expand(1, hh, ww).type_as(fwd_flow)  # [1, H, W]
    pixel_uv = torch.stack((j_range, i_range), dim=1)  # [1, 2, H, W]
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
    w_val =  1- (torch.sparse.FloatTensor(_idx, _val, torch.Size([hh+1,ww+1,3])).to_dense()[:-1,:-1]==0).float()

    return w_rgb.detach().cpu().numpy() * 0.5 + 0.5, w_val.detach().cpu().numpy()


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
    # train_transform = custom_transforms.Compose([
    #     custom_transforms.RandomHorizontalFlip(),
    #     custom_transforms.RandomScaleCrop(),
    #     custom_transforms.ArrayToTensor(),
    #     normalize
    # ])
    # train_transform = custom_transforms.Compose([
    #     custom_transforms.RandomHorizontalFlip(),
    #     custom_transforms.ArrayToTensor(),
    #     normalize
    # ])
    # train_transform = custom_transforms.Compose([
    #     custom_transforms.RandomScaleCrop(),
    #     custom_transforms.ArrayToTensor(),
    #     normalize
    # ])
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
    val_set = SequenceFolder(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        train=False,
        max_demi=args.max_demi,
        proportion=10
    )
    print('{} samples found in {} train scenes'.format(
        len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(
        len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

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
    disp_net = getattr(models, 'DispResNet')().to(device)

    if args.pretrained_sf:
        print("=> using pre-trained weights for SFNet")
        weights = torch.load(args.pretrained_sf)
        sf_net.load_state_dict(weights['state_dict'], strict=False)
        # Sf_net.load_state_dict(weights, strict=False)
    else:
        sf_net.init_weights()

    if args.pretrained_disp:
        print("=> using pre-trained weights for DispNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=True)
    else:
        disp_net.init_weights()

    cudnn.benchmark = True
    sf_net = torch.nn.DataParallel(sf_net)
    disp_net = torch.nn.DataParallel(disp_net)

    train_loss = train(args, train_loader, sf_net, disp_net)
    test_loss  = validate_without_gt(args, val_loader, sf_net, epoch)

@torch.no_grad()
def train(args, train_loader, sf_net, disp_net):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.flow_loss_weight
    # w1, w2 = args.photo_loss_weight, args.smooth_loss_weight

    # switch to train mode
    sf_net.eval()
    disp_net.eval()

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


        # im = imread('/seokju/EuRoC_MAV_448/MH_01_0/1403636639213555456.jpg').astype(np.float32)
        # im = imread('/seokju/EuRoC_MAV_448/MH_01_1/1403636757863555584.jpg').astype(np.float32)
        # im = imread('/seokju/EuRoC_MAV_448/MH_01_1/1403636752863555584.jpg').astype(np.float32)
        # im = imread('/seokju/EuRoC_MAV_448/V1_01_0/1403715401762142976.jpg').astype(np.float32)
        im = imread('/seokju/EuRoC_MAV_448/V1_01_0/1403715398812143104.jpg').astype(np.float32)
        im = np.transpose(im, (2, 0, 1))
        im = torch.from_numpy(im).float()/255
        im = im.sub(0.5).div(0.5).unsqueeze(0).to(device)

        img = im[0].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5

        # txs = np.arange(0, 0.1, 0.01).tolist() + np.arange(0.1, 0, -0.01).tolist() 

        # tt = np.pi * np.arange(0, 2, 0.05)
        # txs = 0.3 * np.sin(tt)
        # tys = 0.3 * np.sin(tt)
        # tzs = 0.3 * np.sin(tt)
        # # traj = np.array([[0,0,0,tx,0,0] for tx in txs]).astype(np.float32)
        # # traj = np.array([[0,0,0,0,ty,0] for ty in tys]).astype(np.float32)
        # # traj = np.array([[0,0,0,0,0,tz] for tz in tzs]).astype(np.float32)
        # # traj = np.array([[0,0,0,tx,ty,0] for tx, ty in zip(txs,tys)]).astype(np.float32)
        # traj = np.array([[0,0,0,0,0,tz] for tz in tzs] + [[0,0,0,tx,0,0] for tx in txs]).astype(np.float32)

        """Generate virtual trajectory via 3D parametric curve"""
        traj = []
        # for th in np.linspace(0., 2 * np.pi, 60):
        #     tx = 0.2 * np.sin(2*th) * np.cos(3*th)
        #     ty = 0.2 * np.sin(2*th) * np.sin(3*th)
        #     tz = 0.2 * (-np.cos(th) + 1)
        #     traj.append([0, 0, 0, tx, ty, tz])
        # for th in np.linspace(0., 2 * np.pi, 40):
        #     tx = 0.2 * np.sin(1*th) * np.cos(2*th)
        #     ty = 0.2 * np.sin(1*th) * np.sin(2*th)
        #     tz = 0.2 * (-np.cos(th) + 1)
        #     traj.append([0, 0, 0, tx, ty, tz])
        for th in np.linspace(0., 1 * np.pi, 30):
            tx = 0.15 * np.sin(1*th) * np.cos(4*th)
            ty = 0.15 * np.sin(1*th) * np.sin(4*th)
            tz = 0.25 * (-np.cos(th) + 1)
            traj.append([0, 0, 0, tx, ty, tz])
        for th in np.linspace(1/2 * np.pi, 2 * np.pi, 20):
            tz = 0.5 * np.sin(th)
            traj.append([0, 0, 0, 0, 0, tz])
        traj = np.array(traj).astype(np.float32)
        # pdb.set_trace()

        fwd_fx, fwd_fy = [], []
        inv_fx, inv_fy = [], []
        img_fw, img_iw = [], []

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
                img_iw.append( fwd_warp(im, fwd_flow)[0] )
                img_fw.append( fwd_warp(im, fwd_flow)[0] )
            save_animation('./temp_fwd.gif', img, img_fw, inv_fx, inv_fy, traj, cmap='inferno', gap=1)
            save_animation('./temp_inv.gif', img, img_iw, fwd_fx, fwd_fy, traj, cmap='inferno', gap=1)


        pdb.set_trace()

        traj = []
        for th in np.linspace(0., 2 * np.pi, 80+1):
            tx = np.sin(2*th) * np.cos(3*th)
            ty = np.sin(2*th) * np.sin(3*th)
            tz = np.cos(th) + 1
            traj.append([0, 0, 0, tx, ty, tz])
        traj = np.array(traj).astype(np.float32)


        save_animation('./temp.gif', img, fwd_fx, fwd_fy, traj, cmap='inferno', gap=1)

        bb, _, hh, ww = fwd_flow.size()
        i_range = torch.arange(0, hh).view(1, hh, 1).expand(1, hh, ww).type_as(fwd_flow)  # [1, H, W]
        j_range = torch.arange(0, ww).view(1, 1, ww).expand(1, hh, ww).type_as(fwd_flow)  # [1, H, W]
        pixel_uv = torch.stack((j_range, i_range), dim=1)  # [1, 2, H, W]
        flow_uv = pixel_uv + fwd_flow

        coo = flow_uv.reshape(2,-1)
        v_im = im.reshape(3,-1).permute(1,0)
        idx = coo.long()[[1,0]]
        idx[0][idx[0]<0] = hh
        idx[0][idx[0]>hh-1] = hh
        idx[1][idx[1]<0] = ww
        idx[1][idx[1]>ww-1] = ww

        _idx, _val = coalesce(idx, v_im, m=hh+1, n=ww+1, op='mean')
        w_rgb = torch.sparse.FloatTensor(_idx, _val, torch.Size([hh+1,ww+1,3])).to_dense()[:-1,:-1]
        w_val =  1- (torch.sparse.FloatTensor(_idx, _val, torch.Size([hh+1,ww+1,3])).to_dense()[:-1,:-1]==0).float()

        plt.close('all')
        aaa = w_rgb.detach().cpu().numpy() * 0.5 + 0.5
        bbb = w_val.detach().cpu().numpy()
        plt.figure(1); plt.imshow(aaa), plt.colorbar(), plt.ion(), plt.show()
        plt.figure(2); plt.imshow(bbb), plt.colorbar(), plt.ion(), plt.show()


        # plt.close('all')
        # fig = plt.figure(1, figsize=(22, 4))
        # ax1 = fig.add_subplot(1,4,1)
        # ax2 = fig.add_subplot(1,4,2)
        # ax3 = fig.add_subplot(1,4,3)
        # fx_max, fx_min = np.array(fx).max(), np.array(fx).min()
        # fy_max, fy_min = np.array(fy).max(), np.array(fy).min()
        # anims = []
        # for idx in range(0, traj.shape[0], 1):
        #     anim1 = [ax1.imshow(fx[idx], animated=True, vmax=fx_max, vmin=fx_min), ax1.annotate("tx: {:.4f}".format(traj[idx,3]), (8,26), bbox={'facecolor': 'silver', 'alpha': 0.5})]
        #     anim2 = [ax2.imshow(fy[idx], animated=True, vmax=fy_max, vmin=fy_min), ax2.annotate("ty: {:.4f}".format(traj[idx,4]), (8,26), bbox={'facecolor': 'silver', 'alpha': 0.5})]
        #     anim3 = [ax3.imshow(img, animated=True), ax3.annotate("input image", (8,26), bbox={'facecolor': 'silver', 'alpha': 0.5})]
        #     anims.append(anim1 + anim2 + anim3)

        # ani = animation.ArtistAnimation(fig, anims, interval=200, blit=False, repeat_delay=20)
        # fig.tight_layout(); fig.colorbar(anim1[0], ax=ax1); fig.colorbar(anim2[0], ax=ax2); fig.colorbar(anim3[0], ax=ax3); plt.ion(); plt.show();
        # ani.save('./exAnimation.gif', writer='imagemagick', fps=10, dpi=100)



        # fig = plt.figure(1)
        # anims = []
        # vmax = np.array(fx).max()
        # vmin = np.array(fx).min()
        # for idx in range(0, len(fx), 1): anims.append([plt.imshow(fx[idx], animated=True, vmax=vmax, vmin=vmin)])
        # ani = animation.ArtistAnimation(fig, anims, interval=200, blit=False, repeat_delay=20)
        # fig.tight_layout(), plt.colorbar(), plt.ion(), plt.show()

        # fig = plt.figure(2)
        # anims = []
        # vmax = np.array(fy).max()
        # vmin = np.array(fy).min()
        # for idx in range(0, len(fy), 1): anims.append([plt.imshow(fy[idx], animated=True, vmax=vmax, vmin=vmin)])
        # ani = animation.ArtistAnimation(fig, anims, interval=200, blit=False, repeat_delay=20)
        # fig.tight_layout(), plt.colorbar(), plt.ion(), plt.show()


        pdb.set_trace()
        '''
            anim.save('superpositionWaves.mp4', writer = 'ffmpeg', fps = 1, dpi=400,extra_args=['-vcodec', 'libx264'])

            plt.close('all')
            bb = 0
            aaa = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            bbb = ref_imgs[0][bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            ccc = ref_imgs[1][bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            plt.figure(1); plt.imshow(aaa); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            plt.figure(2); plt.imshow(bbb); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            plt.figure(3); plt.imshow(ccc); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();

            viz_filter(sf_net)

            aaa = sf_net.module.conv1[0].weight
            bbb = sf_net.module.conv1[2].weight
            ccc = sf_net.module.conv5[0].conv1.weight
            ddd = sf_net.module.conv7[0].conv2.weight
            eee = sf_net.module.conv7[2].conv2.weight
            p (aaa.abs()<0.001).sum()/(aaa.abs()<9999).sum().float()
            p (bbb.abs()<0.001).sum()/(bbb.abs()<9999).sum().float()
            p (ccc.abs()<0.001).sum()/(ccc.abs()<9999).sum().float()
            p (ddd.abs()<0.001).sum()/(ddd.abs()<9999).sum().float()
            p (eee.abs()<0.001).sum()/(eee.abs()<9999).sum().float()

        '''

        

        # compute output
        tgt_depth = 1/disp_net(tgt_img).detach()
        ref_depths = [1/disp_net(ref_img).detach() for ref_img in ref_imgs]
        # pdb.set_trace()

        """About two-way-flow"""
        '''
            Input: single frame -> 4ch output inv/fwd flows
            
            [1st two xy-channels] inv_r2t_flows: {(I_ref, P_t2r) input -> F_r2t} => {(I_ref, P_r2t) input -> F_r2t}
            [2nd two xy-channels] fwd_t2r_flows: {(I_ref, P_r2t) input -> F_t2r}
            
            [1st two xy-channels] inv_t2r_flows: {(I_tgt, P_r2t) input -> F_t2r} => {(I_tgt, P_t2r) input -> F_t2r}
            [2nd two xy-channels] fwd_r2t_flows: {(I_tgt, P_t2r) input -> F_r2t}
        
            compute_two_way_flow(sf_net, tgt_img, ref_imgs, r2t_poses, t2r_poses)
        '''
        if args.two_way_flow:
            inv_r2t_flows, fwd_t2r_flows, inv_t2r_flows, fwd_r2t_flows = compute_two_way_flow(sf_net, tgt_img, ref_imgs, r2t_poses_c, t2r_poses_c)
        elif args.fwd_flow:
            r2t_flows, t2r_flows = compute_fwd_flow(sf_net, tgt_img, ref_imgs, r2t_poses_c, t2r_poses_c)
        else:
            r2t_flows, t2r_flows = compute_inv_flow(sf_net, tgt_img, ref_imgs, r2t_poses_c, t2r_poses_c)

        # poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        pdb.set_trace()
        '''
            ### dpoint ###

            plt.close('all')
            bb = 1
            aaa = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            bbb = ref_imgs[0][bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            ccc = r2t_flows[0][0][bb,0].detach().cpu().numpy()
            plt.figure(1); plt.imshow(aaa); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            plt.figure(2); plt.imshow(bbb); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            plt.figure(3); plt.imshow(ccc, cmap='plasma'); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();

            
            plt.close('all')
            bb = 1
            aaa = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            bbb = tgt_depth[bb,0].detach().cpu().numpy()
            plt.figure(1); plt.imshow(aaa); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            plt.figure(2); plt.imshow(bbb, cmap='plasma'); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            
            
            plt.figure(2); plt.imshow(bbb, cmap='plasma', vmax=0.5); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            aaa = pose_vec2mat(poses[0], rotation_mode='euler')
        '''
        if args.two_way_flow:
            loss_1 = compute_photo_loss(tgt_img, ref_imgs, inv_r2t_flows, inv_t2r_flows, args)
            loss_1 += compute_photo_loss(tgt_img, ref_imgs, fwd_r2t_flows, fwd_t2r_flows, args)
            loss_2 = compute_flow_smooth_loss(inv_r2t_flows, tgt_img, inv_t2r_flows, ref_imgs)
            loss_2 += compute_flow_smooth_loss(fwd_r2t_flows, tgt_img, fwd_t2r_flows, ref_imgs)
            loss_3 = compute_rigid_flow_loss(tgt_img, ref_imgs, inv_r2t_flows, inv_t2r_flows, tgt_depth, ref_depths, r2t_poses, t2r_poses, intrinsics, args)
            loss_3 += compute_rigid_flow_loss(tgt_img, ref_imgs, fwd_r2t_flows, fwd_t2r_flows, tgt_depth, ref_depths, r2t_poses, t2r_poses, intrinsics, args)
        else:
            loss_1 = compute_photo_loss(tgt_img, ref_imgs, r2t_flows, t2r_flows, args)
            loss_2 = compute_flow_smooth_loss(r2t_flows, tgt_img, t2r_flows, ref_imgs)
            loss_3 = compute_rigid_flow_loss(tgt_img, ref_imgs, r2t_flows, t2r_flows, tgt_depth, ref_depths, r2t_poses, t2r_poses, intrinsics, args)

        # loss = w1*loss_1 + w2*loss_2
        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss_1.item(), loss_2.item(), loss_3.item(), loss.item()])
            # writer.writerow([loss_1.item(), loss_2.item(), loss.item()])

    return losses.avg[0]



@torch.no_grad()
def validate_without_gt(args, val_loader, sf_net):
    global device
    batch_time = AverageMeter()
    # losses = AverageMeter(i=4, precision=4)
    losses = AverageMeter(i=3, precision=4)

    # w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.flow_loss_weight
    w1, w2 = args.photo_loss_weight, args.smooth_loss_weight

    # switch to evaluate mode
    sf_net.eval()

    end = time.time()

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, r2t_poses, t2r_poses) in enumerate(val_loader):
        # if i > 5: break;
        
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)
        r2t_poses = [r2t_pose.to(device) for r2t_pose in r2t_poses]
        t2r_poses = [t2r_pose.to(device) for t2r_pose in t2r_poses]

        r2t_poses = convert_pose(r2t_poses, output_rot_mode=args.rotation_mode)
        t2r_poses = convert_pose(t2r_poses, output_rot_mode=args.rotation_mode)

        """ compute output """
        if args.two_way_flow:
            inv_r2t_flows, fwd_t2r_flows = [], []
            inv_t2r_flows, fwd_r2t_flows = [], []
            for ref_img, r2t_pose in zip(ref_imgs, r2t_poses):
                outputs = [sf_net(ref_img, r2t_pose)]
                inv_flows = [output[:,:2] for output in outputs]
                fwd_flows = [output[:,2:] for output in outputs]
                inv_r2t_flows.append( inv_flows )
                fwd_t2r_flows.append( fwd_flows )
            for t2r_pose in t2r_poses:
                outputs = [sf_net(tgt_img, t2r_pose)]
                inv_flows = [output[:,:2] for output in outputs]
                fwd_flows = [output[:,2:] for output in outputs]
                inv_t2r_flows.append( inv_flows )
                fwd_r2t_flows.append( fwd_flows )
        elif args.fwd_flow:
            r2t_flows, t2r_flows = [], []
            for t2r_pose in t2r_poses:
                r2t_flows.append( [sf_net(tgt_img, t2r_pose)] )
            for ref_img, r2t_pose in zip(ref_imgs, r2t_poses):
                t2r_flows.append( [sf_net(ref_img, r2t_pose)] )
        else:
            r2t_flows, t2r_flows = [], []
            for ref_img, r2t_pose in zip(ref_imgs, r2t_poses):
                r2t_flows.append( [sf_net(ref_img, r2t_pose)] )
            for t2r_pose in t2r_poses:
                t2r_flows.append( [sf_net(tgt_img, t2r_pose)] )

        '''
            plt.close('all')
            bb = 1
            aaa = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            bbb = 1/tgt_depth[0][bb,0].detach().cpu().numpy()
            plt.figure(1); plt.imshow(aaa); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            plt.figure(2); plt.imshow(bbb, cmap='plasma'); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();

        '''
        if args.two_way_flow:
            loss_1 = compute_photo_loss(tgt_img, ref_imgs, inv_r2t_flows, inv_t2r_flows, args)
            loss_1 += compute_photo_loss(tgt_img, ref_imgs, fwd_r2t_flows, fwd_t2r_flows, args)
            loss_2 = compute_flow_smooth_loss(inv_r2t_flows, tgt_img, inv_t2r_flows, ref_imgs)
            loss_2 += compute_flow_smooth_loss(fwd_r2t_flows, tgt_img, fwd_t2r_flows, ref_imgs)
        else:
            loss_1 = compute_photo_loss(tgt_img, ref_imgs, r2t_flows, t2r_flows, args)
            loss_2 = compute_flow_smooth_loss(r2t_flows, tgt_img, t2r_flows, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()

        loss = w1*loss_1 + w2*loss_2
        losses.update([loss, loss_1, loss_2])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    return losses.avg



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
