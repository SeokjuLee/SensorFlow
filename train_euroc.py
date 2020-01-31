import argparse
import time
import csv
from path import Path
import datetime

import numpy as np
import torch
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
parser.add_argument('--fwd-warp', action='store_true', help='forward-warp mode or not')


best_error = -1
n_iter = 0
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global best_error, n_iter, device
    args = parser.parse_args()
    if args.dataset_format == 'stacked':
        from datasets.stacked_sequence_folders import SequenceFolder
    elif args.dataset_format == 'sequential':
        # from datasets.sequence_folders import SequenceFolder
        from datasets.sj_sequence_folders import SequenceFolder
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    args.save_path = 'checkpoints'/Path(args.name)/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path)

    # Data loading
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])
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

    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if args.with_gt:
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            args.data,
            transform=valid_transform
        )
    else:
        val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            max_demi=args.max_demi
        )
    print('{} samples found in {} train scenes'.format(
        len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(
        len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    if args.rotation_mode == 'quaternion':
        sf_net = getattr(models, args.sfnet)(dim_motion=7).to(device)
    elif args.rotation_mode in ['euler', '6D']:
        sf_net = getattr(models, args.sfnet)(dim_motion=6).to(device)

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

    print('=> setting adam solver')

    optim_params = [
        {'params': sf_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['photo_loss', 'smooth_loss', 'flow_loss', 'train_loss'])
        # writer.writerow(['photo_loss', 'smooth_loss', 'train_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(
        len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    if not args.debug_mode:
        if args.pretrained_sf:
            logger.reset_valid_bar()
            if args.with_gt:
                errors, error_names = validate_with_gt(
                    args, val_loader, sf_net, 0, logger)
            else:
                errors, error_names = validate_without_gt(args, val_loader, sf_net, 0, logger)
            for error, name in zip(errors, error_names):
                training_writer.add_scalar(name, error, 0)
            error_string = ', '.join('{} : {:.3f}'.format(name, error)
                                     for name, error in zip(error_names[2:9], errors[2:9]))
            logger.valid_writer.write(' * Avg {}'.format(error_string))

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, sf_net, disp_net, optimizer, args.epoch_size, logger, training_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, sf_net, epoch, logger)
        else:
            errors, error_names = validate_without_gt(args, val_loader, sf_net, epoch, logger)
        error_string = ', '.join('{} : {:.3f}'.format(name, error)
                                 for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        # save_checkpoint(
        #     epoch,
        #     args.save_path, {
        #         'epoch': epoch + 1,
        #         'state_dict': sf_net.module.state_dict()
        #     }, {
        #         'epoch': epoch + 1,
        #         'state_dict': pose_net.module.state_dict()
        #     },
        #     is_best)
        save_checkpoint(
            epoch,
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': sf_net.module.state_dict()
            },
            is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, sf_net, disp_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.flow_loss_weight
    # w1, w2 = args.photo_loss_weight, args.smooth_loss_weight

    # switch to train mode
    sf_net.train()
    disp_net.eval()

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, r2t_poses, t2r_poses) in enumerate(train_loader):
        # if i > 5: break;

        log_losses = i > 0 and n_iter % args.print_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        r2t_poses = [r2t_pose.to(device) for r2t_pose in r2t_poses]
        t2r_poses = [t2r_pose.to(device) for t2r_pose in t2r_poses]

        r2t_poses_c = convert_pose(r2t_poses, output_rot_mode=args.rotation_mode)
        t2r_poses_c = convert_pose(t2r_poses, output_rot_mode=args.rotation_mode)

        # pdb.set_trace()
        '''
            plt.close('all')
            bb = 0
            aaa = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            bbb = ref_imgs[0][bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            ccc = ref_imgs[1][bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            plt.figure(1); plt.imshow(aaa); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            plt.figure(2); plt.imshow(bbb); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            plt.figure(3); plt.imshow(ccc); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();

        '''

        # compute output
        tgt_depth = 1/disp_net(tgt_img).detach()
        ref_depths = [1/disp_net(ref_img).detach() for ref_img in ref_imgs]

        if args.fwd_warp:
            r2t_flows, t2r_flows = compute_fwd_flow(sf_net, tgt_img, ref_imgs, r2t_poses_c, t2r_poses_c)
        else:
            r2t_flows, t2r_flows = compute_flow(sf_net, tgt_img, ref_imgs, r2t_poses_c, t2r_poses_c)
        # poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        # pdb.set_trace()
        '''
            ### dpoint ###

            plt.close('all')
            bb = 1
            aaa = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            bbb = r2t_flows[0][0][bb,0].detach().cpu().numpy()
            plt.figure(1); plt.imshow(aaa); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            plt.figure(2); plt.imshow(bbb, cmap='plasma'); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();

            
            plt.close('all')
            bb = 1
            aaa = tgt_img[bb].detach().cpu().numpy().transpose(1,2,0) * 0.5 + 0.5
            bbb = tgt_depth[bb,0].detach().cpu().numpy()
            plt.figure(1); plt.imshow(aaa); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            plt.figure(2); plt.imshow(bbb, cmap='plasma'); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            
            
            plt.figure(2); plt.imshow(bbb, cmap='plasma', vmax=0.5); plt.colorbar(); plt.tight_layout(); plt.ion(); plt.show();
            aaa = pose_vec2mat(poses[0], rotation_mode='euler')
        '''

        loss_1 = compute_photo_loss(tgt_img, ref_imgs, r2t_flows, t2r_flows, args)
        loss_2 = compute_flow_smooth_loss(r2t_flows, tgt_img, t2r_flows, ref_imgs)
        loss_3 = compute_rigid_flow_loss(tgt_img, ref_imgs, r2t_flows, t2r_flows, tgt_depth, ref_depths, r2t_poses, t2r_poses, intrinsics, args)

        # loss = w1*loss_1 + w2*loss_2
        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        if log_losses:
            train_writer.add_scalar('photo_loss', loss_1.item(), n_iter)
            train_writer.add_scalar('smooth_loss', loss_2.item(), n_iter)
            train_writer.add_scalar('flow_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss_1.item(), loss_2.item(), loss_3.item(), loss.item()])
            # writer.writerow([loss_1.item(), loss_2.item(), loss.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write(
                'Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]



@torch.no_grad()
def validate_without_gt(args, val_loader, sf_net, epoch, logger):
    global device
    batch_time = AverageMeter()
    # losses = AverageMeter(i=4, precision=4)
    losses = AverageMeter(i=3, precision=4)

    # w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.flow_loss_weight
    w1, w2 = args.photo_loss_weight, args.smooth_loss_weight

    # switch to evaluate mode
    sf_net.eval()

    end = time.time()
    logger.valid_bar.update(0)

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

        # compute output
        r2t_flows = []
        for ref_img, r2t_pose in zip(ref_imgs, r2t_poses):
            r2t_flows.append( [sf_net(ref_img, r2t_pose)] )

        t2r_flows = []
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

        loss_1 = compute_photo_loss(tgt_img, ref_imgs, r2t_flows, t2r_flows, args)
        loss_2 = compute_flow_smooth_loss(r2t_flows, tgt_img, t2r_flows, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()

        # loss = w1*loss_1 + w2*loss_2 + w3*loss_3
        loss = w1*loss_1 + w2*loss_2
        # losses.update([loss, loss_1, loss_2, loss_3])
        losses.update([loss, loss_1, loss_2])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))
    # return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss']
    return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss']



@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, logger):
    global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)

        # compute output
        output_disp = disp_net(tgt_img)
        output_depth = 1/output_disp[:, 0]

        errors.update(compute_errors(depth, output_depth))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(
                batch_time, errors.val[0], errors.avg[0]))
    logger.valid_bar.update(len(val_loader))
    return errors.avg, error_names


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
    

def compute_flow(sf_net, tgt_img, ref_imgs, r2t_poses, t2r_poses):
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
