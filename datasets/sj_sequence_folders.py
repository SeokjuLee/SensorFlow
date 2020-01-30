import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from pyquaternion import Quaternion

import pdb


def load_as_float(path):
    return imread(path).astype(np.float32)


def rot2eul(R):
    beta = -np.arcsin(R[2,0]) * 180 / np.pi
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta)) * 180 / np.pi
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta)) * 180 / np.pi
    return np.array((alpha, beta, gamma))


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, demi_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        # scene_list_path = self.root/'tmp.txt' if train else self.root/'val.txt'
        # scene_list_path = self.root/'val.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(demi_length)

    def crawl_folders(self, demi_length):
        sequence_set = []
        '''
            demi_length = 3
            shifts = list(range(-demi_length, demi_length + 1))
            shifts.pop(demi_length)
        '''
        # demi_length = (sequence_length-1)//2
        # shifts = list(range(-demi_length, demi_length + 1))
        # shifts.pop(demi_length)
        shifts = [-demi_length, demi_length]

        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
            with open(scene/'poses.txt') as f:
                poses = f.readlines()
            # with open(scene/'body_poses.txt') as f:
            #     body_poses = f.readlines()

            if len(imgs) < demi_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                # sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [], 
                #           'tgt_pose': np.array(poses[i].split(' ')).astype(np.float32), 'ref_poses': [],
                #           'tgt_body_pose': np.array(body_poses[i].split(' ')).astype(np.float32), 'ref_body_poses': []}
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [], 
                          'tgt_pose': np.array(poses[i].split(' ')).astype(np.float32), 'ref_poses': []}  
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                    sample['ref_poses'].append(np.array(poses[i+j].split(' ')).astype(np.float32))
                    # sample['ref_body_poses'].append(np.array(body_poses[i+j].split(' ')).astype(np.float32))
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        
        tgt_pose = np.vstack((sample['tgt_pose'].reshape(3,4), [0, 0, 0, 1])).astype(np.float32)
        ref_poses = [np.vstack((pose.reshape(3,4), [0, 0, 0, 1])).astype(np.float32) for pose in sample['ref_poses']]

        # r2t_poses = [tgt_pose @ np.linalg.inv(ref_pose) for ref_pose in ref_poses]
        # t2r_poses = [ref_pose @ np.linalg.inv(tgt_pose) for ref_pose in ref_poses]

        # r2t_poses = [np.linalg.inv(ref_pose) @ tgt_pose for ref_pose in ref_poses]
        # t2r_poses = [np.linalg.inv(tgt_pose) @ ref_pose for ref_pose in ref_poses]
        
        # r2t_poses = [np.linalg.inv(tgt_pose) @ ref_pose for ref_pose in ref_poses]
        # t2r_poses = [np.linalg.inv(ref_pose) @ tgt_pose for ref_pose in ref_poses]
        
        # r2t_poses = [ref_pose @ np.linalg.inv(tgt_pose) for ref_pose in ref_poses]
        # t2r_poses = [tgt_pose @ np.linalg.inv(ref_pose) for ref_pose in ref_poses]

        # r2t_poses = [np.linalg.inv(np.linalg.inv(ref_pose) @ tgt_pose) for ref_pose in ref_poses]
        # t2r_poses = [np.linalg.inv(np.linalg.inv(tgt_pose) @ ref_pose) for ref_pose in ref_poses]

        ### correct poses ###
        # r2t_poses = [np.linalg.inv(np.linalg.inv(tgt_pose) @ ref_pose) for ref_pose in ref_poses]
        # t2r_poses = [np.linalg.inv(np.linalg.inv(ref_pose) @ tgt_pose) for ref_pose in ref_poses]

        ### correct poses ###
        r2t_poses = [np.linalg.inv(ref_pose) @ tgt_pose for ref_pose in ref_poses]
        t2r_poses = [np.linalg.inv(tgt_pose) @ ref_pose for ref_pose in ref_poses]
        # pdb.set_trace()
        '''
            plt.figure(1); plt.imshow(tgt_img/255); plt.colorbar(); plt.ion(); plt.show();

            plt.close('all')
            aaa = tgt_img/255
            bbb = ref_imgs[0]/255
            ea1 = 1; ea2 = 2; ii = 1;
            fig = plt.figure(1, figsize=(14, 6))
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(aaa), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar(), plt.text(0, 20, "I_{t}", bbox={'facecolor': 'yellow', 'alpha': 0.5});
            fig.add_subplot(ea1,ea2,ii); ii += 1;
            plt.imshow(bbb), plt.grid(linestyle=':', linewidth=0.4), plt.colorbar(), plt.text(0, 20, "I_{t-1}", bbox={'facecolor': 'yellow', 'alpha': 0.5});
            fig.tight_layout(), plt.ion(), plt.show()
            

            plt.close('all')
            fig = plt.figure(1)
            imgs = []
            imgs.append([plt.imshow(ref_imgs[0]/255, animated=True)])
            imgs.append([plt.imshow(tgt_img/255, animated=True)])
            imgs.append([plt.imshow(ref_imgs[1]/255, animated=True)])
            ani = animation.ArtistAnimation(fig, imgs, interval=500, blit=False, repeat_delay=20)
            fig.tight_layout(), plt.ion(), plt.show()
            
            v1x = r2t_poses[0][0,3]
            v1y = r2t_poses[0][1,3]
            v1z = r2t_poses[0][2,3]
            v2x = t2r_poses[1][0,3]
            v2y = t2r_poses[1][1,3]
            v2z = t2r_poses[1][2,3]
            print("{:.4f}\t{:.4f}\t{:.4f}\n{:.4f}\t{:.4f}\t{:.4f}\n".format(v1x, v1y, v1z, v2x, v2y, v2z))
            print("{}\n{}".format(rot2eul(r2t_poses[0][:3,:3]), rot2eul(t2r_poses[1][:3,:3])))
            
            

            p1 = sample['ref_body_poses'][0][:3]
            p2 = sample['tgt_body_pose'][:3]
            p3 = sample['ref_body_poses'][1][:3]
            print("{}\n{}".format(p2 - p1, p3 - p2))
            

        '''

        if self.transform is not None:
            imgs, intrinsics, poses = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']), r2t_poses + t2r_poses)
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
            r2t_poses = poses[0:len(ref_imgs)]
            t2r_poses = poses[len(ref_imgs):len(ref_imgs)+len(ref_imgs)]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), r2t_poses, t2r_poses

    def __len__(self):
        return len(self.samples)
