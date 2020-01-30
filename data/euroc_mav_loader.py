from __future__ import division
import numpy as np
from path import Path
import scipy.misc
from collections import Counter
import yaml
import cv2
import time
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from matplotlib import pyplot as plt
import pdb

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def pose_from_oxts_packet(metadata, scale):

    lat, lon, alt, roll, pitch, yaw = metadata
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    Taken from https://github.com/utiasSTARS/pykitti
    """

    er = 6378137.  # earth radius (approx.) in meters
    # Use a Mercator projection to get the translation vector
    ty = lat * np.pi * er / 180.

    tx = scale * lon * np.pi * er / 180.
    # ty = scale * er * \
    #     np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    t = np.array([tx, ty, tz]).reshape(-1,1)

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))
    return transform_from_rot_trans(R, t)


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def transform_from_pos_qtn(pos, qtn):
    '''
        Transforation matrix from position vector and quaternion.
        pos: px, py, pz [m]
        qtn: qw, qx, qy, qz
    '''
    # q_idx = [1,2,3,0]   # qw, qx, qy, qz -> qx, qy, qz, qw
    # qtn = qtn[q_idx]

    R = Quaternion(qtn).rotation_matrix     # R(3,3)
    t = pos.reshape(3,1)                    # t(3,1)

    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))



class EurocMavLoader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416):

        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.cam_ids = ['0', '1']
        self.date_list = ['MH_01', 'MH_02', 'MH_03', 'MH_04', 'MH_05', 'V1_01', 'V1_02', 'V1_03', 'V2_01', 'V2_02', 'V2_03']
        self.collect_train_folders()

    def collect_train_folders(self):
        self.scenes = []
        for date in self.date_list:
            drive_set = self.dataset_dir/date
            self.scenes.append(drive_set)

    def collect_scenes(self, drive):
        train_scenes = []
        for c in self.cam_ids:
            # pdb.set_trace()
            with open(drive + '/mav0/state_groundtruth_estimate0/data.csv') as f: 
                navs = f.readlines()
            with open(drive + '/mav0/cam' + c + '/data.csv') as f: 
                imgs = f.readlines()            
            # navs = drive + '/mav0/state_groundtruth_estimate0/data.csv'
            # oxts = sorted((drive/'mav0'/'state_groundtruth_estimate0').files('*.txt'))

            # scene_data = {'cid': c, 'dir': drive, 'frame_id': [], 'K':[], 'D':[], 'w':[], 'h':[], 'K_r':[], 'cam_pose':[], 'body_pos': [], 'body_qtn':[], 'rel_path': drive.name + '_' + c, 'from_img_dir': drive + '/mav0/cam' + c}
            scene_data = {'cid': c, 'dir': drive, 'frame_id': [], 'K':[], 'D':[], 'w':[], 'h':[], 'K_r':[], 'cam_pose':[], 'rel_path': drive.name + '_' + c, 'from_img_dir': drive + '/mav0/cam' + c}
            scale = None
            origin = None
            cam_calib = yaml.safe_load(open(drive + '/mav0/cam' + c + '/sensor.yaml', 'r'))
            imu_calib = yaml.safe_load(open(drive + '/mav0/imu0' + '/sensor.yaml', 'r'))
            if drive.name[0] == 'M':
                pos_calib = yaml.safe_load(open(drive + '/mav0/leica0' + '/sensor.yaml', 'r'))
            elif drive.name[0] == 'V':
                pos_calib = yaml.safe_load(open(drive + '/mav0/vicon0' + '/sensor.yaml', 'r'))

            fu, fv, cu, cv = cam_calib['intrinsics']
            scene_data['K'] = np.asarray([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])   # K(3,3)
            scene_data['D'] = np.asarray(cam_calib['distortion_coefficients'])    # D(4,1)
            scene_data['w'], scene_data['h'] = cam_calib['resolution']
            scene_data['K_r'], roi = cv2.getOptimalNewCameraMatrix(scene_data['K'], scene_data['D'], (scene_data['w'], scene_data['h']), 0)

            c2b = np.asarray(cam_calib['T_BS']['data']).reshape(4,4)
            i2b = np.asarray(imu_calib['T_BS']['data']).reshape(4,4)
            p2b = np.asarray(pos_calib['T_BS']['data']).reshape(4,4)

            if drive.name[0] == 'M':
                p2b[0][:-1] = [0,  0,  1]
                p2b[1][:-1] = [0, -1,  0]
                p2b[2][:-1] = [1,  0,  0]
            
            i2c = c2b @ np.linalg.inv(i2b)
            p2c = c2b @ np.linalg.inv(p2b)

            ### 이미지의 타임스탬프를 읽고 동일한 ID의 pose정보를 가져와! ###
            start = time.time()
            frame_list = []
            search_from = 0
            for img in imgs[1:]:
                ts = img.split(',')[0]
                frame_data = [[ts, idx, nav] for idx, nav in enumerate(navs[search_from:]) if ts[:-3] == nav[:16]]
                # frame_data = [[ts, nav] for nav in navs if ts[:-3] == nav[:16]]
                # frame_data = [[idx, nav] for idx, nav in enumerate(navs) if ts[:-3] == nav[:16]]
                # frame_data = [n for n, nav in enumerate(navs) if ts[:-3] == nav.split(',')[0][:-3]]
                # frame_data = [n for n, nav in enumerate(navs) if ts == nav.split(',')[0]]
                # frame_data = [n for n, nav in enumerate(navs) if ts[:-3] in nav]
                if frame_data:
                    search_from += frame_data[0][1]
                    frame_list.append(frame_data[0])
                    scene_data['frame_id'].append(ts)
            print("cam: \"{:}\", time: {:.3f} [sec]".format(drive + '/mav0/cam' + c, time.time() - start))
            # pdb.set_trace()

            ### 카메라의 실제 위치를 구해! ###
            for fdata in frame_list:
                metadata = np.array(fdata[2].split(',')).astype(float)
                pos = metadata[1:4]     # LEICA, VICON -> X, BODY is correct
                qtn = metadata[4:8] 
                # pdb.set_trace()

                ### {LEICA, VICON} -> {body}
                # pos_mat = transform_from_pos_qtn(pos, [1,0,0,0])
                # pos_body = np.linalg.inv(p2b) @ pos_mat
                # pos_p2b = (np.linalg.inv(p2b) @ np.hstack((pos, 1)).reshape(4,1)).reshape(4)[:3]
                # pos_p2b = (p2b @ np.hstack((pos, 1)).reshape(4,1)).reshape(4)[:3]

                # Rt_body = transform_from_pos_qtn(pos_p2b, qtn)
                # Rt_cam = c2b @ Rt_body
                # Rt_cam = np.linalg.inv(c2b) @ Rt_body

                # Rt_pos = transform_from_pos_qtn(pos, qtn)
                # Rt_bod = p2b @ Rt_pos
                # Rt_cam = np.linalg.inv(c2b) @ Rt_bod
                
                Rt_bod = transform_from_pos_qtn(pos, qtn)
                Rt_cam = np.linalg.inv( np.linalg.inv(c2b) @ np.linalg.inv(Rt_bod) )
                # pdb.set_trace()
                
                scene_data['cam_pose'].append(Rt_cam)
                # scene_data['body_pos'].append(pos_p2b)
                # scene_data['body_qtn'].append(qtn)
            # pdb.set_trace()

            ### 카메라의 실제 움직임을 구해! ###
            ### train_scene에 해당 정보를 모두 저장! ###

            train_scenes.append(scene_data)
        return train_scenes


    def get_scene_imgs(self, scene_data):
        def construct_sample(scene_data, i, frame_id):
            sample = {"img":self.load_image(scene_data, i), "id":frame_id}
            sample['pose'] = scene_data['cam_pose'][i]
            # sample['body_pose'] = np.hstack((scene_data['body_pos'][i], scene_data['body_qtn'][i]))
            return sample

        drive = str(scene_data['dir'].name)
        for (i,frame_id) in enumerate(scene_data['frame_id']):
            yield construct_sample(scene_data, i, frame_id)


    def load_image(self, scene_data, tgt_idx):
        img_file = scene_data['dir']/'mav0/cam{}'.format(scene_data['cid'])/'data'/scene_data['frame_id'][tgt_idx]+'.png'

        if not img_file.isfile():
            return None
        img = cv2.imread(img_file)
        # img = scipy.misc.imread(img_file)

        mapx, mapy = cv2.initUndistortRectifyMap(scene_data['K'], scene_data['D'], None, scene_data['K_r'], (img.shape[1], img.shape[0]), 5)
        img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        img = cv2.resize(img, (self.img_width, self.img_height))
        # img = scipy.misc.imresize(img, (self.img_height, self.img_width))
        # pdb.set_trace()
        '''
            plt.figure(1), plt.imshow(img), plt.colorbar(), plt.ion(), plt.show()
        '''

        return img


    def read_raw_calib_file(self, filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                        data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                        pass
        return data
