import argparse
import os, sys
from SensorData import SensorData

import scipy.misc
import numpy as np
from pebble import ProcessPool
from tqdm import tqdm
from path import Path
import cv2
import imageio
import pdb

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--input_path', required=True, help='path to output folder')
parser.add_argument('--output_path', required=True, help='path to output folder')
parser.add_argument('--width', type=int, default=640, help="output image height")
parser.add_argument('--height', type=int, default=480, help="output image height")
parser.add_argument('--num_threads', type=int, default=8, help="number of threads to use")
opt = parser.parse_args()
print(opt)


def dump_example(opt, scene):
    sensor_file = str( scene + '/' + scene.basename() + '.sens' )
    sd = SensorData(sensor_file)
    sd.save_images(output_path=os.path.join(opt.output_path, scene.basename()), image_size=[opt.height, opt.width], frame_skip=3)
    sd.save_poses(output_path=os.path.join(opt.output_path, scene.basename()), frame_skip=3)

def main():
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    dataset_dir = Path(opt.input_path)
    scenes = dataset_dir.dirs()
    n_scenes = len(scenes)
    print('Found {} potential scenes'.format(n_scenes))
    print('Retrieving frames')

    if opt.num_threads == 1:
        for scene in tqdm(scenes):
            dump_example(opt, scene)
    else:
        with ProcessPool(max_workers=opt.num_threads) as pool:
            tasks = pool.map(dump_example, [opt]*n_scenes, scenes)
            try:
                for _ in tqdm(tasks.result(), total=n_scenes):
                    pass
            except KeyboardInterrupt as e:
                tasks.cancel()
                raise e

    print('Generating train val lists')
    np.random.seed(8964)
    subdirs = Path(opt.output_path).dirs()
    dirnames = set([subdir.basename() for subdir in subdirs])
    with open(Path(opt.output_path) / 'train.txt', 'w') as tf:
        with open(Path(opt.output_path) / 'val.txt', 'w') as vf:
            for dn in tqdm(dirnames):
                if np.random.random() < 0.1:
                    vf.write('{}\n'.format(dn))
                else:
                    tf.write('{}\n'.format(dn))
    # pdb.set_trace()

    # # load the data
    # sys.stdout.write('loading %s...' % opt.filename)
    # sd = SensorData(opt.filename)
    # sys.stdout.write('loaded!\n')
    # if opt.export_depth_images:
    #     sd.export_depth_images(os.path.join(opt.output_path, 'depth'))
    # if opt.export_color_images:
    #     sd.export_color_images(os.path.join(opt.output_path, 'color'))
    # if opt.export_poses:
    #     sd.export_poses(os.path.join(opt.output_path, 'pose'))
    # if opt.export_intrinsics:
    #     sd.export_intrinsics(os.path.join(opt.output_path, 'intrinsic'))


if __name__ == '__main__':
    main()