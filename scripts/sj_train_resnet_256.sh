# --pretrained-disp /seokju/Insta_DM/checkpoints/resnet50-19c8e357.pth \


CUDA_VISIBLE_DEVICES=0,1,2,3 TRAIN_SET=/seokju/KITTI/kitti_256/
python train.py $TRAIN_SET \
--dispnet DispResNet \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 3 \
--with-mask \
--with-ssim \
--with-gt \
--name resnet_256