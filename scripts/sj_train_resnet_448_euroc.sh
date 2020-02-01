# --pretrained-disp /seokju/Insta_DM/checkpoints/resnet50-19c8e357.pth \

# --pretrained-disp /seokju/SC-SfMLearner-Release/checkpoints/resnet_448_euroc/01-22-01:40/dispnet_24_checkpoint.pth.tar \
# --pretrained-pose /seokju/SC-SfMLearner-Release/checkpoints/resnet_448_euroc/01-22-01:40/exp_pose_24_checkpoint.pth.tar \


# ### 200130 (211) ### 
# TRAIN_SET=/seokju/EuRoC_MAV_448/
# CUDA_VISIBLE_DEVICES=2,3 python train_euroc.py $TRAIN_SET \
# --sfnet SFResNet \
# --num-scales 1 \
# --demi-length 1 \
# -j 0 \
# -b 8 -s 0 --epoch-size 1000 --sequence-length 3 \
# --with-mask \
# --with-ssim \
# --pretrained-sf /seokju/SC-SfMLearner-Release/checkpoints/resnet_448_euroc/01-29-18:43/dispnet_60_checkpoint.pth.tar \
# --rotation-mode euler \
# --name resnet_448_euroc \


### 200130-v2 ### 
TRAIN_SET=/seokju/EuRoC_MAV_448/
CUDA_VISIBLE_DEVICES=2,3 python train_euroc.py $TRAIN_SET \
--sfnet SFResNet \
--num-scales 1 \
--max-demi 2 \
-b 4 -p 1 -s 0 -f 0 \
--epoch-size 1000 --sequence-length 3 \
--with-mask \
--with-ssim \
--fwd-warp \
--pretrained-disp ./pretrained/dispnet_60_checkpoint.pth.tar \
--rotation-mode euler \
--name resnet_448_euroc
