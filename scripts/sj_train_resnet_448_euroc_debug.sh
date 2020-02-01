# --pretrained-disp /seokju/Insta_DM/checkpoints/resnet50-19c8e357.pth \

# --pretrained-disp /seokju/SC-SfMLearner-Release/checkpoints/resnet_448_euroc/01-22-01:40/dispnet_21_checkpoint.pth.tar \
# --pretrained-pose /seokju/SC-SfMLearner-Release/checkpoints/resnet_448_euroc/01-22-01:40/exp_pose_21_checkpoint.pth.tar \

# --pretrained-disp /seokju/SC-SfMLearner-Release/checkpoints/resnet_448_euroc/01-22-01:40/dispnet_24_checkpoint.pth.tar \
# --pretrained-pose /seokju/SC-SfMLearner-Release/checkpoints/resnet_448_euroc/01-22-01:40/exp_pose_24_checkpoint.pth.tar \

# --pretrained-disp /seokju/SC-SfMLearner-Release/checkpoints/resnet_256/12-08-01:15/dispnet_model_best.pth.tar \
# --pretrained-pose /seokju/SC-SfMLearner-Release/checkpoints/resnet_256/12-08-01:15/exp_pose_model_best.pth.tar \

# --pretrained-disp /seokju/SC-SfMLearner-Release/checkpoints/resnet_448_euroc/01-28-23:18/dispnet_39_checkpoint.pth.tar \

# --pretrained-sf ./pretrained/dispnet_60_checkpoint.pth.tar \

# --pretrained-sf /data3/seokju/SensorFlow-v2-euroc/checkpoints/211/sfnet_checkpoint_q_ep7.pth.tar \

# --pretrained-sf /data3/seokju/SensorFlow-v2-euroc/checkpoints/211/sfnet_checkpoint_e_ep6.pth.tar \



### 200130 ### 
TRAIN_SET=/seokju/EuRoC_MAV_448/
CUDA_VISIBLE_DEVICES=5,7 python train_euroc.py $TRAIN_SET \
--sfnet SFResNet \
--num-scales 1 \
--max-demi 1 \
-j 0 \
-b 2 -p 1 -s 0.001 -f 0.5 \
--epoch-size 1000 --sequence-length 3 \
--with-mask \
--with-ssim \
--pretrained-disp ./pretrained/dispnet_60_checkpoint.pth.tar \
--rotation-mode euler \
--seed 5 \
--name resnet_448_euroc_debug \
--debug-mode