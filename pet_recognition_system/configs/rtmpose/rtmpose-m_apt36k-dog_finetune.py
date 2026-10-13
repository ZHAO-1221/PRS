"""P1: fine-tune the official RTMPose-m AP-10K model on the APT-36K dog subset.

Everything not set here (model, codec, augmentation) comes from the official AP-10K config.
Data and the starting weights are read from the local folder $PET_DATA_ROOT;
tools/prepare_apt36k_split.py writes the two annotation files.
"""
import os

_base_ = 'mmpose::animal_2d_keypoint/rtmpose/ap10k/rtmpose-m_8xb64-210e_ap10k-256x256.py'

data_root = os.environ['PET_DATA_ROOT'].replace('\\', '/').rstrip('/')
load_from = data_root + '/models/rtmpose-m_ap10k/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth'

# a short, gentle schedule: a tenth of the official learning rate, 40 epochs instead of 210
max_epochs = 40
stage2_num_epochs = 10  # the last epochs train with the lighter augmentation, as in the official schedule
base_lr = 4e-4

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-3, by_epoch=False, begin=0, end=50),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin=max_epochs // 2, end=max_epochs,
         T_max=max_epochs // 2, by_epoch=True, convert_to_iter_based=True),
]

# load_from replaces every weight, so skip downloading the ImageNet/COCO backbone
model = dict(backbone=dict(init_cfg=None))

train_dataloader = dict(
    num_workers=4,
    dataset=dict(data_root=data_root, ann_file='apt36k_dog_train.json', data_prefix=dict(img='')))
val_dataloader = dict(
    num_workers=4,
    dataset=dict(data_root=data_root, ann_file='apt36k_dog_val.json', data_prefix=dict(img='')))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + '/apt36k_dog_val.json')
test_evaluator = val_evaluator

custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49),
    dict(type='mmdet.PipelineSwitchHook', switch_epoch=max_epochs - stage2_num_epochs,
         switch_pipeline=_base_.train_pipeline_stage2),
]
