# The new config inherits a base config to highlight the necessary modification
_base_ = ['../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py']

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    pretrained='checkpoints/resnet50-19c8e357.pth',
    roi_head=dict(
        bbox_head=dict(num_classes=1)
        ),
    train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                gpu_assign_thr=50))))

albu_train_transforms = [
    dict(
        type='ColorJitter',
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussianBlur', blur_limit=(3, 5), p=1.0),
            dict(type='GaussNoise', var_limit=(5.0, 50.0), p=1.0)
        ],
        p=0.3),
    dict(type='ShiftScaleRotate',
        border_mode=0, p=0.5, shift_limit=(-0.1,0.1), rotate_limit=(0,0), scale_limit=(-0.2,0.1))
]

# pipelines
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 360), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=[0.5,0.5], direction=['horizontal', 'vertical']),
    dict(type='Albu', 
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'])),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 360),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('object',)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix='/DATASET/train_real/',
        classes=classes,
        pipeline=train_pipeline,
        ann_file='/DATASET/train_real/annotation_coco_300_finetune.json'),
    val=dict(
        type=dataset_type,
        img_prefix='/DATASET/test_real/',
        classes=classes,
        pipeline=test_pipeline,
        ann_file='/DATASET/test_real/annotation_coco_200.json'),
    test=dict(
        type=dataset_type,
        img_prefix='/DATASET/test_real/',
        classes=classes,
        pipeline=test_pipeline,
        ann_file='/DATASET/test_real/annotation_coco_200.json'))



# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
load_from = '/DATASET/mmdetection_work_dir/latest.pth'

optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0003)
#optimizer_config = dict(grad_clip=None)

lr_config = dict(
	policy='step',
	warmup='linear',
	warmup_iters=500,
	warmup_ratio=0.001,
	step=[10000])
#total_epochs=356 #400
runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=400) # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`
#log_config = dict(interval=100)
checkpoint_config = dict(interval=100)
#log_config = dict(
#	interval=1000,
#	hooks=[
#	dict(type='TextLoggerHook'),
#	dict(type='TensorboardLoggerHook')
#	])

workflow = [('train', 1), ('val',1)]
work_dir = 'DATASET/mmdetection_work_dir'
