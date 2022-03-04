checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None#'https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'

resume_from =None# './work_dirs/yolo/latest.pth'
workflow = [('train', 1)]
model = dict(
    type='PsDetector',
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(5,),#没有多尺度，只有一个头，对于点来说多尺度有没有必要值得后续实验
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    bbox_head=dict(
        type='PsdHead',
        in_channels=1024, #对于512的输入，backbone输出为1024通道
        stacked_convs=2,
        feat_channels=512,
))
                 
#其他类型的数据增强未经测试，可能有bug
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadPsdAnnotations'),
   
    dict(type='ResizeforPsd', img_scale=[(512, 512),], test_mode=False,),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['img']),
    dict(type='Collect_psd', keys=['img', 'gt_bboxes'])
]
    
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    # dict(type='LoadPsdAnnotations'),
   
    dict(type='ResizeforPsd', img_scale=[(512, 512),], test_mode=True,),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['img']),
    dict(type='Collect_psd', keys=['img',])
]

'''数据集的标签事先汇总到单个json里面，格式如下：
import os
import mmcv
import json
anns = list()
root = '/mnt/disk2/FedPSDet/Server/label/test/'
for file in os.listdir(root):
    if file.endswith(".json"):
        sample = {}
        sample['filename'] = os.path.splitext(file)[0]+'.jpg'
        sample['width'] = 600
        sample['height'] = 600
        sample['ann'] = mmcv.load(root+file)
        anns.append(sample)
with open('./test_psd.json', 'w') as file:
    json.dump(anns, file)
    
    训练集需要手动使用DMPS原始的增强方法增强一下，增强后的标签坐标为相对坐标，每个坐标是个6维向量，含义分别为：
 0: confidence, 1: point_shape, 2: offset_x, 3: offset_y, 4: cos(direction), 5: sin(direction)
    测试集使用原始数据格式，即一个字典，里面有'marks'等键值。
'''
dataset_type = 'MarkingDataset'
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(#训练集的图像和标签是经过DMPR-PS增强后的，注意他的标签格式与测试集不同
        type = dataset_type,
        img_prefix='/home/sushangchao/huwei3_det/DMPR-PS/dataset/train/train',
        ann_file='/home/sushangchao/mmdetection/train_psd.json',
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(#测试集的图像和标签都是原始的
        type = dataset_type,
        img_prefix='/mnt/disk2/FedPSDet/Server/test',
        ann_file='/home/sushangchao/mmdetection/test_psd.json',
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type = dataset_type,
        img_prefix='/mnt/disk2/FedPSDet/Server/test',
        ann_file='/home/sushangchao/mmdetection/test_psd.json',
        pipeline=test_pipeline,
        test_mode=True,
    ))

#Adam可以，SGD会很难收敛
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(
#     policy='step',
#     step=[2, 3])
# runner = dict(type='EpochBasedRunner', max_epochs=4)

# DMPR源码中不用step，这里暂时也不用
lr_config = dict(
    policy='step',
    step=[50000,])
runner = dict(type='IterBasedRunner', max_iters=5000)

checkpoint_config = dict(interval=1000)
evaluation = dict(interval=500,)
gpu_ids = range(0,8)


