# dataset settings
dataset_type = 'PVDataset' # pv
data_root = 'data/PVdataset/' # 数据集文件夹路径
crop_size = (512, 512) # 图片尺寸
train_pipeline = [
    dict(type='LoadTiffImageFromFile'), #把加载数据的方法换成自己上面实现的LoadTiffImageFromFile
    dict(type='LoadAnnotations'), # 读取标签
    dict(
        type='RandomResize', # 随机更改大小
        scale=(1024, 1024), # 短边改大小
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), # 裁剪固定大小
    dict(type='RandomFlip', prob=0.5), # 随机翻转
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadTiffImageFromFile'), #把加载数据的方法换成自己上面实现的LoadTiffImageFromFile
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadTiffImageFromFile', backend_args=None), #把加载数据的方法换成自己上面实现的LoadTiffImageFromFile
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root, # 文件夹目录
        data_prefix=dict(
            img_path='images/training', # 图像目录
            seg_map_path='annotations/training'), # 标签目录
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation', #图像目录
            seg_map_path='annotations/validation'), #标签目录
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
