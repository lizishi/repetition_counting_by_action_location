# runtime
checkpoint_config = dict(interval=20, by_epoch=True)
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ],
)
# runtime settings
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

num_queries = 40
clip_len = 32
stride_rate = 0.25

# include the contrastive epoch
total_epochs = 80

# model settings
model = dict(
    type="ReactBackbone",
    backbone=dict(
        type="ResNet",
        pretrained="torchvision://resnet50",
        depth=50,
    ),
    input_feat_dim=2048,
    num_class=2,
    feat_dim=256,
    n_head=8,
    encoder_sample_num=4,
    decoder_sample_num=4,
    num_encoder_layers=2,
    num_decoder_layers=4,
    num_queries=num_queries,
    clip_len=clip_len,
    stride_rate=stride_rate,
    test_bg_thershold=0.0,
    coef_l1=5.0,
    coef_iou=2.0,
    coef_ce=1.0,
    coef_aceenc=0.1,
    coef_acedec=1.0,
    coef_quality=1.0,
    coef_iou_decay=100.0,
)

# dataset settings
dataset_type = "RepCountDatasetE2E"
data_root_train = "/DATA/disk1/lizishi/LLSP/frames/train"
data_root_val = "/DATA/disk1/lizishi/LLSP/frames/valid"
data_root_test = "/DATA/disk1/lizishi/LLSP/frames/test"
flow_root_train = None
flow_root_val = None

ann_file_train = "/DATA/disk1/lizishi/LLSP/annotation/train_new.csv"
ann_file_val = "/DATA/disk1/lizishi/LLSP/annotation/valid_new.csv"
ann_file_test = "/DATA/disk1/lizishi/LLSP/annotation/test_new.csv"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False
)
frame_interval_list = [32]

test_pipeline = [
    dict(
        type="RawFrameDecode",
    ),
    dict(type="RandomRescale", scale_range=(256, 320)),
    dict(type="RandomCrop", size=256),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW", collapse=True),
    dict(
        type="Collect",
        keys=["imgs", "gt_bbox", "video_gt_box", "clip_len"],
        meta_name="video_meta",
        meta_keys=["video_name", "origin_snippet_num"],
    ),
    dict(
        type="ToTensor",
        keys=["imgs", "clip_len"],
    ),
    dict(
        type="ToDataContainer",
        fields=[
            dict(key="gt_bbox", stack=False, cpu_only=True),
            dict(key="imgs", stack=False, cpu_only=True),
            dict(key="video_gt_box", stack=False, cpu_only=True),
            dict(key="clip_len", stack=True, cpu_only=True),
        ],
    ),
]

train_pipeline = [
    dict(
        type="RawFrameDecode",
    ),
    dict(type="RandomRescale", scale_range=(256, 320)),
    dict(type="RandomCrop", size=256),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW", collapse=True),
    dict(
        type="Collect",
        keys=["imgs", "gt_bbox", "clip_len"],
        meta_name="video_meta",
        meta_keys=["video_name", "origin_snippet_num"],
    ),
    dict(
        type="ToTensor",
        keys=["imgs", "clip_len"],
    ),
    dict(
        type="ToDataContainer",
        fields=[
            dict(key="gt_bbox", stack=True, cpu_only=True),
            dict(key="imgs", stack=True, cpu_only=True),
        ],
    ),
]
val_pipeline = [
    dict(
        type="RawFrameDecode",
    ),
    dict(type="RandomRescale", scale_range=(256, 320)),
    dict(type="RandomCrop", size=256),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW", collapse=True),
    dict(
        type="Collect",
        keys=["imgs", "gt_bbox", "video_gt_box", "clip_len"],
        meta_name="video_meta",
        meta_keys=["video_name", "origin_snippet_num"],
    ),
    dict(
        type="ToTensor",
        keys=["imgs", "clip_len"],
    ),
    dict(
        type="ToDataContainer",
        fields=[
            dict(key="gt_bbox", stack=False, cpu_only=True),
            dict(key="imgs", stack=False, cpu_only=True),
            dict(key="video_gt_box", stack=False, cpu_only=True),
            dict(key="clip_len", stack=True, cpu_only=True),
        ],
    ),
]

data = dict(
    train_dataloader=dict(
        workers_per_gpu=4,
        videos_per_gpu=4,
        drop_last=False,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=4,
    ),
    val_dataloader=dict(
        workers_per_gpu=1,
        videos_per_gpu=1,
        pin_memory=True,
        shuffle=False,
    ),
    test_dataloader=dict(
        workers_per_gpu=1,
        videos_per_gpu=1,
        pin_memory=True,
        shuffle=False,
    ),
    test=dict(
        type=dataset_type,
        prop_file=ann_file_test,
        root_folder=data_root_test,
        pipeline=test_pipeline,
        test_mode=True,
        clip_len=clip_len,
        stride_rate=stride_rate,
        frame_interval_list=frame_interval_list,
    ),
    val=dict(
        type=dataset_type,
        prop_file=ann_file_val,
        root_folder=data_root_val,
        pipeline=val_pipeline,
        test_mode=True,
        clip_len=clip_len,
        stride_rate=stride_rate,
        frame_interval_list=frame_interval_list,
    ),
    train=dict(
        type=dataset_type,
        prop_file=ann_file_train,
        root_folder=data_root_train,
        pipeline=train_pipeline,
        epoch_multiplier=1,
        clip_len=clip_len,
        stride_rate=stride_rate,
        frame_interval_list=frame_interval_list,
    ),
)

# only work when set --validate
evaluation = dict(interval=1, save_best="OBO", by_epoch=True, rule="greater")

# for fp16 training
# fp16 = dict(loss_scale=512.)

# optimizer
optimizer = dict(type="AdamW", lr=0.0002, weight_decay=0.0)
optimizer_config = dict()
# learning policy
lr_config = dict(policy="step", step=[7], gamma=0.1, by_epoch=True)

# runtime settings
work_dir = "/DATA/disk1/lizishi/react_out/repcount_20230228_r50"
output_config = dict(out=f"{work_dir}/results.json")
