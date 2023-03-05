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
clip_len = 2048
stride_rate = 0.25

# include the contrastive epoch
total_epochs = 80
feature_type = "TSN"

# model settings
model = dict(
    type="React",
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
dataset_type = "RepCountDataset"
data_root_train = "/DATA/disk1/lizishi/LLSP/feature-frame/train_rgb.h5"
data_root_val = "/DATA/disk1/lizishi/LLSP/feature-frame/valid_rgb.h5"
data_root_test = "/DATA/disk1/lizishi/LLSP/feature-frame/test_rgb.h5"
flow_root_train = None
flow_root_val = None

ann_file_train = "/DATA/disk1/lizishi/LLSP/annotation/train_new.csv"
ann_file_val = "/DATA/disk1/lizishi/LLSP/annotation/valid_new.csv"
ann_file_test = "/DATA/disk1/lizishi/LLSP/annotation/test_new.csv"

train_frame_interval_list_aug = [1, 2, 4]

test_pipeline = [
    dict(
        type="Collect",
        keys=["raw_feature", "gt_bbox", "video_gt_box", "snippet_num"],
        meta_name="video_meta",
        meta_keys=["video_name", "origin_snippet_num"],
    ),
    dict(
        type="ToDataContainer",
        fields=[
            dict(key="gt_bbox", stack=False, cpu_only=True),
            dict(key="raw_feature", stack=False, cpu_only=True),
            dict(key="video_gt_box", stack=False, cpu_only=True),
            dict(key="snippet_num", stack=True, cpu_only=True),
        ],
    ),
]
train_pipeline = [
    dict(
        type="Collect",
        keys=[
            "raw_feature",
            "gt_bbox",
            "snippet_num",
        ],
        meta_name="video_meta",
        meta_keys=["video_name", "origin_snippet_num"],
    ),
    dict(
        type="ToTensor",
        keys=[
            "snippet_num",
        ],
    ),
    dict(
        type="ToDataContainer",
        fields=[
            dict(key="gt_bbox", stack=True, cpu_only=True),
            dict(key="raw_feature", stack=True, cpu_only=True),
        ],
    ),
]
val_pipeline = [
    dict(
        type="Collect",
        keys=["raw_feature", "gt_bbox", "video_gt_box", "snippet_num"],
        meta_name="video_meta",
        meta_keys=["video_name", "origin_snippet_num"],
    ),
    dict(
        type="ToDataContainer",
        fields=[
            dict(key="gt_bbox", stack=False, cpu_only=True),
            dict(key="raw_feature", stack=False, cpu_only=True),
            dict(key="video_gt_box", stack=False, cpu_only=True),
            dict(key="snippet_num", stack=True, cpu_only=True),
        ],
    ),
]

data = dict(
    train_dataloader=dict(
        workers_per_gpu=4,
        videos_per_gpu=16,
        drop_last=False,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=4,
    ),
    val_dataloader=dict(
        workers_per_gpu=4,
        videos_per_gpu=16,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=2,
    ),
    test_dataloader=dict(
        workers_per_gpu=4,
        videos_per_gpu=16,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=2,
    ),
    test=dict(
        type=dataset_type,
        prop_file=ann_file_test,
        ft_path=data_root_test,
        pipeline=test_pipeline,
        test_mode=True,
        clip_len=clip_len,
        stride_rate=stride_rate,
        feature_type=feature_type,
    ),
    val=dict(
        type=dataset_type,
        prop_file=ann_file_val,
        ft_path=data_root_val,
        pipeline=val_pipeline,
        test_mode=True,
        clip_len=clip_len,
        stride_rate=stride_rate,
        feature_type=feature_type,
    ),
    train=dict(
        type=dataset_type,
        prop_file=ann_file_train,
        ft_path=data_root_train,
        pipeline=train_pipeline,
        epoch_multiplier=1,
        feature_type=feature_type,
        clip_len=clip_len,
        stride_rate=stride_rate,
        frame_interval_list=train_frame_interval_list_aug,
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
work_dir = (
    "/DATA/disk1/lizishi/react_out/repcount_20230224_frame_interval_1_2__clip_len_2048"
)
output_config = dict(out=f"{work_dir}/results.json")
