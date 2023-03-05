import copy
import os
import os.path
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import h5py
from math import ceil
from numpy.random import randint
import mmcv

from React.utill.temporal_box_producess import softnms_v2, segment_iou
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.pipelines import Compose
from tools.misc.evaluation import eval_map


type2label = [
    "pull_up",
    "others",
    "situp",
    "battle_rope",
    "pommelhorse",
    "jump_jack",
    "bench_pressing",
    "front_raise",
    "push_up",
    "squat",
]
# type2label = type2label[:]


def fix_and_get_label(cls):
    if cls == "pullups":
        cls = "pull_up"
    elif cls == "squant":
        cls = "squat"
    elif cls == "pushups":
        cls = "push_up"
    elif cls == "jumpjacks":
        cls = "jump_jack"
    elif cls == "benchpressing":
        cls = "bench_pressing"
    elif cls == "frontraise":
        cls = "front_raise"
    return type2label.index(cls)


def segment_iou_numpy(segment1, segment2):
    assert (segment1[..., 1] >= segment1[..., 0]).all()
    assert (segment2[..., 1] >= segment2[..., 0]).all()

    if segment1.ndim == 2 and segment2.ndim == 2:
        inter = np.clip(
            np.min(
                [
                    segment1[:, None, 1] * np.ones_like(segment2[None, :, 1]),
                    np.ones_like(segment1[:, None, 1]) * segment2[None, :, 1],
                ],
                axis=0,
            )
            - np.max(
                [
                    segment1[:, None, 0] * np.ones_like(segment2[None, :, 0]),
                    np.ones_like(segment1[:, None, 0]) * segment2[None, :, 0],
                ],
                axis=0,
            ),
            a_min=0,
            a_max=None,
        )
        union = (
            (segment1[:, None, 1] - segment1[:, None, 0])
            + (segment2[None, :, 1] - segment2[None, :, 0])
            - inter
        )
        iou = inter / (union + 1e-6)
    elif segment1.ndim == 3 and segment2.ndim == 3:
        inter = np.max(
            np.zeros(1),
            np.min(segment1[:, :, None, 1], segment2[:, None, :, 1])
            - np.max(segment1[:, :, None, 0], segment2[:, None, :, 0]),
        )
        union = (
            (segment1[:, :, None, 1] - segment1[:, :, None, 0])
            + (segment2[:, None, :, 1] - segment2[:, None, :, 0])
            - inter
        )
        iou = inter / (union + 1e-6)
    else:
        raise Exception("not implement for this shape, please check.")
    return iou


def load_proposal_file(filename):
    df = pd.read_csv(filename)
    proposal_list = []
    for i, rows in df.iterrows():
        try:
            video_name = rows["name"].replace(".mp4", "")
            cls = fix_and_get_label(rows["type"])
            if cls == -1:
                return
            count = int(rows["count"]) if pd.notna("count") else 0
            n_frames = int(rows["total_frames"])

            labels = rows[6:]
            gt_bbox = []
            for i in range(count):
                if pd.notna(labels[2 * i]) and pd.notna(labels[2 * i + 1]):
                    gt_bbox.append([0, int(labels[2 * i]), int(labels[2 * i + 1])])
                    # gt_bbox.append([cls, int(labels[2 * i]), int(labels[2 * i + 1])])
        except ValueError:
            continue

        proposal_list.append([video_name, n_frames, gt_bbox, []])
    return proposal_list


class SegmentInstance:
    def __init__(
        self,
        start_frame,
        end_frame,
        video_frame_count,
        fps=1,
        label=None,
        best_iou=None,
        overlap_self=None,
    ):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, video_frame_count)
        self._label = label
        self.fps = fps

        self.coverage = (end_frame - start_frame) / video_frame_count

        self.best_iou = best_iou
        self.overlap_self = overlap_self

    @property
    def start_time(self):
        return self.start_frame / self.fps

    @property
    def end_time(self):
        return self.end_frame / self.fps

    @property
    def label(self):
        return self._label if self._label is not None else -1


class VideoRecord:
    def __init__(self, prop_record):
        self._data = prop_record

        frame_count = int(self._data[1])

        # build instance record
        self.gt = [
            SegmentInstance(
                int(x[1]), int(x[2]), frame_count, label=int(x[0]), best_iou=1.0
            )
            for x in self._data[2]
            if int(x[2]) > int(x[1])
        ]

        self.gt = list(filter(lambda x: x.start_frame < frame_count, self.gt))

        self.proposals = [
            SegmentInstance(
                int(x[3]),
                int(x[4]),
                frame_count,
                label=int(x[0]),
                best_iou=float(x[1]),
                overlap_self=float(x[2]),
            )
            for x in self._data[3]
            if int(x[4]) > int(x[3])
        ]

        self.proposals = list(
            filter(lambda x: x.start_frame < frame_count, self.proposals)
        )

    @property
    def id(self):
        return self._data[0].strip("\n").split("/")[-1]

    @property
    def num_frames(self):
        return int(self._data[1])


@DATASETS.register_module()
class RepCountDatasetE2E(data.Dataset):
    def __init__(
        self,
        prop_file,
        root_folder,
        pipeline,
        exclude_empty=True,
        epoch_multiplier=1,
        clip_len=128,
        class_num=2,
        stride_rate=0.75,
        test_mode=False,
        soft_nms_sigma=0.1,
        soft_nms_threshold=1e-4,
        frame_interval_list=[1],
    ):
        self.prop_file = prop_file
        self.root_folder = root_folder
        self.test_mode = test_mode
        self.clip_len = clip_len
        self.stride_rate = stride_rate
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_threshold = soft_nms_threshold
        self.frame_interval_list = frame_interval_list

        self.exclude_empty = exclude_empty
        self.epoch_multiplier = epoch_multiplier
        self.class_num = class_num
        self.best = 0.0

        self.pipeline = Compose(pipeline)

        parse_time = time.time()
        self._parse_data_list()

        if not test_mode:
            self.training_clip_list, self.cls_list = self.prepare_training_clip()
            self.real_len = len(self.training_clip_list)
        else:
            self.real_len = len(self.video_list)

        print("File parsed. Time:{:.2f}".format(time.time() - parse_time))

    def _parse_data_list(self):
        prop_info = load_proposal_file(self.prop_file)

        self.video_list = [VideoRecord(p) for p in prop_info]

        if self.exclude_empty:
            self.video_list = list(filter(lambda x: len(x.gt) > 0, self.video_list))

        # todo trick: sort for test(train, no used) dataset, then data of a batch have similiar len, which help reduce the testing time
        self.video_list = sorted(self.video_list, key=lambda x: x.num_frames)

        self.video_dict = {v.id: v for v in self.video_list}
        self.gt_list = self.get_all_gt()

    def get_all_gt(self):
        gt_list = []
        for video in self.video_list:
            gts = [
                [
                    x.label,
                    (x.start_frame + x.end_frame) / (2 * video.num_frames),
                    (x.end_frame - x.start_frame) / video.num_frames,
                ]
                for x in video.gt
            ]
            gt_list.append(
                sorted(gts, key=lambda x: x[1] - 0.5 * x[2])
            )  # sort gt by time
        return gt_list

    def keep_gt_with_thershold(self, gt, start_ratio, end_ratio, thershold):
        gt_start = gt[:, 1] - 0.5 * gt[:, 2]
        gt_end = gt[:, 1] + 0.5 * gt[:, 2]

        inter = np.maximum(
            0, np.minimum(gt_end, end_ratio) - np.maximum(gt_start, start_ratio)
        )
        union = gt_end - gt_start

        gt_iou = inter / union

        idx = gt_iou >= thershold
        return gt[idx]

    def rescale_gt(self, gt, start_ratio, origin_len, new_len):
        gt[:, 1] = (gt[:, 1] - start_ratio) * (origin_len / new_len)
        gt[:, 2] = gt[:, 2] * (origin_len / new_len)

        return gt

    def get_safe_inds(self, frame_inds, total_frames):
        safe_inds = frame_inds < total_frames
        unsafe_inds = 1 - safe_inds
        last_ind = np.max(safe_inds * frame_inds, axis=0)
        new_inds = safe_inds * frame_inds + (unsafe_inds.T * last_ind).T
        return new_inds

    def prepare_training_clip(self):
        all_clip_list = []
        cls_list = [[] for _ in range(self.class_num - 1)]
        for v_idx in range(len(self.video_list)):
            video = self.video_list[v_idx]
            gt = self.gt_list[v_idx]
            gt = np.array(gt)

            vid_full_name = video.id
            vid = vid_full_name.split("/")[-1]
            total_frames = video.num_frames

            for frame_interval in self.frame_interval_list:
                ori_clip_len = self.clip_len * frame_interval
                clips_num = (
                    ceil(
                        (total_frames - ori_clip_len)
                        / (ori_clip_len * self.stride_rate)
                    )
                    + 1
                )
                clips_ratio = ori_clip_len / total_frames

                result = {}
                result["frame_dir"] = os.path.join(self.root_folder, vid)
                result["filename_tmpl"] = "img_{:05d}.jpg"
                result["modality"] = "RGB"
                result["video_name"] = vid
                result[
                    "origin_snippet_num"
                ] = total_frames  # the snippet number of the whole video

                if clips_num <= 1:
                    frame_num = total_frames // frame_interval
                    result["frame_inds"] = np.arange(frame_num) * frame_interval + 1
                    result["gt_bbox"] = gt  # (gt_num, 3)
                    result["clip_len"] = frame_num  # the snippet number of the clip
                    result["num_clips"] = 1
                    v_cls_list = []
                    for each_labels in gt:
                        cls = int(each_labels[0])
                        if cls not in v_cls_list:
                            v_cls_list.append(cls)
                            cls_list[cls].append(len(all_clip_list))
                    all_clip_list.append(result)
                    continue

                # sample all data for train
                for window_num in range(clips_num):
                    start_snippet = int(ori_clip_len * self.stride_rate) * window_num

                    # finding matched gt
                    start_ratio = (clips_ratio * self.stride_rate) * window_num
                    end_ratio = start_ratio + clips_ratio

                    frame_inds = (
                        self.get_safe_inds(
                            np.arange(self.clip_len) * frame_interval + start_snippet,
                            total_frames,
                        )
                        + 1
                    )

                    # find the matching gt in the windows
                    selected_gt = self.keep_gt_with_thershold(
                        gt, start_ratio, end_ratio, thershold=0.75
                    )

                    if len(selected_gt) == 0:
                        continue

                    # selected_gt = gt[index]
                    clip_gt = self.rescale_gt(
                        selected_gt, start_ratio, total_frames, ori_clip_len
                    )  # gt_num_i, 3

                    result["frame_inds"] = frame_inds
                    result["clip_len"] = self.clip_len  # the snippet number of the clip
                    result["gt_bbox"] = clip_gt  # (gt_num, 3)
                    result["num_clips"] = 1
                    v_cls_list = []
                    for each_labels in clip_gt:
                        cls = int(each_labels[0])
                        if cls not in v_cls_list:
                            v_cls_list.append(cls)
                            cls_list[cls].append(len(all_clip_list))
                    all_clip_list.append(result)

        return all_clip_list, cls_list

    def prepare_testing_clip(self, result):
        total_frames = result["origin_snippet_num"]
        frame_interval = self.frame_interval_list[0]
        ori_clip_len = self.clip_len * frame_interval
        gt = result["gt_bbox"]

        clips_num = (
            ceil((total_frames - ori_clip_len) / (ori_clip_len * self.stride_rate)) + 1
        )
        clips_ratio = ori_clip_len / total_frames

        frame_inds_list = []
        gt_for_each_clips = []

        if clips_num <= 1:
            frame_num = total_frames // frame_interval
            result["frame_inds"] = (
                np.arange(frame_num) * frame_interval + 1
            )  # 1 x clip_len x feature_dim
            result["gt_bbox"] = [gt]  # 1 x gt_num x 3
            result["snippet_num"] = frame_num
            result["clip_len"] = frame_num
            result["num_clips"] = 1
            return result

        snippet_num_list = []
        # sample all data for test
        for window_num in range(clips_num):
            start_snippet = int(self.clip_len * self.stride_rate) * window_num
            (
                frame_inds_list.append(
                    self.get_safe_inds(
                        np.arange(self.clip_len) * frame_interval + start_snippet,
                        total_frames,
                    )
                    + 1
                )
            )
            snippet_num_list.append(self.clip_len)

            # finding matched gt
            start_ratio = (clips_ratio * self.stride_rate) * window_num
            end_ratio = start_ratio + clips_ratio

            # find the matching gt in the windows
            index = np.logical_and(
                (gt[:, 1] - gt[:, 2] / 2) >= start_ratio,
                (gt[:, 1] + gt[:, 2] / 2) <= end_ratio,
            )

            clip_gt = self.rescale_gt(
                gt[index], start_ratio, total_frames, self.clip_len
            )  # gt_num_i, 3
            gt_for_each_clips.append(clip_gt)

        result["frame_inds"] = np.array(
            frame_inds_list
        ).flatten()  # clip_num, clip_len, feature_dim
        result["gt_bbox"] = gt_for_each_clips
        # list: clip_num x [gt_num_i , 3]
        result["clip_len"] = len(result["frame_inds"])
        result["snippet_num"] = snippet_num_list
        result["num_clips"] = 1
        return result

    def dump_results(self, result, out=None):
        mmcv.dump(result, out)
        return self.evaluate(result)

    def evaluate(self, result, logger=None, metrics=None):
        result = pd.DataFrame(result).values
        result_pred = result[:, 0]
        result_gt = result[:, 1]

        softnms_v2(
            result_pred,
            sigma=self.soft_nms_sigma,
            top_k=30,
            score_threshold=self.soft_nms_threshold,
        )
        result_pred = copy.deepcopy(result_pred)
        thershold = np.linspace(0.3, 0.7, 5)
        iou_out = []
        for each in thershold:
            mean_ap, eval_results = eval_map(
                result_pred, result_gt, iou_thr=each, mode="anet", nproc=8
            )
            iou_out.append(np.round(mean_ap * 100, decimals=1))

        print("\nmAP for 0.3-0.7", iou_out)
        mean_map = np.round(np.mean(iou_out), decimals=1)
        print("Average mAP", mean_map)
        if mean_map > self.best:
            self.best = mean_map

        threshold = 0.2
        test_mae = []
        test_obo = []
        for pred, gt in zip(result_pred, result_gt):
            pred_num = len(
                [
                    i
                    for j in range(self.class_num - 1)
                    for i in pred[j]
                    if i[-1] > threshold
                ]
            )
            gt_num = len(gt["labels"])
            mae = np.abs(pred_num - gt_num) / (gt_num + 1e-6)
            obo = np.abs(pred_num - gt_num) <= 1
            test_mae.append(mae)
            test_obo.append(obo)
        mae_mean = np.mean(test_mae)
        obo_mean = np.mean(test_obo)
        print("MAE: {}, OBO:{}".format(mae_mean, obo_mean))

        iou_decay = 0
        for pred in result_pred:
            pred_all = np.array(
                [
                    i
                    for j in range(self.class_num - 1)
                    for i in pred[j]
                    if i[-1] > threshold
                ]
            )
            if len(pred_all) == 0:
                continue
            iou_reg_loss = segment_iou_numpy(pred_all, pred_all)
            iou_overlap_matrix = np.triu(iou_reg_loss, 1)
            iou_decay += np.mean(iou_overlap_matrix)
        iou_decay /= len(result_pred)

        print("best mAP: ", self.best)
        return {
            "mAP": mean_map,
            "MAE": mae_mean,
            "OBO": obo_mean,
            "iou_decay": iou_decay,
        }

    def __getitem__(self, index):
        """
        Args:
            index: clip index

        Return:
            result: dict, include keys "frame_dir", "filename_tmpl",
            "frame_inds", "modality", "video_name", "snippt_num",
            "origin_snippet_num", "gt_bbox", "video_gt_bbox"
        """
        real_index = index % self.real_len

        # if training, get training clip (any clips from any videos)
        if not self.test_mode:
            result = self.training_clip_list[real_index]
            return self.pipeline(result)

        # get testing data (all clips for a videos)
        video = self.video_list[real_index]

        vid_full_name = video.id
        vid = vid_full_name.split("/")[-1]
        total_frames = video.num_frames

        gt = self.gt_list[real_index]

        result = {}
        result["frame_dir"] = os.path.join(self.root_folder, vid)
        result["filename_tmpl"] = "img_{:05d}.jpg"
        result["modality"] = "RGB"
        result["video_name"] = vid
        result[
            "origin_snippet_num"
        ] = total_frames  # The snippet number of the whole video
        result["gt_bbox"] = np.array(gt)
        result["video_gt_box"] = np.array(gt)

        result = self.prepare_testing_clip(result)

        return self.pipeline(result)

    def __len__(self):
        return self.real_len * self.epoch_multiplier


if __name__ == "__main__":
    dataset = RepCountDataset()
