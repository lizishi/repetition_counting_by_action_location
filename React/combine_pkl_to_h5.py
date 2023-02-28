import os
import h5py
import numpy as np
import pickle


sub_dir = "test"
ROOT_DIR = "/DATA/disk1/lizishi/LLSP/feature-frame/{}".format(sub_dir)

f = h5py.File("/DATA/disk1/lizishi/LLSP/feature-frame/{}_rgb.h5".format(sub_dir), "w")
video_list = os.listdir(ROOT_DIR)
for video in video_list:
    video_name = video.replace(".pkl", "")
    with open(os.path.join(ROOT_DIR, video), "rb") as input_file:
        data = pickle.load(input_file)
    f.create_dataset(video_name, data=data.squeeze())
f.close()
