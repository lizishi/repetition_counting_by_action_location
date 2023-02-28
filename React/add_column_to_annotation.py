import os
import pandas as pd


sub_dir = "test"
frames_folder = "/DATA/disk1/lizishi/LLSP/frames/{}".format(sub_dir)
df = pd.read_csv("/DATA/disk1/lizishi/LLSP/annotation/{}.csv".format(sub_dir))

frames_num = []
for i, rows in df.iterrows():
    video_name = rows["name"]
    frames = len(os.listdir(os.path.join(frames_folder, video_name.replace(".mp4", ""))))
    frames_num.append(frames)

df.insert(loc=4, column="total_frames", value=frames_num)
df.to_csv("/DATA/disk1/lizishi/LLSP/annotation/{}_new.csv".format(sub_dir))
