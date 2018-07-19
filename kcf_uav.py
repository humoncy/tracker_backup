from __future__ import print_function
import cv2
import sys
import os
import glob
from os.path import basename, splitext
import csv
import numpy as numpy
import matplotlib.pyplot as plt
import time

import dlib
from utils import get_data_lists, sort_nicely


if __name__ == '__main__' :

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    print(cv2.__version__)

    data = {
        'image_folder': '/home/peng/data/sort_data/images/',
        'annot_folder': '/home/peng/data/sort_data/annotations/'
    }

    annots, videos = get_data_lists(data)

    total_time = 0.0
    total_frames = 0

    for i, video in enumerate(videos):
        video_name = splitext(basename(video))[0]
        tracker = cv2.TrackerKCF_create()
        with open('uav_output/kcf_output/' + video_name + '.txt', 'w') as out_file:
            print("Processing %s." % video_name)
            img_paths = sorted(glob.glob(os.path.join(video, "*.jpg")))
            sort_nicely(img_paths)
            for frame_index, img_path in enumerate(img_paths):
                frame = cv2.imread(img_path)
                if frame_index == 0:
                    # Start a track on the juice box. If you look at the first frame you
                    # will see that the juice box is contained within the bounding
                    # box (74, 67, 112, 153).
                    with open(annots[i]) as f:
                        initial_box = [int(i) for i in f.readline().strip().split(',')]
                    x, y, w, h = initial_box
                    bbox = (x,y,w,h)
                    ok = tracker.init(frame, bbox)
                else:
                    start_time = time.time()
                    # Else we just attempt to track from the previous frame
                    ok, bbox = tracker.update(frame)
                    cycle_time = time.time() - start_time
                    total_time += cycle_time

                    total_frames += 1

                print('%d,%d,%d,%d' % (bbox[0], bbox[1], bbox[2], bbox[3]), file=out_file)
    
    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
 