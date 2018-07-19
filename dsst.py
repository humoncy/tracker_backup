#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# This example shows how to use the correlation_tracker from the dlib Python
# library.  This object lets you track the position of an object as it moves
# from frame to frame in a video sequence.  To use it, you give the
# correlation_tracker the bounding box of the object you want to track in the
# current video frame.  Then it will identify the location of the object in
# subsequent frames.
#
# In this particular example, we are going to run on the
# video sequence that comes with dlib, which can be found in the
# examples/video_frames folder.  This video shows a juice box sitting on a table
# and someone is waving the camera around.  The task is to track the position of
# the juice box as the camera moves around.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy
from __future__ import print_function

import os
import glob
from os.path import basename, splitext
import csv
import numpy as numpy
import matplotlib.pyplot as plt
import time

import dlib
from utils import get_data_lists, sort_nicely

def track(video, annot):
    # Create the correlation tracker - the object needs to be initialized
    # before it can be used
    tracker = dlib.correlation_tracker()

    win = dlib.image_window()
    # We will track the frames as we load them off of disk
    img_paths = sorted(glob.glob(os.path.join(video, "*.jpg")))
    sort_nicely(img_paths)
    for k, f in enumerate(img_paths):
        print("Processing Frame {}".format(k))
        img = dlib.load_rgb_image(f)

        # We need to initialize the tracker on the first frame
        if k == 0:
            # Start a track on the juice box. If you look at the first frame you
            # will see that the juice box is contained within the bounding
            # box (74, 67, 112, 153).
            with open(annot) as f:
                initial_box = [int(i) for i in f.readline().strip().split(',')]
            x, y, w, h = initial_box
            tracker.start_track(img, dlib.rectangle(x, y, x+w, y+h))
        else:
            # Else we just attempt to track from the previous frame
            tracker.update(img)

        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(tracker.get_position())
        dlib.hit_enter_to_continue()


if __name__ == '__main__':

    data = {
        'image_folder': '/home/peng/data/sort_data/images/',
        'annot_folder': '/home/peng/data/sort_data/annotations/'
    }
    # Path to the video frames
    video_folder = os.path.join(data['image_folder'], 'person22')
    annot = os.path.join(data['annot_folder'], 'person22.txt')
    track(video_folder, annot)

    exit()

    annots, videos = get_data_lists(data)

    total_time = 0.0
    total_frames = 0

    if not os.path.exists('dsst_output'):
        os.makedirs('dsst_output')

    for i, video in enumerate(videos):
        video_name = splitext(basename(video))[0]
        tracker = dlib.correlation_tracker()
        with open('dsst_output/' + video_name + '.txt', 'w') as out_file:
            print("Processing %s." % video_name)
            img_paths = sorted(glob.glob(os.path.join(video, "*.jpg")))
            sort_nicely(img_paths)
            for k, f in enumerate(img_paths):
                img = dlib.load_rgb_image(f)
                if k == 0:
                    # Start a track on the juice box. If you look at the first frame you
                    # will see that the juice box is contained within the bounding
                    # box (74, 67, 112, 153).
                    with open(annots[i]) as f:
                        initial_box = [int(i) for i in f.readline().strip().split(',')]
                    x, y, w, h = initial_box
                    tracker.start_track(img, dlib.rectangle(x, y, x+w, y+h))
                else:
                    start_time = time.time()
                    # Else we just attempt to track from the previous frame
                    tracker.update(img)
                    cycle_time = time.time() - start_time
                    total_time += cycle_time

                    total_frames += 1
                    
                bbox = tracker.get_position()
                print('%d,%d,%d,%d' % (bbox.left(), bbox.top(), bbox.width(), bbox.height()), file=out_file)
    
    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
    
