# Introduction
This repository contains some trackers, their tracking results, and evaluations

# Other trackers

### deep_sort/
https://github.com/nwojke/deep_sort

### KCFpy/
https://github.com/uoip/KCFpy

### kcf.py
KCF using OpenCV
- kcf_uav.py: testing on our test dataset

### iou-tracker/
https://github.com/bochinski/iou-tracker

### dsst.py
DSST using dlib


# Evaluations

### evaluate.py
AOS, ATR, Success Plot, and create results for draw_graph from tracker_benchmark
See the main function for knowing how to use the script

# Visulization

### demo.py
Draw many trackers' tracking results at the same time

# Utilities
*utils.py: nothing special

# Path to data

### config.py
**Data paths are stored in this script, modify it when running on a new device**

# Ohter diectory

### output_video/

### tracked ration/
video-wise ATR

### overlap score/
video-wise AOS

### results/
Store results for draw_graph.py in tracker_benchmark/

### uav_output/
Store tracking results of test dataset using different trackers
