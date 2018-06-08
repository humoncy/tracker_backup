import os
import cv2
import glob
import math
import numpy as np
from os.path import basename, splitext
from tqdm import tqdm

from utils import sort_nicely, get_data_lists, isNAN
from bbox_utils import *

colours = [(250,0,0), (0,250,0), (255,255,0), (255,0,255), (0,255,255)]

def demo(videos, annots, trk_results, isMotformat):
    video_id = 0
    for video, annot in zip(videos, annots):
        video_name = basename(video)
        if basename(annot).find(video_name) == -1:
            raise Exception("No corresding video and annotation!")

        if video_name != "person14_2":
            video_id += 1
            continue
        print("Processing " + video_name)
        FPS = 30
        # remember to modify frame width and height before testing video
        frame_width = 1280
        frame_height = 720
        video_writer = cv2.VideoWriter("output_video/"+video_name+'.avi', cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width, frame_height))
        
        image_paths = sorted(glob.glob(os.path.join(video, '*jpg')))
        sort_nicely(image_paths)

        labels = np.loadtxt(annot, delimiter=',')
        preds_per_trk = []
        for trk_r in trk_results:
            preds = np.loadtxt(trk_r[video_id], delimiter=',')
            preds_per_trk.append(preds)
        
        nb_lost = 0.0
        for fi, image_path in enumerate(tqdm(image_paths)):
            image = cv2.imread(image_path)
            for trk_id, preds in enumerate(preds_per_trk):
                if isMotformat[trk_id] is True:
                    index_list = np.argwhere(preds[:, 0] == (fi+1))
                    if index_list.shape[0] != 0:
                        for index in index_list[:, 0]:
                            if preds[index, 1] == 1:
                                target_bbox = preds[index, 2:6]
                                intersect = bbox_intersection(target_bbox, labels[fi])
                            if (intersect/bbox_area(labels[fi])>0.8) and (bbox_area(target_bbox)/bbox_area(labels[fi])<8):
                                target_bbox = xywh_to_xyxy(target_bbox)
                                cv2.rectangle(image, 
                                    (int(target_bbox[0]),int(target_bbox[1])), 
                                    (int(target_bbox[2]),int(target_bbox[3])), 
                                    colours[trk_id], 2)
                        else:
                            nb_lost += 1
                else:
                    rect = xywh_to_xyxy(preds[fi])
                    cv2.rectangle(image, 
                        (int(rect[0]),int(rect[1])), 
                        (int(rect[2]),int(rect[3])), 
                        colours[trk_id], 2)

            if isNAN(labels[fi]) is not True:
                gt_rect = xywh_to_xyxy(labels[fi])
                cv2.rectangle(image, 
                    (int(gt_rect[0]),int(gt_rect[1])), 
                    (int(gt_rect[2]),int(gt_rect[3])), 
                    (0,0,255), 2)

            video_writer.write(image)
        video_id += 1

    print("Lost target: {}".format(nb_lost))


if __name__ == '__main__':
    data = {
        'image_folder': '/home/peng/data/sort_data/images/',
        'annot_folder': '/home/peng/data/sort_data/annotations/',
        'dsst_tracked_results': '/home/peng/trackers/dsst_output/',
        'sort_tracked_results': '/home/peng/darknet/sort/kf_output/',
        'ukf_tracked_results': '/home/peng/darknet/sort/output/'
    }
    annots, videos = get_data_lists(data)

    dsst_r = sorted(glob.glob((data['dsst_tracked_results'] + "*")))
    sort_nicely(dsst_r)
    sort_r = sorted(glob.glob((data['sort_tracked_results'] + "*")))
    sort_nicely(sort_r)
    ukf_r = sorted(glob.glob((data['ukf_tracked_results'] + "*")))
    sort_nicely(ukf_r)

    # demo(videos, annots, [dsst_r], [False])
    # demo(videos, annots, [sort_r], [True])
    demo(videos, annots, [dsst_r, sort_r], [False, True])
    # demo(videos, annots, [dsst_r, sort_r, ukf_r], [False, True, True])
