import os
import cv2
import glob
import math
import numpy as np
from os.path import basename, splitext
from tqdm import tqdm
import matplotlib.cm

from utils import sort_nicely, get_data_lists, isNAN
from bbox_utils import *

cmap = matplotlib.cm.get_cmap('tab20')
ci = np.linspace(0,1,20)
colours = cmap(ci)[:,:3]
colours = colours[:,::-1] * 255

def demo(videos, annots, trk_results, isMotformat):
    video_id = 0
    for video, annot in zip(videos, annots):
        video_name = basename(video)
        if basename(annot).find(video_name) == -1:
            raise Exception("No corresding video and annotation!")

        if video_name != "person14_1":
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
                        max_iou = 0.0
                        target_index = -1
                        for index in index_list[:, 0]:
                            bbox = preds[index, 2:6]
                            iou = bbox_iou(bbox, labels[fi])
                            if iou > max_iou:
                                max_iou = iou
                                target_index = index
                        if max_iou > 0.2 and target_index != -1:
                            target_bbox = preds[target_index, 2:6]
                            target_bbox = xywh_to_xyxy(target_bbox)
                            cv2.rectangle(image, 
                                (int(target_bbox[0]),int(target_bbox[1])), 
                                (int(target_bbox[2]),int(target_bbox[3])), 
                                colours[trk_id], 2)
                            cv2.rectangle(image, 
                                        (int(target_bbox[0])-2,int(target_bbox[1]-20)), 
                                        (int(target_bbox[0])+20+int(preds[target_index, 1])//10 ,int(target_bbox[1])+1), 
                                        colours[trk_id], -1)
                            cv2.putText(image, 
                                        str(int(preds[target_index, 1])), 
                                        (int(target_bbox[0]+2), int(target_bbox[1])-3), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.6, 
                                        (0,0,0), 2)
                        else:
                            nb_lost += 1
                elif isMotformat[trk_id] == 'DET':
                    index_list = np.argwhere(preds[:, 0] == (fi+1))
                    if index_list.shape[0] != 0:
                        max_iou = 0.0
                        target_index = -1
                        for index in index_list[:, 0]:
                            bbox = preds[index, 2:6]
                            iou = bbox_iou(bbox, labels[fi])
                            if iou > max_iou:
                                max_iou = iou
                                target_index = index
                        if max_iou > 0.5 and target_index != -1:
                            target_bbox = preds[target_index, 2:6]
                            target_bbox = xywh_to_xyxy(target_bbox)
                            cv2.rectangle(image, 
                                (int(target_bbox[0]),int(target_bbox[1])), 
                                (int(target_bbox[2]),int(target_bbox[3])), 
                                colours[trk_id], 4)
                else:
                    rect = xywh_to_xyxy(preds[fi])
                    cv2.rectangle(image, 
                        (int(rect[0]),int(rect[1])), 
                        (int(rect[2]),int(rect[3])), 
                        # (colours[trk_id]), 4)
                        (255,0,0), 4)

            if isNAN(labels[fi]) is not True:
                gt_rect = xywh_to_xyxy(labels[fi])
                cv2.rectangle(image, 
                    (int(gt_rect[0]),int(gt_rect[1])), 
                    (int(gt_rect[2]),int(gt_rect[3])), 
                    (0,0,255), 2)

            video_writer.write(image)
            # dir_path = 'output_video/person14_1/'
            # frame_name = "{0:0=3d}".format(fi) + ".jpg"
            # cv2.imwrite(dir_path + frame_name, image)

        video_id += 1

    print("Lost target: {}".format(nb_lost))


if __name__ == '__main__':
    data = {
        'image_folder': '/home/peng/data/sort_data/images/',
        'annot_folder': '/home/peng/data/sort_data/annotations/',
        'dsst_tracked_results': '/home/peng/trackers/dsst_output/',
        'sort_tracked_results': '/home/peng/darknet/sort/kf_output/',
        'ukf_tracked_results': '/home/peng/darknet/sort/output/',
        'reid_sort_results': '/home/peng/darknet/sort/reid_output/',
        'reid_thr45_results': '/home/peng/darknet/sort/reid_thr45_output/',

        # 'yolo2_sort_results': '/home/peng/basic-yolo-keras/sort/output/',
        # 'y2_ridsort_results': '/home/peng/basic-yolo-keras/sort/reid_output/'
        'yolo2_sort_results': '/home/peng/darknetv2/sort/output/',
        'y2_ridsort_results': '/home/peng/darknetv2/sort/reid_output/',

        'yolo2_det_results': '/home/peng/darknetv2/det_mot/',
        'yolo3_det_results': '/home/peng/darknet/det_mot/'
    }
    annots, videos = get_data_lists(data)

    dsst_r = sorted(glob.glob((data['dsst_tracked_results'] + "*")))
    sort_nicely(dsst_r)
    sort_r = sorted(glob.glob((data['sort_tracked_results'] + "*")))
    sort_nicely(sort_r)
    # ukf_r = sorted(glob.glob((data['ukf_tracked_results'] + "*")))
    # sort_nicely(ukf_r)
    rid_sort_r = sorted(glob.glob((data['reid_sort_results'] + "*")))
    sort_nicely(rid_sort_r)
    # rid45_sort_r = sorted(glob.glob((data['reid_thr45_results'] + "*")))
    # sort_nicely(rid45_sort_r)
    y2_sort_r = sorted(glob.glob((data['yolo2_sort_results'] + "*")))
    sort_nicely(y2_sort_r)

    y2_ridsort_r = sorted(glob.glob((data['y2_ridsort_results'] + "*")))
    sort_nicely(y2_ridsort_r)

    yolo2_det_r = sorted(glob.glob((data['yolo2_det_results'] + "*")))
    sort_nicely(yolo2_det_r)
    yolo3_det_r = sorted(glob.glob((data['yolo3_det_results'] + "*")))
    sort_nicely(yolo3_det_r)

    # demo(videos, annots, [dsst_r], [False])
    demo(videos, annots, [dsst_r, rid_sort_r], [False, True])
    # demo(videos, annots, [dsst_r, sort_r], [False, True])
    # demo(videos, annots, [yolo3_r, sort_r], ['DET', True])
