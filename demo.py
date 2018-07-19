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
    """ Draw multiple tracking results and annotations on video at the same time
        Params:
            videos: list of path to video directory (each directory contains video frames)
            annots: list of path to annotation files
            trk_results: list of list of path to tracking results
            isMotformat: True, False, or 'DET'(corresponding results are detection results)
        Return:
            None, video will be stored in output_video
    """
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
                # Iterate over tracker results
                if isMotformat[trk_id] is True:
                    # Multi-object tracking results
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
                    # Detection results
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
                    # Single object tracking results
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

            if fi > 180:
                break

        video_id += 1

    print("Lost target: {}".format(nb_lost))


if __name__ == '__main__':
    data = {
        'image_folder': '/home/peng/data/sort_data/images/',
        'annot_folder': '/home/peng/data/sort_data/annotations/',
        'dsst_tracked_results': '/home/peng/trackers/uav_output/dsst_output/',
        'ecohc_tracked_results': '/home/peng/trackers/uav_output/eco_hc_output/',
        'eco_tracked_results': '/home/peng/trackers/uav_output/eco_output/',
        're3_tracked_results': '/home/peng/trackers/uav_output/re3_output/',
        'kcf_tracked_results': '/home/peng/trackers/uav_output/kcf_output/',
        'deep_sort_results': '/home/peng/trackers/uav_output/deep_sort_output/',
        'iou_traker_results': '/home/peng/trackers/uav_output/ioutrk_output/',
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

    dsst_r = sorted(glob.glob((data['dsst_tracked_results'] + "*")))
    sort_nicely(dsst_r)
    ecohc_r = sorted(glob.glob((data['ecohc_tracked_results'] + "*")))
    sort_nicely(ecohc_r)
    eco_r = sorted(glob.glob((data['eco_tracked_results'] + "*")))
    sort_nicely(eco_r)
    re3_r = sorted(glob.glob((data['re3_tracked_results'] + "*")))
    sort_nicely(re3_r)
    kcf_r = sorted(glob.glob((data['kcf_tracked_results'] + "*")))
    sort_nicely(kcf_r)
    deepsort_r = sorted(glob.glob((data['deep_sort_results'] + "*")))
    sort_nicely(deepsort_r)
    ioutrk_r = sorted(glob.glob((data['iou_traker_results'] + "*")))
    sort_nicely(ioutrk_r)
    sort_r = sorted(glob.glob((data['sort_tracked_results'] + "*")))
    sort_nicely(sort_r)
    rid_sort_r = sorted(glob.glob((data['reid_sort_results'] + "*")))
    sort_nicely(rid_sort_r)
    y2_sort_r = sorted(glob.glob((data['yolo2_sort_results'] + "*")))
    sort_nicely(y2_sort_r)
    y2_ridsort_r = sorted(glob.glob((data['y2_ridsort_results'] + "*")))
    sort_nicely(y2_ridsort_r)
    yolo2_det_r = sorted(glob.glob((data['yolo2_det_results'] + "*")))
    sort_nicely(yolo2_det_r)
    yolo3_det_r = sorted(glob.glob((data['yolo3_det_results'] + "*")))
    sort_nicely(yolo3_det_r)

    annots, videos = get_data_lists(data)

    # demo(videos, annots, [dsst_r], [False])
    # demo(videos, annots, [ecohc_r, rid_sort_r], [False, True])
    # demo(videos, annots, [dsst_r, sort_r], [False, True])
    demo(videos, annots, [sort_r], [True])
    # demo(videos, annots, [yolo3_r, sort_r], ['DET', True])
