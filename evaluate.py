import os
from os.path import basename, splitext
import numpy as np
import glob
import matplotlib.pyplot as plt
import csv
import json
from tqdm import tqdm

from utils import sort_nicely, isNAN
from bbox_utils import *


###################################################################################
# Average Overlap Score
def average_overlap_score(ann, trk_results, isMOTformat, trk_name):
    nb_video = len(ann)
    aos = 0.0
    data = []
    data.append(["video", "overlap score", "#frame"])
    for i, annot in enumerate(tqdm(ann)):
        video_name = splitext(basename(annot))[0]
        if basename(annot)[:-4] != basename(trk_results[i])[:-4]:
            print("Annotations:", annot)
            print("Tracked result:", trk_results[i])
            raise ValueError("Wrong annotation and track correspondence.")

        # if video_name != "person22":
            # continue

        labels = np.loadtxt(annot, delimiter=',')
        preds = np.loadtxt(trk_results[i], delimiter=',')

        nb_frame = len(labels)

        overlap_score = 0.0
        lose_frames = 0

        for fi, label in enumerate(labels):
            if fi == 0:
                # Tracking start
                continue
            if isNAN(labels[fi]):
                # No target in this frame
                continue

            if isMOTformat:
                index_list = np.argwhere(preds[:, 0] == (fi+1))
                if index_list.shape[0] != 0:
                    # Only consider maximum IoU
                    max_iou = 0.0 
                    for index in index_list[:, 0]:
                        bbox = preds[index, 2:6]
                        iou = bbox_iou(bbox, label)
                        if iou > max_iou:
                            max_iou = iou

                    overlap_score += max_iou

                    # Consider right identity
                    # for index in index_list[:, 0]:
                    #     if preds[index, 1] == target_identity:
                    #         target_pred = preds[index, 2:6]
                    #         intersect = bbox_intersection(target_pred, label)
                    #         cover_rate = intersect/bbox_area(label)
                    #         magnification = bbox_area(target_pred)/bbox_area(label)
                    #         position_error = bbox_distance(target_pred, label)
                    #         if (cover_rate>0.5 and magnification<3 and position_error<15) or (cover_rate>0.7 and magnification<8 and position_error<30):
                    #             lose_frames = 0
                    #             atr += 1
                    #         else:
                    #             lose_frames += 1
                else:
                    # print("Lost frame:", fi+1)
                    lose_frames += 1
            else:
                iou = bbox_iou(preds[fi], label)
                if iou > 0:
                    overlap_score += iou
                else:
                    lose_frames += 1

        overlap_score /= (nb_frame - 1)
        data.append([splitext(basename(annot))[0], "{:.2f}%".format(overlap_score * 100), nb_frame])
        aos += overlap_score

    with open('overlap score/' + trk_name + '.csv', "w") as f:
        w = csv.writer(f)
        w.writerows(data)
    
    aos /= nb_video
    
    return aos

def AOS(ann, trk_results, trk_names):
    print("Average Overlap Score:")
    for trk_r, trk_name in zip(trk_results, trk_names):
        if trk_name.find('SORT') != -1 or trk_name.find('YOLO') != -1:
            isMOTformat = True
        else:
            isMOTformat = False
        aos = average_overlap_score(ann, trk_r, isMOTformat, trk_name)
        print('{:.3f}%'.format(aos*100), trk_name)

# Average Overlap Score
###################################################################################

###################################################################################
# Average Tracked Ratio
def average_tracked_ratio(ann, trk_results, isMOTformat, filename):
    nb_video = len(ann)
    total_atr = 0.0
    data = []
    data.append(["video", "tracked_ratio", "#frame"])
    for i, annot in enumerate(tqdm(ann)):
        video_name = splitext(basename(annot))[0]
        if basename(annot)[:-4] != basename(trk_results[i])[:-4]:
            print("Annotations:", annot)
            print("Tracked result:", trk_results[i])
            raise ValueError("Wrong annotation and track correspondence.")

        # if video_name != "person22":
            # continue

        labels = np.loadtxt(annot, delimiter=',')
        preds = np.loadtxt(trk_results[i], delimiter=',')

        nb_frame = len(labels)

        atr = 0.0
        lose_frames = 0

        for fi, label in enumerate(labels):
            if fi == 0:
                # Tracking start
                if isMOTformat:
                    target_identity = 1
                    index_list = np.argwhere(preds[:, 0] == (fi+1))
                    if index_list.shape[0] != 0:
                        max_intersection = 0.0 
                        for index in index_list[:, 0]:
                            bbox = preds[index, 2:6]
                            intersect = bbox_intersection(bbox, label)
                            if intersect > max_intersection:
                                max_intersection = intersect
                                target_pred = bbox
                                target_identity = preds[index, 1]
                continue
            if isNAN(labels[fi]):
                # No target in this frame
                continue

            if isMOTformat:
                index_list = np.argwhere(preds[:, 0] == (fi+1))
                if index_list.shape[0] != 0:
                    # Only consider maximum IoU
                    # max_intersection = 0.0 
                    # for index in index_list[:, 0]:
                    #     bbox = preds[index, 2:6]
                    #     intersect = bbox_intersection(bbox, label)
                    #     if intersect > max_intersection:
                    #         max_intersection = intersect
                    #         target_pred = bbox
                    # cover_rate = max_intersection/bbox_area(label)
                    # magnification = bbox_area(target_pred)/bbox_area(label)
                    # position_error = bbox_distance(target_pred, label)
                    # if (cover_rate>0.6 and magnification<2 and position_error<15) or (cover_rate>0.7 and magnification<8 and position_error<25):
                    #         lose_frames = 0
                    #         atr += 1
                    # else:
                    #     lose_frames += 1

                    # Consider right identity
                    for index in index_list[:, 0]:
                        if preds[index, 1] == target_identity:
                            target_pred = preds[index, 2:6]
                            intersect = bbox_intersection(target_pred, label)
                            cover_rate = intersect/bbox_area(label)
                            magnification = bbox_area(target_pred)/bbox_area(label)
                            position_error = bbox_distance(target_pred, label)
                            if (cover_rate>0.5 and magnification<3 and position_error<15) or (cover_rate>0.7 and magnification<8 and position_error<30):
                                lose_frames = 0
                                atr += 1
                            else:
                                lose_frames += 1
                else:
                    # print("Lost frame:", fi+1)
                    lose_frames += 1
            else:
                intersect = bbox_intersection(preds[fi], label)
                cover_rate = intersect/bbox_area(label)
                magnification = bbox_area(preds[fi])/bbox_area(label)
                position_error = bbox_distance(preds[fi], label)
                if (cover_rate>0.5 and magnification<3 and position_error<15) or (cover_rate>0.7 and magnification<8 and position_error<30):
                    lose_frames = 0
                    atr += 1
                else:
                    lose_frames += 1

            # if lose_frames > 30:
            #     # Consecutively losing tracks for more than some threshold
            #     break
        atr /= (nb_frame - 1)
        data.append([splitext(basename(annot))[0], "{:.2f}%".format(atr * 100), nb_frame])
        total_atr += atr

    with open('tracked_ratio/' + filename + '.csv', "w") as f:
        w = csv.writer(f)
        w.writerows(data)
    
    total_atr /= nb_video
    
    return total_atr

def ATR(ann, trk_results, trk_names):
    print("Average tracked ratio:")
    for trk_r, trk_name in zip(trk_results, trk_names):
        if trk_name.find('SORT') != -1 or trk_name.find('YOLO') != -1:
            isMOTformat = True
        else:
            isMOTformat = False
        atr = average_tracked_ratio(ann, trk_r, isMOTformat, trk_name)
        print('{:.3f}%'.format(atr*100), trk_name)

# Average Tracked Ratio
###################################################################################

def overlap_precision(ann, trks, threshold, isMOTformat):
    nb_video = len(ann)
    total_precision = 0.0
    for i, annot in enumerate(ann):
        if basename(annot)[:-4] != basename(trks[i])[:-4]:
            raise ValueError("Wrong annotation and track correspondence.")

        labels = np.loadtxt(annot, delimiter=',')
        if isMOTformat:
            mot_results = np.loadtxt(trks[i], delimiter=',')
        else:
            results = np.loadtxt(trks[i], delimiter=',')

        nb_frame = len(labels)

        precision = 0.0

        for fi, label in enumerate(labels):
            if fi == 0:
                # Tracking start
                if isMOTformat:
                    target_identity = 1
                    index_list = np.argwhere(mot_results[:, 0] == (fi+1))
                    if index_list.shape[0] != 0:
                        max_intersection = 0.0 
                        for index in index_list[:, 0]:
                            bbox = mot_results[index, 2:6]
                            intersect = bbox_intersection(bbox, label)
                            if intersect > max_intersection:
                                max_intersection = intersect
                                target_pred = bbox
                                target_identity = mot_results[index, 1]
                continue

            if isNAN(labels[fi]) is True:
                # No target in the frame
                continue

            if isMOTformat:
                index_list = np.argwhere(mot_results[:, 0] == (fi+1))
                if index_list.shape[0] != 0:
                    # Only consider maximum IOU
                    # max_iou = 0.0
                    # for index in index_list[:, 0]:
                    #     bbox = mot_results[index, 2:6]
                    #     iou = bbox_iou(bbox, label)
                    #     if iou > max_iou:
                    #         max_iou = iou
                    # if max_iou > threshold:
                    #     precision += 1

                    # Consider right identity
                    for index in index_list[:, 0]:
                        if mot_results[index, 1] == target_identity:
                            bbox = mot_results[index, 2:6]
                            iou = bbox_iou(bbox, label)
                            if iou > threshold:
                                precision += 1
            else:
                iou = bbox_iou(results[fi], label)
                if iou > threshold:
                    precision += 1

        precision /= (nb_frame - 1)

        total_precision += precision
    
    total_precision /= nb_video
    
    return total_precision

#################################################################################################
# Create json file for OTB results

def mkjson(trk_name, success_rate_list):
    data = {
        "tracker": trk_name,
        "overlap": None,
        "seqs": None,
        "overlapScores": None,
        "successRateList": success_rate_list,
        "desc": "All attributes",
        "name": "ALL",
        "evalType": "OPE",
        "errorNum": None
    }

    json_result_dir = 'results/OPE/' + trk_name
    if not os.path.exists(json_result_dir):
        os.mkdir(json_result_dir)
    json_result_dir += '/scores_uav123'
    if not os.path.exists(json_result_dir):
        os.mkdir(json_result_dir)

    json_file = os.path.join(json_result_dir, "ALL.json")

    with open(json_file, 'w') as f:
        json.dump(data, f)


def create_otb_results(ann, trk_results, trk_names):
    t = np.linspace(0.0, 1.0, 21)

    for trk_r, trk_name in zip(trk_results, trk_names):
        json_result_dir = 'results/OPE/' + trk_name + '/scores_uav123'
        json_file = os.path.join(json_result_dir, "ALL.json")
        if os.path.exists(json_file):
            continue

        print("Processing:", trk_name)

        success_rate_list = []
        for i, threshold in enumerate(t):
            if trk_name.find('SORT') != -1:
                isMOTformat = True
            else:
                isMOTformat = False
            success_rate_list.append(overlap_precision(ann, trk_r, threshold, isMOTformat))
        mkjson(trk_name, success_rate_list)

#################################################################################################

def success_plot_auc(ann, trk_results, trk_names):
    """ Draw success plot.
        Arguments:
            ann: a list of paths to all the video annotations
            trk_results: a list of paths to all the video tracked results
            trk_names: a list of names to each tracker
    """

    fig = plt.figure("Success plot")
    t = np.linspace(0.0, 1.0, 21)
    aucs = []
    handles = []
    for trk_r, trk_name in zip(trk_results, trk_names):
        s = np.zeros_like(t)
        for i, threshold in enumerate(t):
            if trk_name.find('SORT') != -1 or trk_name.find('YOLO') != -1:
                isMOTformat = True
            else:
                isMOTformat = False
            s[i] = overlap_precision(ann, trk_r, threshold, isMOTformat)

        auc_score = np.mean(s)
        aucs.append(auc_score)
        legend_str = trk_name + " [{:.3f}]".format(auc_score)
        line, = plt.plot(t, s, label=legend_str)
        handles.append(line)

    trk_order = np.argsort(np.array(aucs))[::-1]
    handles = [handles[order] for order in trk_order]
    plt.legend(handles=handles, loc='upper right')

    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    plt.title('Success plots of OPE on UAV123 - Person only')
    plt.grid(color='#101010', alpha=0.5, ls=':')

    plt.savefig('comparison.png')
    plt.show()


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

    annots = sorted(glob.glob((data['annot_folder'] + "*")))
    sort_nicely(annots)
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

    # total_avg_iou, total_lost = average_IOU(annots, trks, "DSST")
    # print("Total average IOU = {}".format(total_avg_iou))
    # print("Total lost track  = {}".format(total_lost))

    # trk_names = ["DSST", "YOLOv3+SORT", "YOLOv3+SORT(ReID)", "YOLOv2+SORT", "YOLOv2+SORT(ReID)"]
    # trk_results = [dsst_r, sort_r, rid_sort_r, y2_sort_r, y2_ridsort_r]

    names = ["YOLOv2", "YOLOv3", "YOLOv2+SORT", "YOLOv3+SORT", "YOLOv2+SORT(ReID)", "YOLOv3+SORT(ReID)"]
    results = [yolo2_det_r, yolo3_det_r, y2_sort_r, sort_r, y2_ridsort_r, rid_sort_r]

    # ATR(annots, [dsst_r], ["DSST"])
    # ATR(annots, [dsst_r, sort_r, ukf_r], trk_names)
    # ATR(annots, [dsst_r, sort_r, rid_sort_r], trk_names)
    # ATR(annots, results, names)
    
    # AOS(annots, [yolo2_det_r, yolo3_det_r, y2_sort_r, sort_r, y2_ridsort_r, rid_sort_r], names)
    AOS(annots, [dsst_r], ['DSST'])

    # success_plot_auc(annots, [yolo2_det_r, yolo3_det_r], det_names)
    # success_plot_auc(annots, [dsst_r, sort_r], trk_names)

    # create_otb_results(annots, trk_results, trk_names)
