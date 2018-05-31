from os.path import basename, splitext
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import csv

from utils import sort_nicely


def isNAN(bbox):
    for value in bbox.flatten():
        if math.isnan(value):
            return True

def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3  

def bbox_iou(box1, box2):
    """ Compute IOU between two bboxes in the form [x1,y1,w,h]
    """
    x1_min = box1[0]
    x1_max = x1_min + box1[2]
    y1_min = box1[1]
    y1_max = y1_min + box1[3]

    x2_min = box2[0]
    x2_max = x2_min + box2[2]
    y2_min = box2[1]
    y2_max = y2_min + box2[3]

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    
    intersect = intersect_w * intersect_h
    
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersect
    
    return float(intersect) / union

def average_IOU(annots, trks, file_name):
    nb_video = len(annots)
    total_frames = 0
    total_avg_iou = 0.0
    total_lost = 0.0
    data = []
    data.append(["video", "avg_iou", "#frame", "#lost"])
    
    for i, annot in enumerate(annots):
        if basename(annot)[:-4] != basename(trks[i])[:-4]:
            print("Annotations:", annot)
            print("Output:", trks)
            raise ValueError("Wrong annotation and track correspondence.")

        print("Evaluating %s." % basename(annot))

        labels = np.loadtxt(annot, delimiter=',')
        preds = np.loadtxt(trks[i], delimiter=',')

        nb_frame = len(labels)
        if nb_frame != len(preds):
            raise ValueError("Wrong annotation and track correspondence.")

        total_frames += nb_frame

        avg_iou = 0.0
        num_lost = 0.0
        for fi, label in enumerate(labels):
            if fi == 0:
                # Tracking start
                continue
            if isNAN(label) is True:
                # No target in the frame
                continue

            iou = bbox_iou(preds[fi], label)

            if iou == 0:
                num_lost += 1
            else:
                avg_iou += iou


        print("\tLost target = {} / {}".format(int(num_lost), nb_frame - 1))

        avg_iou /= (nb_frame - 1)

        print("\tAverage IOU = {:.2f}%".format(avg_iou * 100))
        print("\tNumber of frames = {}".format(nb_frame))

        data.append([splitext(basename(annot))[0], "{:.2f}%".format(avg_iou * 100), nb_frame, int(num_lost)])

        total_avg_iou += avg_iou
        total_lost += num_lost

        print("==================================================")
    
    with open(file_name +'.csv', "w") as f:
        w = csv.writer(f)
        w.writerows(data)
    
    total_avg_iou /= nb_video

    print("Total frames: {}".format(total_frames))
    print("==================================================")
    
    return total_avg_iou, total_lost


def overlap_precision(annots, trks, threshold, isMOTformat):
    nb_video = len(annots)
    total_precision = 0.0
    for i, annot in enumerate(annots):
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
                continue

            if isNAN(labels[fi]) is True:
                # No target in the frame
                continue

            if isMOTformat:
                index_list = np.argwhere(mot_results[:, 0] == (fi+1))
                if index_list.shape[0] != 0:
                    max_iou = 0.0
                    for index in index_list[:, 0]:
                        bbox = mot_results[index, 2:6]
                        iou = bbox_iou(bbox, label)
                        if iou > max_iou:
                            max_iou = iou
                    if max_iou > threshold:
                        precision += 1
            else:
                iou = bbox_iou(results[fi], label)
                if iou > threshold:
                    precision += 1

        precision /= (nb_frame - 1)

        total_precision += precision
    
    total_precision /= nb_video
    
    return total_precision


def success_plot_auc(ann, trackers, trk_names):
    fig = plt.figure("Success plot")
    t = np.linspace(0.0, 1.0, 30)
    legends = []
    for trk, trk_name in zip(trackers, trk_names):
        s = np.zeros_like(t)
        for i, threshold in enumerate(t):
            if trk_name.find('SORT') != -1:
                isMOTformat = True
            else:
                isMOTformat = False
            s[i] = overlap_precision(ann, trk, threshold, isMOTformat)
        plt.plot(t, s)
        auc_score = np.mean(s)
        legend_str = trk_name + " [{:.3f}]".format(auc_score)
        legends.append(legend_str)

    plt.legend(tuple(legends), loc='upper right')

    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    plt.title('Success plots of OPE on UAV123 - Person only')
    plt.grid(True)

    plt.savefig('comparison.png')
    plt.show()

if __name__ == '__main__':
    data = {
        'image_folder': '/home/peng/data/sort_data/images/',
        'annot_folder': '/home/peng/data/sort_data/annotations/',
        'dsst_tracked_results': '/home/peng/trackers/dsst_output/',
        'sort_tracked_results': '/home/peng/darknet/sort/kf_output/',
        'ukf_tracked_results': '/home/peng/darknet/sort/output/'
    }

    annots = sorted(glob.glob((data['annot_folder'] + "*")))
    sort_nicely(annots)
    dsst_r = sorted(glob.glob((data['dsst_tracked_results'] + "*")))
    sort_nicely(dsst_r)
    sort_r = sorted(glob.glob((data['sort_tracked_results'] + "*")))
    sort_nicely(sort_r)
    ukf_r = sorted(glob.glob((data['ukf_tracked_results'] + "*")))
    sort_nicely(ukf_r)

    # total_avg_iou, total_lost = average_IOU(annots, trks, "DSST")
    # print("Total average IOU = {}".format(total_avg_iou))
    # print("Total lost track  = {}".format(total_lost))

    trk_names = ["DSST", "YOLOv3+SORT(KF)", "YOLOv3+SORT(UKF)"]

    success_plot_auc(annots, [dsst_r, sort_r, ukf_r], trk_names)
