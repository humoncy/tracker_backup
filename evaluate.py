from os.path import basename, splitext
import numpy as np
import glob
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

from utils import sort_nicely, isNAN
from bbox_utils import *


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
            print("Output:", trks[i])
            raise ValueError("Wrong annotation and track correspondence.")

        print("Evaluating %s." % basename(annot)[:-4])

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


def average_tracked_ratio(ann, trk_results, isMOTformat, filename):
    nb_video = len(ann)
    total_atr = 0.0
    data = []
    data.append(["video", "tracked_ratio", "#frame"])
    for i, annot in enumerate(tqdm(annots)):
        video_name = basename(annot)[:-4]
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
        if trk_name.find('SORT') != -1:
            isMOTformat = True
        else:
            isMOTformat = False
        atr = average_tracked_ratio(ann, trk_r, isMOTformat, trk_name)
        print('{:.3f}%'.format(atr*100), trk_name)


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


def success_plot_auc(ann, trk_results, trk_names):
    """ Draw success plot.
        Arguments:
            ann: a list of paths to all the video annotations
            trk_results: a list of paths to all the video tracked results
            trk_names: a list of names to each tracker
    """
    fig = plt.figure("Success plot")
    t = np.linspace(0.0, 1.0, 30)
    legends = []
    for trk_r, trk_name in zip(trk_results, trk_names):
        s = np.zeros_like(t)
        for i, threshold in enumerate(t):
            if trk_name.find('SORT') != -1:
                isMOTformat = True
            else:
                isMOTformat = False
            s[i] = overlap_precision(ann, trk_r, threshold, isMOTformat)
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

    # trk_names = ["DSST", "YOLOv3+SORT(KF)", "YOLOv3+SORT(UKF)"]
    trk_names = ["DSST", "YOLOv3+SORT"]

    # ATR(annots, [dsst_r], ["DSST"])
    # ATR(annots, [dsst_r, sort_r, ukf_r], trk_names)
    ATR(annots, [dsst_r, sort_r], trk_names)
    
    # success_plot_auc(annots, [dsst_r, sort_r, ukf_r], trk_names)
