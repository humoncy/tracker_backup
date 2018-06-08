import numpy as np
import math


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

def bbox_intersection(box1, box2):
    """ Compute intersection between two boxes
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
    
    intersection = intersect_w * intersect_h

    return intersection

def bbox_area(box):
    return float(box[2] * box[3])

def bbox_iou(box1, box2):
    """ Compute IOU between two bboxes in the form [x1,y1,w,h]
    """
    intersect = bbox_intersection(box1, box2)
    union = bbox_area(box1) + bbox_area(box2) - intersect
    
    return float(intersect) / union

def bbox_distance(box1, box2):
    box1_center_x = box1[0] + box1[2]/2
    box1_center_y = box1[1] + box1[3]/2
    box2_center_x = box2[0] + box2[2]/2
    box2_center_y = box2[1] + box2[3]/2

    dx = abs(box1_center_x - box2_center_x)
    dy = abs(box1_center_y - box2_center_y)

    return math.sqrt(dx**2 + dy**2)
    
# [x1 y1 w h] to [x1 y1 x2 y2]
def xywh_to_xyxy(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    w = bbox[2]
    h = bbox[3]

    return [x1, y1, x1+w, y1+h]