""" This script stores the absolute path of data and tracking results.
    You may need to modify the path when you are using the codes in a new device.
"""
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
    'y2_nft_det_results': '/home/peng/darknetv2/det_mot(before_ft)/',
    'yolo3_det_results': '/home/peng/darknet/det_mot/',
    'y3_nft_det_results': '/home/peng/darknet/det_mot(before_ft)/'
}