import os
import glob
import re
from os.path import basename, splitext


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    return l.sort(key=alphanum_key)


def get_data_lists(data):
    """ Prepare data to be tracked
        Arguments:
            data: config of the following form:
            {
                'image_folder': 'path/to/images/',
                'annot_folder': 'path/to/annotations',
            }
        Returns:
            video_folders: list of video folder paths
            video_annotations: list of annotation file paths
    """

    if not os.path.exists(data['image_folder']):
        raise IOError("Wrong image folder path:", data['image_folder'])
    else:
        print("Data folder:", data['image_folder'])
    if not os.path.exists(data['annot_folder']):
        raise IOError("Wrong annotation folder path:", data['annot_folder'])
    else:
        print("Annotations folder:", data['annot_folder'])

    # Get the annotations as a list: [video1ann.txt, video2ann.txt, video3ann.txt, ...]
    video_annots = sorted(glob.glob((data['annot_folder'] + "*")))
    sort_nicely(video_annots)

    video_folders = []

    for i, annot_path in enumerate(video_annots):
        video_name = splitext(basename(annot_path))[0]   # Get the file name from its full path
        video_folder = os.path.join(data['image_folder'], video_name)
        if not os.path.exists(video_folder):
            raise IOError("Video folder does not exit:", video_folder)        
        video_folders.append(video_folder)

    return video_annots, video_folders
