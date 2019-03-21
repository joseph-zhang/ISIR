import os
import tifffile
import numpy as np
from glob import glob
from tqdm import tqdm

SINGLE_IMG_DIR = None
SINGLE_IMG_STR = "_AGL"
SOURCE_IMG_EXT = "tif"
RESULT_FILE_NAME = 'height_info.txt'

def get_img_paths(img_dir):
    return glob(os.path.join(img_dir, "*%s*.%s" % (SINGLE_IMG_STR, SOURCE_IMG_EXT)))


def deal_with_nan(value):
    valid_mask = np.invert(np.isnan(value))
    valid_indices = np.where(valid_mask)
    mean_wo_nan = np.mean(value[valid_indices])
    value[np.isnan(value)] = mean_wo_nan
    return value


def get_height_info(img_dir=SINGLE_IMG_DIR, detail=False):
    img_paths = get_img_paths(img_dir)
    max_list = []
    min_list = []
    for imgPath in tqdm(img_paths):
        img = deal_with_nan(tifffile.imread(imgPath))
        max_list.append(np.max(img))
        min_list.append(np.min(img))
    max_height = np.max(max_list)
    min_height = np.min(min_list)

    if detail == True:
        with open(RESULT_FILE_NAME, 'w') as f:
            f.write(str(max_list))
            f.write(str(min_list))
            f.write(str([max_height, min_height]))
        print("Detailed height info has generated -> {}".format(RESULT_FILE_NAME))
    else:
        print("max_val: {}, min_val: {}".format(max_height, min_height))
    return [max_height, min_height]


if __name__ == '__main__':
    get_height_info(detail=True)
