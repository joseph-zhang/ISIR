import os
import tifffile
import numpy as np
import matplotlib
import matplotlib.cm
import matplotlib.image as mpimg
from glob import glob
from tqdm import tqdm

SEMANTIC_MODE = 0
SINGLE_MODE = 1

SINGLE_IMG_DIR = None
SEMANTIC_IMG_DIR = SINGLE_IMG_DIR

SINGLE_IMG_STR = "_AGL"
SEMANTIC_IMG_STR = "_CLS"

SOURCE_IMG_EXT = "tif"
COLOR_IMG_EXT = "png"

SINGLE_COLOR_DIR = os.path.join(SINGLE_IMG_DIR, 'colorized_single_pred/')
SEMANTIC_COLOR_DIR = os.path.join(SEMANTIC_IMG_DIR, 'colorized_semantic_pred/')

SINGLE_CMAP = 'jet'
#mode = SEMANTIC_MODE
mode = SINGLE_MODE
GLOBAL_HEIGHT_RANGE = True


def single_colorize(value, vmin=None, vmax=None):
    valid_mask = np.invert(np.isnan(value))
    valid_indices = np.where(valid_mask)
    mean_wo_nan = np.mean(value[valid_indices])
    value[np.isnan(value)] = mean_wo_nan

    vmin = np.min(value) if vmin is None else vmin
    vmax = np.max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)

    value = np.squeeze(value)
    indices = np.round(value * 255).astype(np.int32)

    cm = matplotlib.cm.get_cmap(SINGLE_CMAP if SINGLE_CMAP is not None else 'gray')
    colors = cm(np.arange(256))[:,:3].astype(np.float32)

    value = colors[indices]
    return value

def create_pascal_label_colormap():
    # use LAS Label, not converted label!
    color_map = np.zeros(shape = (256,3), dtype = np.uint8)
    color_map[2]  = [0, 0, 0]
    color_map[5]  = [128, 0, 0]
    color_map[6]  = [0, 128, 0]
    color_map[9]  = [128, 128, 0]
    color_map[17] = [0, 0, 128]
    return color_map

def semantic_colorize(value):
    colormap = create_pascal_label_colormap().astype(np.uint8)
    value = np.squeeze(value)
    res = colormap[value]
    return res

def get_img_paths():
    if mode == SINGLE_MODE:
        img_str = SINGLE_IMG_STR
        img_dir = SINGLE_IMG_DIR
    elif mode == SEMANTIC_MODE:
        img_str = SEMANTIC_IMG_STR
        img_dir = SEMANTIC_IMG_DIR
    else:
        pass
    return glob(os.path.join(img_dir, "*%s*.%s" % (img_str, SOURCE_IMG_EXT)))

if __name__ == '__main__':
    img_paths = get_img_paths()
    color_method = None
    out_dir = None

    if mode == SINGLE_MODE:
        if not os.path.isdir(SINGLE_COLOR_DIR):
            os.makedirs(SINGLE_COLOR_DIR)

        if GLOBAL_HEIGHT_RANGE == True:
            from height_range import get_height_info
            print("load height information...\n")
            max_height, min_height = get_height_info(SINGLE_IMG_DIR)
        else:
            max_height = None
            min_height = None

        color_method = lambda img: single_colorize(img, vmin=min_height, vmax=max_height)
        out_dir = SINGLE_COLOR_DIR
    elif mode == SEMANTIC_MODE:
        if not os.path.isdir(SEMANTIC_COLOR_DIR):
            os.makedirs(SEMANTIC_COLOR_DIR)
        color_method = semantic_colorize
        out_dir = SEMANTIC_COLOR_DIR
    else:
        pass

    print("processing start =>\n")
    for imgPath in tqdm(img_paths):
        img_name = os.path.split(imgPath)[-1]
        out_name = img_name.replace(SOURCE_IMG_EXT, COLOR_IMG_EXT)
        outPath = os.path.join(out_dir, out_name)
        img = tifffile.imread(imgPath)
        colorized_img = color_method(img)
        mpimg.imsave(outPath, colorized_img)
