import os
import tifffile
import numpy as np
import matplotlib
import matplotlib.cm
import matplotlib.image as mpimg
from glob import glob
from tqdm import tqdm

SOURCE_IMG_DIR = None
COLOR_MAP_DIR = os.path.join(SOURCE_IMG_DIR, 'colorized_msi/')

SOURCE_IMG_STR = '_MSI'
SOURCE_IMG_EXT = "tif"
COLOR_IMG_EXT = "png"
COLOR_MAP = None
NUM_CHANNELS = 8
RGB_MODE = True


def colorize(value, vmin=None, vmax=None):
    valid_mask = np.invert(np.isnan(value))
    valid_indices = np.where(valid_mask)
    mean_wo_nan = np.mean(value[valid_indices])
    value[np.isnan(value)] = mean_wo_nan

    vmin = np.min(value) if vmin is None else vmin
    vmax = np.max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)

    value = np.squeeze(value)
    indices = np.round(value * 255).astype(np.int32)

    cm = matplotlib.cm.get_cmap(COLOR_MAP if COLOR_MAP is not None else 'gray')
    colors = cm(np.arange(256))[:,:3].astype(np.float32)

    value = colors[indices]
    return value


def get_img_paths():
    return glob(os.path.join(SOURCE_IMG_DIR, "*%s*.%s" % (SOURCE_IMG_STR, SOURCE_IMG_EXT)))


if __name__ == '__main__':
    img_paths = get_img_paths()

    if not os.path.isdir(COLOR_MAP_DIR):
        os.makedirs(COLOR_MAP_DIR)

    print("processing start =>\n")
    for imgPath in tqdm(img_paths):
        img_name = os.path.split(imgPath)[-1]
        img = tifffile.imread(imgPath)
        if RGB_MODE is True:
            curr_name = img_name
            out_name = curr_name.replace(SOURCE_IMG_EXT, COLOR_IMG_EXT)
            outPath = os.path.join(COLOR_MAP_DIR, out_name)
            puls = lambda img: (img-np.min(img))/(np.max(img) - np.min(img))
            rgb_img = np.concatenate([puls(img[:,:,4:5]),  # R
                                      puls(img[:,:,2:3]),  # G
                                      puls(img[:,:,1:2])], # B
                                     axis=-1)
            mpimg.imsave(outPath, rgb_img)

        else:
            for i in range(NUM_CHANNELS):
                colorized_img = colorize(img[:,:,i])
                curr_name = img_name
                out_name = curr_name.replace('.'+SOURCE_IMG_EXT, '_'+str(i)+'.'+COLOR_IMG_EXT)
                outPath = os.path.join(COLOR_MAP_DIR, out_name)
                mpimg.imsave(outPath, colorized_img)
