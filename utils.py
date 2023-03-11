import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage.color import rgb2lab, lab2rgb
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, RepeatVector,Reshape
from tensorflow.keras.layers import  Dense, Flatten, Input, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image


def loadImagesToArray(dir_path, num_of_img=-1):
    """
    dir_path : path of directory from which images will be imported
    num_of_imgs (Integer): number of images to be imported from the directory if not
                given than all images will be imported
    search_inside (boolean, default : False) : If true all images inside that directory
                along with the images in subdirectory will be added to output array
    """
    images = []
    count = -1
    images_path = glob.glob(str(dir_path) + '/' +  '*.jpeg')
    rdm = random.sample(images_path, num_of_img)
    for filename in rdm:          
      img = img_to_array(Image.open(filename).resize((256,256)))
      if img.shape[0] != 256 or img.shape[1] != 256 or img.shape[2] != 3:
          continue
      images.append(img)
    return np.array(images, dtype=float) / 255.0

def custom_img_to_array(filename):
    img = img_to_array(Image.open(filename).resize((256,256)))
    return np.array(img, dtype=float) / 255.0


def RGB2GRAY(img, add_channel_dim=False):
    conv_matrix = np.array([0.212671, 0.715160, 0.072169])
    gray_img = img @ conv_matrix
    if add_channel_dim == True:
        return gray_img.reshape(np.array([*list(gray_img.shape), 1]))
    else:
        return gray_img


def RGB2ab(img, use_skimage=True):
    """
    Refrences
    * https://en.wikipedia.org/wiki/Lab_color_space
    * https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py#L990-L1050
    """
    if use_skimage == False:

        def finv(cie):
            cond = cie > 0.008856
            cie[cond] = np.cbrt(cie[cond])
            cie[~cond] = 7.787 * cie[~cond] + 16.0 / 116.0
            return cie

        conv_matrix = np.array(
            [
                [0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227],
            ]
        )
        CIE = np.matmul(img, conv_matrix.T)
        CIE[0] = CIE[0] / 0.95047
        CIE[2] = CIE[2] / 1.08883
        CIE = finv(CIE)
        x, y, z = CIE[..., 0], CIE[..., 1], CIE[..., 2]
        a = (500 * (x - y) + 127) / 255.0
        b = (200 * (y - z) + 127) / 255.0
        return np.concatenate([x[..., np.newaxis] for x in [a, b]], axis=-1)
    else:
        Lab = rgb2lab(img)
        a = (Lab[..., 1] + 127) / 255.0
        b = (Lab[..., 2] + 127) / 255.0
        return np.concatenate([x[..., np.newaxis] for x in [a, b]], axis=-1)


def Lab2RGB(gray, ab):
    ab = ab * 255.0 - 127
    gray = gray * 100
    Lab = np.concatenate(
        [x[..., np.newaxis] for x in [gray[..., 0], ab[..., 0], ab[..., 1]]], axis=-1
    )
    return lab2rgb(Lab)