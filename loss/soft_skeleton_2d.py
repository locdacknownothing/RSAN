from keras import layers as KL
from keras import backend as K
import numpy as np
import tensorflow as tf


def soft_erode(img):
    """[This function performs soft-erosion operation on a float32 2D image]

    Args:
        img ([float32]): [image to be soft eroded, shape (batch, height, width, channel)]

    Returns:
        [float32]: [the eroded image]
    """
    p1 = -KL.MaxPool2D(pool_size=(3, 1), strides=(1, 1), padding='same')(-img)
    p2 = -KL.MaxPool2D(pool_size=(1, 3), strides=(1, 1), padding='same')(-img)
    return tf.math.minimum(p1, p2)


def soft_dilate(img):
    """[This function performs soft-dilation operation on a float32 2D image]

    Args:
        img ([float32]): [image to be soft dilated, shape (batch, height, width, channel)]

    Returns:
        [float32]: [the dilated image]
    """
    return KL.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(img)


def soft_open(img):
    """[This function performs soft-open operation on a float32 2D image]

    Args:
        img ([float32]): [image to be soft opened]

    Returns:
        [float32]: [image after soft-open]
    """
    img = soft_erode(img)
    img = soft_dilate(img)
    return img


def soft_skel(img, iters):
    """[Soft skeletonization for 2D images]

    Args:
        img ([float32]): [input image, shape (batch, height, width, channel)]
        iters ([int]): [number of skeletonization iterations]

    Returns:
        [float32]: [skeletonized image]
    """
    img1 = soft_open(img)
    skel = tf.nn.relu(img - img1)

    for j in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = tf.nn.relu(img - img1)
        intersect = tf.math.multiply(skel, delta)
        skel += tf.nn.relu(delta - intersect)
    return skel
