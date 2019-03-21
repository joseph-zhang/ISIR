#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.cm
import cv2

def single_colorize(value, mask_off=None, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
    - value: 2D Tensor of shape [height, width] or 3D Tensor of shape [height, width, 1].
    - vmin: the minimum value of the range used for normalization. (Default: value minimum)
    - vmax: the maximum value of the range used for normalization. (Default: value maximum)
    - cmap: a valid cmap named for use with matplotlib's `get_cmap`. (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```

    Returns a 3D tensor of shape [height, width, 3].
    """
    # deal with nan value if mask_off is set
    if mask_off is not None:
        valid_mask = tf.not_equal(value, mask_off)
        nan_mask = tf.logical_not(valid_mask)
        valid_indices = tf.where(valid_mask)
        valid_vals = tf.gather_nd(params=value, indices=valid_indices)
        vmean = tf.reduce_mean(valid_vals)

        mask_true = tf.cast(valid_mask, tf.float32)
        mask_false = tf.cast(nan_mask, tf.float32)
        value = value * mask_true + mask_false * vmean
    else:
        pass

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')

    # get colormap
    colors = cm(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)

    value = tf.gather(colors, indices)
    return value


def create_pascal_label_colormap():
    """
    Creates a label colormap used in PASCAL VOC segmentation benchmark.
    :return: A Colormap for visualizing segmentation results.
    -> the number of classes of this colormap is up to 256.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


def semantic_colorize(segmap):
    """
    For input segmentation map visualization.
    :segmap: semantic of gt or pred_out, [batch_size, img_sz[0], img_sz[1], NUM_CATEGORIES]
    :return: colorized segmap with size [batch_size, img_sz[0], img_sz[1], 3]
    """
    colormap = create_pascal_label_colormap()
    colormap_tensor = tf.constant(colormap, dtype = tf.uint8)
    value = tf.squeeze(segmap)
    res = tf.gather(colormap_tensor, tf.argmax(value, axis = -1))
    return res


def gradient(image):
    """
    This function calculates the gradient of input image.
    Note that the form of input image is a 4D tensor.
    Return a two-element list, every element is a 4D tensor.
    """
    # define sobel filter kernels
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    D_dx = tf.nn.conv2d(image, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    D_dy = tf.nn.conv2d(image, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')

    return [D_dx, D_dy]


def cosine(a, b, dim):
    """
    This function calculates cosine similarity between two 4D float tensors.
    Inputs are 4D tensors, and the calculation will be implemented on dim.
    Return a 3D float tensor.
    """
    normalized_a = tf.nn.l2_normalize(a, dim)
    normalized_b = tf.nn.l2_normalize(b, dim)
    cos_similarity = tf.reduce_sum(tf.multiply(normalized_a, normalized_b), axis=dim)
    return cos_similarity


def fspecial_gauss(size, sigma):
    """
    Function to get gaussian filter calculator, just like fspecial(gaussian).
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))

    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    """
    A tensorflow function to calculate SSIM over two images.
    Note that the input image img1 and img2 are both 4D tensors.
    """
    window = fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID') - mu1_mu2

    if cs_map:
        value = (
            ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)),
            (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
        )
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def get_labels_from_annotation(annotation_tensor, class_labels):
    """Returns tensor of size (width, height, num_classes) derived from annotation tensor.
    The function returns tensor that is of a size (width, height, num_classes) which
    is derived from annotation tensor with sizes (width, height) where value at
    each position represents a class. The functions requires a list with class
    values like [0, 1, 2 ,3] -- they are used to derive labels. Derived values will
    be ordered in the same way as the class numbers were provided in the list. Last
    value in the aforementioned list represents a value that indicate that the pixel
    should be masked out. So, the size of num_classes := len(class_labels) - 1.
    Parameters
    ----------
    annotation_tensor : Tensor of size (width, height)
        Tensor with class labels for each element
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.
    Returns
    -------
    labels_2d_stacked : Tensor of size (width, height, num_classes).
        Tensor with labels for each pixel.
    """

    # Last value in the classes list should show
    # which number was used in the annotation to mask out
    # the ambigious regions or regions that should not be
    # used for training.
    # TODO: probably replace class_labels list with some custom object
    valid_entries_class_labels = class_labels[:-1]

    # Stack the binary masks for each class
    labels_2d = list(map(lambda x: tf.equal(annotation_tensor, x), valid_entries_class_labels))

    # Perform the merging of all of the binary masks into one matrix, generating one-hot
    labels_2d_stacked = tf.stack(labels_2d, axis=2)

    # Convert tf.bool to tf.float
    # Later on in the labels and logits will be used
    # in tf.softmax_cross_entropy_with_logits() function
    # where they have to be of the float type.
    labels_2d_stacked_float = tf.to_float(labels_2d_stacked)

    return labels_2d_stacked_float

def get_labels_from_annotation_batch(annotation_batch_tensor, class_labels):
    """Returns tensor of size (batch_size, width, height, num_classes) derived
    from annotation batch tensor. The function returns tensor that is of a size
    (batch_size, width, height, num_classes) which is derived from annotation tensor
    with sizes (batch_size, width, height) where value at each position represents a class.
    The functions requires a list with class values like [0, 1, 2 ,3] -- they are
    used to derive labels. Derived values will be ordered in the same way as
    the class numbers were provided in the list. Last value in the aforementioned
    list represents a value that indicate that the pixel should be masked out.
    So, the size of num_classes len(class_labels) - 1.
    Parameters
    ----------
    annotation_batch_tensor : Tensor of size (batch_size, width, height)
        Tensor with class labels for each element
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.
    Returns
    -------
    batch_labels : Tensor of size (batch_size, width, height, num_classes).
        Tensor with labels for each batch.
    """

    batch_labels = tf.map_fn(fn=lambda x: get_labels_from_annotation(annotation_tensor=x, class_labels=class_labels),
                             elems=annotation_batch_tensor,
                             dtype=tf.float32)

    return batch_labels

def get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor, class_labels):
    """Returns tensor of size (num_valid_eintries, 3).
    Returns tensor that contains the indices of valid entries according
    to the annotation tensor. This can be used to later on extract only
    valid entries from logits tensor and labels tensor. This function is
    supposed to work with a batch input like [b, w, h] -- where b is a
    batch size, w, h -- are width and height sizes. So the output is
    a tensor which contains indexes of valid entries. This function can
    also work with a single annotation like [w, h] -- the output will
    be (num_valid_eintries, 2).
    Parameters
    ----------
    annotation_batch_tensor : Tensor of size (batch_size, width, height)
        Tensor with class labels for each batch
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.
    Returns
    -------
    valid_labels_indices : Tensor of size (num_valid_eintries, 3).
        Tensor with indices of valid entries
    """

    # Last value in the classes list should show
    # which number was used in the annotation to mask out
    # the ambigious regions or regions that should not be
    # used for training.
    # TODO: probably replace class_labels list with some custom object
    mask_out_class_label = class_labels[-1]

    # Get binary mask for the pixels that we want to
    # use for training. We do this because some pixels
    # are marked as ambigious and we don't want to use
    # them for trainig to avoid confusing the model
    valid_labels_mask = tf.not_equal(annotation_batch_tensor,
                                     mask_out_class_label)

    valid_labels_indices = tf.where(valid_labels_mask)

    return tf.to_int32(valid_labels_indices)


def get_valid_logits_and_labels(annotation_batch_tensor,
                                logits_batch_tensor,
                                class_labels):
    """Returns two tensors of size (num_valid_entries, num_classes).
    The function converts annotation batch tensor input of the size
    (batch_size, height, width) into label tensor (batch_size, height,
    width, num_classes) and then selects only valid entries, resulting
    in tensor of the size (num_valid_entries, num_classes). The function
    also returns the tensor with corresponding valid entries in the logits
    tensor. Overall, two tensors of the same sizes are returned and later on
    can be used as an input into tf.softmax_cross_entropy_with_logits() to
    get the cross entropy error for each entry.
    Parameters
    ----------
    annotation_batch_tensor : Tensor of size (batch_size, width, height)
        Tensor with class labels for each batch
    logits_batch_tensor : Tensor of size (batch_size, width, height, num_classes)
        Tensor with logits. Usually can be achived after inference of fcn network.
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.
    Returns
    -------
    (valid_labels_batch_tensor, valid_logits_batch_tensor) : Two Tensors of size (num_valid_eintries, num_classes).
        Tensors that represent valid labels and logits.
    """

    labels_batch_tensor = get_labels_from_annotation_batch(annotation_batch_tensor=annotation_batch_tensor,
                                                           class_labels=class_labels)

    valid_batch_indices = get_valid_entries_indices_from_annotation_batch(
        annotation_batch_tensor=annotation_batch_tensor,
        class_labels=class_labels)

    valid_labels_batch_tensor = tf.gather_nd(params=labels_batch_tensor, indices=valid_batch_indices)

    valid_logits_batch_tensor = tf.gather_nd(params=logits_batch_tensor, indices=valid_batch_indices)

    return valid_labels_batch_tensor, valid_logits_batch_tensor


def get_annotation_from_label_batch(batch_labels):
    return tf.to_int32(tf.argmax(batch_labels, axis=-1))
