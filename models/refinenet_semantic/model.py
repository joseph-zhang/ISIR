#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ..resnet_v1 import resnet_v1_50
from ..resnet_v1 import resnet_arg_scope
from tensorflow.contrib import slim


def get_tensor_shape(inputs, dim = None):
    inputs_size = inputs.shape
    if dim is None:
        return inputs_size.as_list()
    else:
        return inputs_size[dim].value

def unpool_scale(inputs, scale):
    inputs_shape = get_tensor_shape(inputs)
    return tf.image.resize_bilinear(inputs, size=[inputs_shape[1]*scale, inputs_shape[2]*scale])

# upsampling to a specific size
def unpool(inputs, target_size):
    return tf.image.resize_bilinear(inputs, target_size)

# return the size of 1-dim and 2-dim of a 4D tensor
def get_tensor_pad_size(inputs):
    inputs_size = inputs.shape
    return (inputs_size[1].value, inputs_size[2].value)

# Residual Conv Unit (RCU) in RefineNet
def ResidualConvUnit(inputs, features=256, kernel_size=3):
    net = tf.nn.relu(inputs)
    net = slim.conv2d(net, features, kernel_size)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, features, kernel_size)
    net = tf.add(net, inputs)
    return net

# Multi-resolution Fusion
def MultiResolutionFusion(high_inputs=None, low_inputs=None, out_channels=256):
    g0 = slim.conv2d(high_inputs, out_channels, 3)

    if low_inputs is None:
        return g0
    else:
        low_size = get_tensor_pad_size(low_inputs)
        g0 = unpool(g0, low_size)
        g1 = slim.conv2d(low_inputs, out_channels, 3)

    return tf.add(g0, g1)

# Chained Residual Pooling
def ChainedResidualPooling(inputs, out_channels=256):
    net_relu = tf.nn.relu(inputs)
    net = slim.max_pool2d(net_relu, [5, 5], stride=1, padding='SAME')
    net = slim.conv2d(net, out_channels, 3)
    return tf.add(net, net_relu)

# RefineNet-m
def RefineBlock(high_inputs=None, low_inputs=None, scope_name = ""):
    with tf.variable_scope(scope_name) as sc:
        print(high_inputs.shape)
        rcu_high = ResidualConvUnit(high_inputs, features=256)
        rcu_low = ResidualConvUnit(low_inputs, features=256) if low_inputs is not None else None
        fuse = MultiResolutionFusion(rcu_high, rcu_low, out_channels=256)
        fuse_pooling = ChainedResidualPooling(fuse, out_channels=256)
        output = ResidualConvUnit(fuse_pooling, features=256)
        return output


# total model
def semantic_model(images, num_classes=5, weight_decay=1e-5, is_training=True):
    # get ResNet vars
    with slim.arg_scope(resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1_50(images,
                                          num_classes,
                                          is_training=is_training,
                                          global_pool=False,
                                          spatial_squeeze=False,
                                          output_stride=16,
                                          scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))

            g = [None, None, None, None]
            h = [None, None, None, None]
            r = [None, None, None, None]

            # pre-convolutions
            for i in range(4):
                h[i] = slim.conv2d(f[i], 256, 1)
                print('Shape of h_{} {}'.format(i, h[i].shape))

            # cascaded refine
            g[0] = RefineBlock(h[0], None, "refine_block0")
            g[1] = RefineBlock(g[0], h[1], "refine_block1")
            g[2] = RefineBlock(g[1], h[2], "refine_block2")
            g[3] = RefineBlock(g[2], h[3], "refine_block3")

            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):
                r[0] = slim.conv2d(g[0], 128, 3)
                r[0] = unpool(r[0], get_tensor_pad_size(g[3]))
                r[1] = slim.conv2d(g[1], 128, 3)
                r[1] = unpool(r[1], get_tensor_pad_size(g[3]))

                # concatenate refine maps
                fusion = tf.concat([tf.concat([r[0], r[1]], -1), g[3]], -1)
                fusion = slim.conv2d(fusion, 256, 3)

            pred = slim.conv2d(fusion, num_classes, 3, activation_fn=None, normalizer_fn=None)
            pred = unpool_scale(pred, 4)
            return pred
