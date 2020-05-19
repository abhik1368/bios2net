"""
    PointNet++ Model for point clouds classification
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module

def placeholder_inputs(batch_size, num_point, num_channels):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_channels))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def conv_net(point_cloud, is_training, bn_decay=None):

    net = tf_util.inception(point_cloud, 256, scope='temp1_1', kernel_heights=[1, 3, 5, 7],
                    kernel_widths=[1, 1, 1, 1], kernels_fraction=[2, 2, 2, 2], bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.max_pool2d(net, kernel_size=[2, 1], scope='temp1_1_max_pool', stride=[2, 1], padding='SAME')

    net = tf_util.inception(net, 512, scope='temp1_2', kernel_heights=[1, 3, 5, 7],
                    kernel_widths=[1, 1, 1, 1], kernels_fraction=[2, 2, 2, 2], bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.max_pool2d(net, kernel_size=[2, 1], scope='temp1_1_max_pool', stride=[2, 1], padding='SAME')

    net = tf_util.inception(net, 1024, scope='temp1_2', kernel_heights=[1, 3, 5, 7],
                    kernel_widths=[1, 1, 1, 1], kernels_fraction=[2, 2, 2, 2], bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.max_pool2d(net, kernel_size=[2, 1], scope='temp1_1_max_pool', stride=[2, 1], padding='SAME')

    net = tf_util.avg_pool2d(net, kernel_size=[net.shape[1], 1], scope='temp1_GAP', stride=[net.shape[1], 1], padding='SAME')

    net = tf.squeeze(net)

    return net


def get_model(point_cloud, is_training, n_classes, bn_decay=None, weight_decay=None, extractor=True, **kwargs):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    channels_size = point_cloud.get_shape()[2].value
    end_points = {}
    if point_cloud.shape[2] > 3:
        l0_xyz = point_cloud[:, :, :3]
        l0_points = point_cloud[:, :, 3:]
    else:
        l0_xyz = point_cloud
        l0_points = None
    end_points['l0_xyz'] = l0_xyz

    if extractor:
        input_image = tf.expand_dims(point_cloud, 2)

        net, conv_ker = tf_util.inception(input_image, 64, scope='seq_conv1',
                                kernel_heights=[1, 3, 5, 7], kernel_widths=[1, 1, 1, 1],
                                kernels_fraction=[2, 2, 2, 2],
                                return_kernel=True,
                                bn=True, bn_decay=bn_decay,
                                is_training=is_training)
        end_points['conv_ker'] = conv_ker
        net, conv_ker = tf_util.inception(input_image, 32, scope='seq_conv2',
                                kernel_heights=[1, 3, 5, 7], kernel_widths=[1, 1, 1, 1],
                                kernels_fraction=[2, 2, 2, 2],
                                return_kernel=True,
                                bn=True, bn_decay=bn_decay,
                                is_training=is_training)

        conv_net = tf.squeeze(net)

        l0_points = tf.concat([l0_points, conv_net], axis=-1)
    end_points['points'] = l0_points

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.
    l1_xyz, l1_points, l1_indices, pt_ker = pointnet_sa_module(l0_xyz, l0_points, npoint=256, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)
    end_points['pt_ker'] = pt_ker

    temporal_1 = conv_net(tf.concat([l1_xyz, l1_points], axis=-1), is_training=is_training, bn_decay=bn_decay)

    l2_xyz, l2_points, l2_indices, _ = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=128, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

    l3_xyz, l3_points, l3_indices, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    net = tf.reshape(l3_points, [batch_size, -1])

    net = tf.concat([net, temporal_1], axis=-1)
    end_points['feature_vector'] = net

    # Fully connected layers
    net = tf_util.dropout(net, rate=0.6, is_training=is_training, scope='dp1') # my change
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    net = tf_util.dropout(net, rate=0.6, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay, weight_decay=weight_decay)
    net = tf_util.dropout(net, rate=0.6, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, n_classes, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
