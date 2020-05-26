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

def conv_network(point_cloud, is_training, scope, channels_out=1024, bn_decay=None):
    
    point_cloud = tf.expand_dims(point_cloud, 2)
    
    net, kernel = tf_util.inception(point_cloud, 64, scope=f'{scope}_1', kernel_heights=[1, 3, 5, 7], dilations=[3, 1],
                    kernel_widths=[1, 1, 1, 1], kernels_fraction=[2, 3, 2, 1], bn=True, bn_decay=bn_decay, 
                    is_training=is_training, return_kernel=True)
    net = tf_util.max_pool2d(net, kernel_size=[2, 1], scope=f'{scope}_1_max_pool', stride=[2, 1], padding='SAME')

    net = tf_util.inception(net, 128, scope=f'{scope}_2', kernel_heights=[1, 3, 5, 7], dilations=[3, 1],
                    kernel_widths=[1, 1, 1, 1], kernels_fraction=[2, 3, 2, 1], bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.max_pool2d(net, kernel_size=[2, 1], scope=f'{scope}_2_max_pool', stride=[2, 1], padding='SAME')

    net = tf_util.inception(net, 128, scope=f'{scope}_3', kernel_heights=[1, 3, 5, 7], dilations=[3, 1],
                    kernel_widths=[1, 1, 1, 1], kernels_fraction=[2, 3, 2, 1], bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.max_pool2d(net, kernel_size=[2, 1], scope=f'{scope}_3_max_pool', stride=[2, 1], padding='SAME')

    net = tf_util.inception(net, 256, scope=f'{scope}_4', kernel_heights=[1, 3, 5, 7], dilations=[3, 1],
                    kernel_widths=[1, 1, 1, 1], kernels_fraction=[2, 3, 2, 1], bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.max_pool2d(net, kernel_size=[2, 1], scope=f'{scope}_4_max_pool', stride=[2, 1], padding='SAME')

    net = tf_util.inception(net, 512, scope=f'{scope}_5', kernel_heights=[1, 3, 5, 7], dilations=[3, 1],
                    kernel_widths=[1, 1, 1, 1], kernels_fraction=[2, 3, 2, 1], bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.max_pool2d(net, kernel_size=[2, 1], scope=f'{scope}_5_max_pool', stride=[2, 1], padding='SAME')

    net = tf_util.inception(net, channels_out, scope=f'{scope}_6', kernel_heights=[1, 3, 5, 7], dilations=[3, 1],
                    kernel_widths=[1, 1, 1, 1], kernels_fraction=[2, 3, 2, 1], bn=True, bn_decay=bn_decay, is_training=is_training)
    

    net = tf_util.avg_pool2d(net, kernel_size=[net.shape[1], 1], scope=f'{scope}_GAP', stride=[net.shape[1], 1], padding='SAME')

    net = tf.squeeze(net)

    return net, kernel



def classification_head(net, is_training, scope, n_classes, bn_decay, weight_decay):
    net = tf_util.dropout(net, rate=0.6, is_training=is_training, scope=scope+'_dp1') # my change
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope=scope+'_fc1', bn_decay=bn_decay, 
                                  weight_decay=weight_decay)
    net = tf_util.dropout(net, rate=0.6, is_training=is_training, scope=scope+'_dp2')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope=scope+'_fc2', bn_decay=bn_decay, 
                                  weight_decay=weight_decay)
    net = tf_util.dropout(net, rate=0.6, is_training=is_training, scope=scope+'_dp3')
    net = tf_util.fully_connected(net, n_classes, activation_fn=None, scope=scope+'_fc3')
    return net


def get_model(point_cloud, is_training, n_classes, bn_decay=None, weight_decay=None, 
              extractor=True, temporal=True, **kwargs):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    channels_size = point_cloud.get_shape()[2].value
    end_points = {}

    end_points['input_point_cloud'] = point_cloud
    
    conv, ker = conv_network(point_cloud, is_training, 'conv', 1024, bn_decay)

    # classification from entire network
    final_pred = classification_head(conv, is_training, 'cls_net', n_classes, bn_decay, weight_decay)
    return final_pred, [], end_points


def get_loss(pred, site_preds, label, end_points, aux_loss_weights):
    """ pred: B*NUM_CLASSES,
        label: B, 
        aux_loss_weights ignored
    """
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss) * 1
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)

    return classify_loss, []


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
