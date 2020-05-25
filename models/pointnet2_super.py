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

def extractor_layers(point_cloud, n, is_training, bn_decay, bn=True, n_out=32, scope='extractor'):
    input_image = tf.expand_dims(point_cloud, 2)
    
    for i in range(n):
        net, conv_ker = tf_util.inception(input_image, n_out, scope=f'seq_conv{i}',
                            kernel_heights=[1, 3, 5, 7], kernel_widths=[1, 1, 1, 1],
                            kernels_fraction=[2, 2, 2, 2],
                            return_kernel=True,
                            bn=bn, bn_decay=bn_decay,
                            is_training=is_training)
        if i == 0:
            ker = conv_ker

    return tf.squeeze(net), ker

def get_model(point_cloud, is_training, n_classes, bn_decay=None, weight_decay=None, 
              extractor=True, temporal=True, **kwargs):
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
    end_points['input_point_cloud'] = point_cloud
    
    if extractor:
        print('==== USING EXTRACTOR ====')
        extr, ker = extractor_layers(point_cloud, 5, is_training, bn_decay, bn=True, n_out=32)
        end_points['extr'] = extr
        end_points['extr_ker'] = ker
        l0_points = tf.concat([l0_points, extr], axis=-1)

    # first pointnet layer
    l1_xyz, l1_points, l1_indices, pt_ker = pointnet_sa_module(
        l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], 
        mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, 
        scope='layer1', use_nchw=True) 
    end_points['pt_ker'] = pt_ker
    
    # second pointnet layer
    l2_xyz, l2_points, l2_indices, _ = pointnet_sa_module(
        l1_xyz, l1_points, npoint=128, radius=0.4, nsample=128, mlp=[128,128,256], 
        mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, 
        scope='layer2')
    
    # third potinent layer
    l3_xyz, l3_points, l3_indices, _ = pointnet_sa_module(
        l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], 
        mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, 
        scope='layer3')
    ptnet_out = tf.reshape(l3_points, [batch_size, -1])
        
    if temporal:
        print('==== USING TEMPORAL CONVOLUTION ====')
        # first temporal
        net = tf.concat([l1_xyz, l0_points[:, ::(num_point // 512), :], l1_points], axis=-1)
        end_points['temp1_input'] = net
        temporal_1, ker = conv_network(net, is_training=is_training, scope='temp1', channels_out=1024, bn_decay=bn_decay)
        end_points['temp1_ker'] = ker

        # second temporal
#         net = tf.concat([l2_xyz, l0_points[:, ::(l0_points.shape[1] // 128), :], l2_points], axis=-1)
#         end_points['temp2_input'] = net
#         temporal_2, ker = conv_network(net, is_training=is_training, scope='temp2', channels_out=1024, bn_decay=bn_decay)

        # classification for temporal
        temp1_pred = classification_head(temporal_1, is_training, 'cls_temp1', n_classes, bn_decay, weight_decay)
#         temp2_pred = classification_head(temporal_2, is_training, 'cls_temp2', n_classes, bn_decay, weight_decay)

    # classification from potinent
    pt_pred = classification_head(ptnet_out, is_training, 'cls_ptnet', n_classes, bn_decay, weight_decay)

    
    # concat output from poitnet and temporal networks
    if temporal:
        net = tf.concat([ptnet_out, temporal_1], axis=-1)
#         net = tf.math.add_n([ptnet_out, temporal_1, temporal_2])
        print(net)
    else:
        net = ptnet_out
    end_points['feature_vector'] = net

    # classification from entire network
    final_pred = classification_head(net, is_training, 'cls_net', n_classes, bn_decay, weight_decay)
    if temporal:
        return final_pred, [pt_pred, temp1_pred], end_points
    else:
        return final_pred, [pt_pred], end_points


def get_loss(pred, site_preds, label, end_points, aux_loss_weights):
    """ pred: B*NUM_CLASSES,
        label: B, """
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss) * aux_loss_weights[0]
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)

    classify_aux_losses = []
    for i, weight in zip(site_preds, aux_loss_weights[1:len(site_preds)+1]):
        print('Weighting loss of', i.name, 'by', weight)
        loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=i, labels=label)
        classify_loss2 = tf.reduce_mean(loss2) * weight
        classify_aux_losses.append(tf.reduce_mean(loss2))
        tf.summary.scalar('classify loss', classify_loss2)
        tf.add_to_collection('losses', classify_loss2)
    return classify_loss, classify_aux_losses


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
