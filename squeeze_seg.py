#!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
SqueezeSeg implementation on tensorflow.
SqueezeSeg regards LiDAR pointcloud classfication,
as LiDAR projected depth map (image) semantic segmentation.
This project provides classfier on semantic segmantation, 
and re-project the depth map into 3D pointcloud.
"""
import numpy as np
#import tensorflow as tf
import os
import random
from glob import glob


class SqueezeSeg(object):
    """
    SqueezeSeg class.
    """
    def __init__(self, learning_rate=0.001, num_classes=14, batch_size=32, epochs=64):
        """
        Init paramters
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        # to enable tf logging info
        tf.logging.set_verbosity(tf.logging.INFO)

    def fire_module(self, input_tensor, name):
        """
        Fire module according to `SqueezeSeg`. Input tensor [H,W,C] and return [H,W,C] tensor.

        Args:
        `input_tensor`:input tensor, should be 4d tensor in NHWC format,[N,H,W,C]   
        `name`:the name of this module  

        Return:
        `output_tensor`:deconv tensor with the shape of [N,H,W,C] 
        """
        C = input_tensor.shape[-1]

        conv1 = tf.layers.conv2d(inputs=input_tensor, filters=C/4, kernel_size=[1,1], strides=[1,1],
            activation=tf.nn.relu, padding='same',name=name+'_squeeze')

        conv2_1 = tf.layers.conv2d(inputs=conv1, filters=C/2, kernel_size=[3,3], strides=[1,1],
            activation=tf.nn.relu, padding='same', name=name+'_expand_1_3x3')
           
        conv2_2 = tf.layers.conv2d(inputs=conv1, filters=C/2, kernel_size=[1,1], strides=[1,1],
            activation=tf.nn.relu, padding='same', name=name+'_expand_2_1x1')   
        #concate conv2_1 and conv2_2
        output_tensor = tf.concat([conv2_1,conv2_2], -1)
        return output_tensor

    def fire_deconv_module(self, input_tensor, name):
        """
        Fire deconv module according to `SqueezeSeg`. Input tensor [H,W,C] and return [H,2*W,C] tensor.

        Args:
        `input_tensor`:input tensor, should be 4d tensor in NHWC format,[N,H,W,C]   
        `name`:the name of this module  

        Return:
        `output_tensor`:deconv tensor with the shape of [N,H,2*W,C] 
        """
        C = input_tensor.shape[-1]

        conv1 = tf.layers.conv2d(inputs=input_tensor, filters=C/4, kernel_size=[1,1], strides=[1,1],
            activation=tf.nn.relu, padding='same', name=name+'_squeeze')

        deconv_x2 = tf.layers.conv2d_transpose(conv1,filters=24,kernel_size=[3,3], strides=[1,2],
            activation=tf.nn.relu, padding='same', name=name+'_deconv2')

        conv2_1 = tf.layers.conv2d(inputs=conv1, filters=C/2, kernel_size=[3,3], strides=[1,1],
            activation=tf.nn.relu, padding='same', name=name+'_expand_1_3x3')
           
        conv2_2 = tf.layers.conv2d(inputs=conv1, filters=C/2, kernel_size=[1,1], strides=[1,1],
            activation=tf.nn.relu, padding='same', name=name+'_expand_2_1x1')   
        #concate conv2_1 and conv2_2
        output_tensor = tf.concat([conv2_1,conv2_2], -1)
        return output_tensor

    def squeeze_seg_fn(self, features, labels, mode):
        """
        Squeeze_seg tensorflow graph.
        It follows description from this TensorFlow tutorial:
        `https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts`
        
        Args:
        `features`:default paramter for tf.model_fn
        `labels`:default paramter for tf.model_fn
        `mode`:default paramter for tf.model_fn
        Ret:
        `EstimatorSpec`:    predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
        """
        input_layer = tf.reshape(features['x'], [-1, 64, 512, 5])
        
        # SqueezeNet : `conv1` to `fire9`
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=[7,7], strides=[2,2],
            activation=tf.nn.relu, padding='same', name='conv1')
        
        # SqueezeNet used max-pooling to down-sample intermediate feature maps
        # in both width and height dimensions, but since our input
        # tensor's height is much smaller than its width, we only downsample
        # the width. So most of the strides are [1, 2]
        maxpool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3,3], strides=[1,2],
            name='maxpool1')               
        fire2 = self.fire_module(maxpool1, 'fire2')
        fire3 = self.fire_module(fire2, 'fire3')
        maxpool3 = tf.layers.max_pooling2d(inputs=fire3, pool_size=[3,3], strides=[1,2],
            name='maxpool3')
        fire4 = self.fire_module(maxpool3, 'fire4')
        fire5 = self.fire_module(fire4, 'fire5')
        maxpool5 = tf.layers.max_pooling2d(inputs=fire5, pool_size=[3,3], strides=[1,2],
            name='maxpool5')
        fire6 = self.fire_module(maxpool5, 'fire6')
        fire7 = self.fire_module(fire6, 'fire7')
        fire8 = self.fire_module(fire7, 'fire8')
        fire9 = self.fire_module(fire8, 'fire9')  
        # Up sameple module
        fire_deconv10 = self.fire_deconv_module(fire9, 'fire_deconv10')
        skip_fire4_deconv10 = tf.add(fire4, fire_deconv10)
        
        fire_deconv11 = self.fire_deconv_module(skip_fire4_deconv10, 'fire_deconv11')
        skip_fire2_deconv11 = tf.add(fire2, fire_deconv11)
        
        fire_deconv12 = self.fire_deconv_module(skip_fire2_deconv11, 'fire_deconv12')
        skip_conv1a_deconv12 = tf.add(conv1, fire_deconv12)

        fire_deconv13 = self.fire_deconv_module(skip_conv1_deconv12, 'fire_deconv13')
        # TODO: (vincent.cheung.mcer@gmail.com) Not sure about the skip layer of `conv1b` and `deconv13`
        skip_conv1b_deconv13 = tf.concat([input_layer,fire_deconv13],-1)

        # 1x1 Conv, similar to fully connected layer; activation is `softmax`, according to the paper
        conv14 = tf.layers.conv2d(inputs=skip_conv1b_deconv13, filters=1, kernel_size=[1,1], strides=[1,1],
            activation=tf.nn.softmax, padding='same', name='conv14')
        #TODO:(vincent.cheung.mcer@gmail.com) Not yet finish the RNN refined module
        pass

if __name__ == '__main__':
    # run the main function and model_fn, according to Tensorflow R1.3 API
    # TODO: (vincent.cheung.mcer@gmail.com) Not yet implemented.
    pass