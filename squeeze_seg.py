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
        pass

if __name__ == '__main__':
    # run the main function and model_fn, according to Tensorflow R1.3 API
    pass