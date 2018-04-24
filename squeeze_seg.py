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
import tensorflow as tf
import os
import random
from glob import glob
import cv2

class SqueezeSeg(object):
    """
    SqueezeSeg class.
    """
    def __init__(self, learning_rate=0.001, num_classes=3, batch_size=32, epochs=64, log_info=True):
        """
        Init paramters.
        Totally 3 classes.
        kitti_dict={'Car':1, 'Van':1, 'Truck':1, 'Pedestrian':2, 'Person_sitting':2, 'Cyclist':2,'Tram':1,
            'Misc':0,'DontCare':0}
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        if log_info:
            # to enable tf logging info
            tf.logging.set_verbosity(tf.logging.INFO)
    def crf_as_rnn_module(self, input_lidar_tensor, cnn_output_tensor, iter_output_tensor, iter_num, name):
        # TODO: (vincent.cheung.mcer@gmail.com) Not yet implemented.
        # Step 1: calculate Gaussian Kernel parameters from `input_lidar_tensor`, size:[3,5]

        # Step 2: message passing, filter the `cnn_output_tensor` with about Gaussian Kernel
        # using locally connected layer
        # Step 3: re-weight and compatibilty transformation, with 1x1 conv
        # Step 4: Update, add possibility and re-weight output
        # Step 5: Softmax normalization
        # Step 6: iterate 
        pass
    def fire_module(self, input_tensor, name, filters):
        """
        Fire module according to `SqueezeSeg`. Input tensor [H,W,C] and return [H,W,C] tensor.

        Args:
        `input_tensor`:input tensor, should be 4d tensor in NHWC format,[N,H,W,C]   
        `name`:the name of this module
        `filters`:the tuple of filters of two fiter size of convolution  

        Return:
        `output_tensor`:deconv tensor with the shape of [N,H,W,C] 
        """
        C = input_tensor.shape[-1]

        conv1 = tf.layers.conv2d(inputs=input_tensor, filters=filters[0], kernel_size=[1,1], strides=[1,1],
            activation=tf.nn.relu, padding='same',name=name+'_squeeze')

        conv2_1 = tf.layers.conv2d(inputs=conv1, filters=filters[1], kernel_size=[3,3], strides=[1,1],
            activation=tf.nn.relu, padding='same', name=name+'_expand_1_3x3')
           
        conv2_2 = tf.layers.conv2d(inputs=conv1, filters=filters[1], kernel_size=[1,1], strides=[1,1],
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

        conv1 = tf.layers.conv2d(inputs=input_tensor, filters=C/2, kernel_size=[1,1], strides=[1,1],
            activation=tf.nn.relu, padding='same', name=name+'_squeeze')

        deconv_x2 = tf.layers.conv2d_transpose(conv1,filters=C/4,kernel_size=[3,3], strides=[1,2],
            activation=tf.nn.relu, padding='same', name=name+'_deconv2')

        conv2_1 = tf.layers.conv2d(inputs=deconv_x2, filters=C/2, kernel_size=[3,3], strides=[1,1],
            activation=tf.nn.relu, padding='same', name=name+'_expand_1_3x3')
           
        conv2_2 = tf.layers.conv2d(inputs=deconv_x2, filters=C/2, kernel_size=[1,1], strides=[1,1],
            activation=tf.nn.relu, padding='same', name=name+'_expand_2_1x1')   
        #concate conv2_1 and conv2_2
        output_tensor = tf.concat([conv2_1,conv2_2], -1)
        return output_tensor
    
    def local_conv_layer(self, input_tensor, kernel_tensor, name):
        """
        Local connected convolution layer, by using tf.extract_image_patches to get tensor patches.
        Elementwise multiplication of filters with the image patches, followed by a summation.  

        Args:
        `input_tensor`:input tensor, should be 4D tensor in NHWC format, [N,H,W,C]  
        `kernel_tensor`:kernel tensor, should be 4D tensor in [N_Kernel,H,W,D], N_Kernel is the nums
        of kernel, and D is the depth   
        `name`:the name of this module  
        Return:
        `output_tensor`:local connected layer with elementwise multiplication of filters, in format 
        [N,H,W,C]
        """
        patches = tf.extract_image_patches(input_tensor, [1, kernel_tensor.shape[1], kernel_tensor.shape[2], 1], 
        [1,1,1,1], [1,1,1,1], padding='SAME')
        for patch in patches:
            pass
        pass

    def get_gaussian_filter(self, image, ksize, name):
        """
        Return the gaussian filter list according to CRF Gaussian filter formula.abs
        
        Args:
        `image`:the p(theta, phi) distribution of project depth map image

        Return:
        `output_filter`:return the Gaussian filter
        """
        # TODO: (vincent.cheung.mcer@gmail.com) Not yet implemented
        if self.filters is None:
            # in this case, it's [64,512,2]. 2 represents [theta, phi]
            image_index = np.zeros([image.shape[0], image.shape[1], 2])
            theta = np.arange(0, 64).reshape(64,1)
            phi = np.arange(0, 512)
            # setting image_index to [[[0,0],[0,1]...,[0,511]]...,[[63,0],[63,1]...,[63,511]]]
            image_index[:, :, 0] = np.repeat(theta, 512, axis=1)
            image_index[:, :, 1] = phi
            # get patches
            with tf.Session() as sess:
                idx_pl = tf.placeholder(tf.float32,shape=[None,image.shape[0],image.shape[1],2])
                get_patch_op = tf.extract_image_patches(idx_pl, [1,ksize[0],ksize[1],1], [1,1,1,1], 
                    [1,1,1,1], padding='SAME')
                out_patch = sess.run(get_patch_op, feed_dict={idx_pl:[image_index]})            
            # as the same shape of image_index
            out_patch = out_patch.reshape(-1,ksize[0]*ksize[1],2)
            # to [[0,0],[0,1]...,[63,511]]
            v_image_index = image_index.reshape(-1,2)
            # repeat 15 times for subtraction like [[[0,0],[0,0]...],[[0,1]...]...]
            vv_image_index = np.repeat(v_image_index, ksize[0]*ksize[1], axis=0).reshape(-1,15,2)
            # ||pi - pj||^2
            p_diff = (vv_image_index - out_path)**2
            pass
        else:
            return self.filters
        

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
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=[3,7], strides=[1,2],
            activation=tf.nn.relu, padding='same', name='conv1')
        
        # SqueezeNet used max-pooling to down-sample intermediate feature maps
        # in both width and height dimensions, but since our input
        # tensor's height is much smaller than its width, we only downsample
        # the width. So most of the strides are [1, 2]
        maxpool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1,2], strides=[1,2],
            padding='same', name='maxpool1')               
        fire2 = self.fire_module(maxpool1, 'fire2',[16,64])
        fire3 = self.fire_module(fire2, 'fire3',[16,64])
        maxpool3 = tf.layers.max_pooling2d(inputs=fire3, pool_size=[1,2], strides=[1,2],
            padding='same', name='maxpool3')
        fire4 = self.fire_module(maxpool3, 'fire4',[32,128])
        fire5 = self.fire_module(fire4, 'fire5',[32,128])
        maxpool5 = tf.layers.max_pooling2d(inputs=fire5, pool_size=[1,2], strides=[1,2],
            padding='same', name='maxpool5')
        fire6 = self.fire_module(maxpool5, 'fire6',[48,192])
        fire7 = self.fire_module(fire6, 'fire7',[48,192])
        fire8 = self.fire_module(fire7, 'fire8',[64,256])
        fire9 = self.fire_module(fire8, 'fire9',[64,256])  
        # Up sameple module
        fire_deconv10 = self.fire_deconv_module(fire9, 'fire_deconv10')
        skip_fire4_deconv10 = tf.concat([fire4, fire_deconv10],-1)
        
        fire_deconv11 = self.fire_deconv_module(skip_fire4_deconv10, 'fire_deconv11')
        skip_fire2_deconv11 = tf.concat([fire2, fire_deconv11],-1)
        
        fire_deconv12 = self.fire_deconv_module(skip_fire2_deconv11, 'fire_deconv12')
        skip_conv1a_deconv12 = tf.concat([conv1, fire_deconv12],-1)

        fire_deconv13 = self.fire_deconv_module(skip_conv1a_deconv12, 'fire_deconv13')
        # TODO: (vincent.cheung.mcer@gmail.com) Not sure about the skip layer of `conv1b` and `deconv13`
        skip_conv1b_deconv13 = tf.concat([input_layer,fire_deconv13],-1)

        # 1x1 Conv, similar to fully connected layer; activation is `softmax`, according to the paper
        # in this case withou rnn module ,modify activation to `relu` instead of `softmax`
        conv14 = tf.layers.conv2d(inputs=skip_conv1b_deconv13, filters=self.num_classes, kernel_size=[1,1], strides=[1,1],
            activation=tf.nn.relu, padding='same', name='conv14')
        
        logits = conv14
        #TODO:(vincent.cheung.mcer@gmail.com) Not yet finish the RNN refined module
        #### CRF_AS_RNN Module should be add here !
        # self.crf_as_rnn_module()
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            'classes': tf.argmax(input=logits, axis=3),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
        #print predictions['classes'].shape
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.num_classes)

        # In this case, use the weighted class label to incorporate the class imbalance situation
        # The following is the weighted entropy
        # TODO:(vincent.cheung.mcer@gmail.com) this weight is a hyper paramter now
        weights = tf.constant([0.5,0.3,0.2])
        class_weights = tf.multiply(onehot_labels, weights)
        xent = tf.multiply(class_weights, tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits))
        #loss = tf.losses.softmax_cross_entropy(onehot_labels=class_weights, logits=logits)
        loss = tf.reduce_mean(xent)
        # re-balance kitti label in 3 types :dontcare, vehicel, pedestrian
        # In this case, only care about label 0,1,2
        # kitti_dict={'Car':1, 'Van':1, 'Truck':1, 'Pedestrian':2, 'Person_sitting':2, 'Cyclist':2,'Tram':1,
        # 'Misc':0,'DontCare':0}


        tf.summary.scalar("loss_", loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            'mean_iou': tf.metrics.mean_iou(
                labels=labels, predictions=predictions['classes'], 
                    num_classes=self.num_classes, name='iou_metric')}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        
def get_all_data(data_folder, mode='train', type='dense', show_process=False):
    """
    Get all depth map and corresponding labels from `data_folder`.
    Args:
    `data_folder`:path to the folder that contains `velo_depth_map_train` and `velo_depth_map_eval`.
    `mode`:folder that contains all the `npy` datasets 
    `type`:type of npy for future use in sparse tensor, values={`dense`,`sparse`} 
    Ret:
    `grids`:list of voxel grids
    `labels`:list of labels
    """
    sub_path = 'velo_depth_map_'+mode
    sub_path_label = sub_path + '_gt'
    
    data_paths = glob(os.path.join(data_folder, sub_path, '*.npy'))
    if show_process == True:
        all_cnt = len(data_paths)
        cnt = 0
    # TODO:(vincent.cheung.mcer@gmail.com) not yet support sparse npy
    grids=[]
    labels=[]
    for depth_map_path in data_paths:
        if show_process == True:
            cnt += 1
            print 'processing:{}, {}/{}={}%%'.format(depth_map_path,cnt,all_cnt,cnt*1.0/all_cnt*100.0)

        # extract the label from path+file_name: e.g.`./voxel_npy/pillar.2.3582_12.npy`
        file_name = depth_map_path.split('/')[-1] #`pillar.2.3582_12.npy`
        depth_map_gt_path =  os.path.join(data_folder, sub_path_label, file_name)
        # load *.npy
        grid = np.load(depth_map_path).astype(np.float32)
        label = np.load(depth_map_gt_path).astype(np.int32).reshape(64,512)

        labels.append(label)
        grids.append(grid)
    return grids, labels


def main(argv):
    """
    The main function for SqueezeSeg training and evaluation.
    """
    # Use default setting
    if len(argv) != 4:
        print ('len(argv) is {}, use default setting'.format(len(argv)))
        data_folder = './'
        batch_size = 32
        epochs = 64
    else:
        data_folder = argv[1]
        batch_size = argv[2]
        epochs = argv[3]

    squeeze_seg = SqueezeSeg()
    # Voxnet Estimator: model init
    squeeze_seg_classifier = tf.estimator.Estimator(
        model_fn=squeeze_seg.squeeze_seg_fn, model_dir='./squeeze_seg/')

    # Trainning data collector
    grids_list, labels_list = get_all_data(data_folder,mode='train')
    train_data = np.array(grids_list)
    train_labels = np.array(labels_list)

    # Evaluating data collector
    eval_grids_list, eval_labels_list = get_all_data(data_folder,mode='eval')
    eval_data = np.array(eval_grids_list)
    eval_labels = np.array(eval_labels_list)

    print('data get')

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=True)

    print ('train start')

    squeeze_seg_classifier.train(
        input_fn=train_input_fn,
        steps=5000,
        hooks=[logging_hook])
    
    print('train done')

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = squeeze_seg_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


def eval(argv):
    # Use default setting
    if len(argv) != 4:
        print ('len(argv) is {}, use default setting'.format(len(argv)))
        data_folder = './'
        batch_size = 32
        epochs = 8
    else:
        data_folder = argv[1]
        batch_size = argv[2]
        epochs = argv[3]

    squeeze_seg = SqueezeSeg()
    # Voxnet Estimator: model init
    squeeze_seg_classifier = tf.estimator.Estimator(
        model_fn=squeeze_seg.squeeze_seg_fn, model_dir='./squeeze_seg/')

    # Evaluating data collector
    eval_grids_list, eval_labels_list = get_all_data(data_folder,mode='eval')
    eval_data = np.array(eval_grids_list)
    eval_labels = np.array(eval_labels_list)

    print('data get')

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        #y=eval_labels,#for evaluation
        num_epochs=1,
        shuffle=False)
    # predict the semantic label
    predictions = squeeze_seg_classifier.predict(input_fn=eval_input_fn)
    # simple example for evaluation
    #eval_results = squeeze_seg_classifier.evaluate(input_fn=eval_input_fn)
    #print(eval_results)
    idx=0
    # predictions is a generator
    for i in predictions:
        # get the predicted depth map
        mat = np.array(i['classes'])
        # set object label as some kind of intensity
        mat[mat!=0]=4.2108e+06
        mat[mat==0]=0

        # data is in shape [x,y,z,intensity], set intensity to predict label
        eval_data[idx][:,:,4] = mat
        x=eval_data[idx][:,:,1]
        y=eval_data[idx][:,:,2]
        z=eval_data[idx][:,:,3]
        l=mat
        # re-project depth map into point cloud
        p=np.stack((x,y,z,l)).reshape(4,-1)
        # index for saving files
        idx+=1
        # uncomment to save pcd file for display
        # pcd file header
        # header=('# .PCD v.7 - Point Cloud Data file format\n'
        #     'VERSION .7\n'
        #     'FIELDS x y z rgb intensity\n'
        #     'SIZE 4 4 4 4 4\n'
        #     'TYPE F F F F F\n'
        #     'COUNT 1 1 1 0 1\n'
        #     'WIDTH 32768\n'
        #     'HEIGHT 1\n'
        #     'VIEWPOINT 0 0 0 1 0 0 0\n'
        #     'POINTS 32768\n'
        #     'DATA ascii\n')
        # np.savetxt('./xyz/'+str(idx)+'.pcd',p.T,fmt='%f %f %f %f',header=header,comments='')
        
        # uncomment for save & display semantic segmentation results in image
        #mat[mat==2]=255
        #cv2.imwrite('./test/'+str(idx)+'.png',mat)
        #cv2.waitKey(0)
        #print mat!=0


if __name__ == '__main__':
    # run the main function and model_fn, according to Tensorflow R1.3 API
    # TODO: (vincent.cheung.mcer@gmail.com) Not yet implemented.
    #tf.app.run(main=main, argv=['./'])
    tf.app.run(main=eval, argv=['./'])
    #get_all_data(data_folder='./',show_process=True)
