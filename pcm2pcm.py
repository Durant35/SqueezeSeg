#!/usr/bin/python2
# -*- coding: utf-8 -*-
"""
A ROS node to perform semantic segmantation on LiDAR pointcloud.
1. Listen to ROS Pointcloud message topic:/
2. Perform FOV filtering according to KITTI setting.
3. Project the FOV filtered pointcloud into 5 channels depth map, i.e. intensity,x,y,z,range
4. Predict the semantic label of depth map.
5. Reproject the depth map into Pointcloud, re-package to pointcloud message topic:/
"""
import numpy as np
import tensorflow as tf
import os
import argparse
import random
from glob import glob
import cv2
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

from squeeze_seg import SqueezeSeg


def hv_in_range(x, y, z, fov, fov_type='h'):
    """ 
    Extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit 
    
    Args:
    `x`:velodyne points x array
    `y`:velodyne points y array
    `z`:velodyne points z array
    `fov`:a two element list, e.g.[-45,45]  
    `fov_type`:the fov type, could be `h` or 'v',defualt in `h`    

    Return:
    `cond`:condition of points within fov or not    

    Raise:
    `NameError`:"fov type must be set between 'h' and 'v' " 
    """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if fov_type == 'h':
        return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi / 180), \
                                np.arctan2(y, x) < (-fov[0] * np.pi / 180))
    elif fov_type == 'v':
        return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180), \
                                np.arctan2(z, d) > (fov[0] * np.pi / 180))
    else:
        raise NameError("fov type must be set between 'h' and 'v' ")

def pto_depth_map(velo_points, H=64, W=512, C=5, dtheta=np.radians(0.4), dphi=np.radians(90/512.0)):
    """
    Project velodyne points into front view depth map.
    
    Args:
    `velo_points`:velodyne points in shape [:,4]    
    `H`:the row num of depth map, could be 64(default), 32, 16      
    `W`:the col num of depth map    
    `C`:the channel size of depth map   
    `dtheta`:the delta theta of H, in radian    
    `dphi`:the delta phi of W, in radian   

    Return:
    `depth_map`:the projected depth map of shape[H,W,C]
    """
    x, y, z = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2]
    d = np.sqrt(x ** 2 + y ** 2 + z**2)
    r = np.sqrt(x ** 2 + z ** 2)#sqrt(x^2+y^2)
    d[d==0]=0.000001
    r[r==0]=0.000001
    phi = np.arcsin(x/r) #y/r
    # with the help of ROS velodyne driver, taking ring as the theta
    if velo_points.shape[-1]>4:
        ring = velo_points[:,4]+16
        theta_ = ring.astype(int)
    else:
        theta = np.arcsin(y/d)
        theta_ = np.abs((theta/dtheta).astype(int)-5)
        theta_[theta_>=64]=63

    phi_ = np.abs((phi/dphi).astype(int)+255)
    phi_[phi_>=512]=511
    #print theta,phi,theta_.shape,phi_.shape
    # print(np.min((phi/dphi)),np.max((phi/dphi)))
    #np.savetxt('./dump/'+'phi'+"dump.txt",(phi_).astype(np.float32), fmt="%f")
    #np.savetxt('./dump/'+'phi_'+"dump.txt",(phi/dphi).astype(np.float32), fmt="%f")

    
    depth_map = np.zeros((H,W,C))
    # 5 channels according to paper
    if C==5:
        depth_map[theta_,phi_,0] = velo_points[:, 3]#intensity
        depth_map[theta_,phi_,1] = x
        depth_map[theta_,phi_,2] = y
        depth_map[theta_,phi_,3] = z
        depth_map[theta_,phi_,4] = d
    else:
        depth_map[theta_,phi_,0] = velo_points[:, 3]
    return depth_map 

class squeezeseg_node(object):
    """
    Squeeze-Seg ros node. 
    """
    def __init__(self, lis_topic='/velodyne_points', pub_topic='/class_topic', field_num=4):
        """
        Init function.

        Args:
        `lis_topic`:the pointcloud message topic to be listenned, default '/velodyne_points'
        `pub_topic`:the pointcloud message topic to be published, default '/class_topic'

        Raise:

        """
        self.lis_topic = lis_topic
        self.pub_topic = pub_topic
        self.point_field = self.make_point_field(field_num)
        # publisher
        self.pcmsg_pub = rospy.Publisher(self.pub_topic, PointCloud2,queue_size=1)
        # ros node init
        rospy.init_node('tasqueezeseg_node', anonymous=True)
        # listener
        rospy.Subscriber(self.lis_topic, PointCloud2, self.pcmsg_cb)
        self.squeeze_seg = SqueezeSeg(log_info=False)
        # squeeze_seg Estimator: model init
        self.squeeze_seg_classifier = tf.estimator.Estimator(
            model_fn=self.squeeze_seg.squeeze_seg_fn, model_dir='./squeeze_seg/')

        print 'now will listen : {}, and publish : {}'.format(self.lis_topic, self.pub_topic)
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    def make_point_field(self, num_field):
        # get from data.fields
        msg_pf1 = pc2.PointField()
        msg_pf1.name = np.str('x')
        msg_pf1.offset = np.uint32(0)
        msg_pf1.datatype = np.uint8(7)
        msg_pf1.count = np.uint32(1)

        msg_pf2 = pc2.PointField()
        msg_pf2.name = np.str('y')
        msg_pf2.offset = np.uint32(4)
        msg_pf2.datatype = np.uint8(7)
        msg_pf2.count = np.uint32(1)

        msg_pf3 = pc2.PointField()
        msg_pf3.name = np.str('z')
        msg_pf3.offset = np.uint32(8)
        msg_pf3.datatype = np.uint8(7)
        msg_pf3.count = np.uint32(1)

        msg_pf4 = pc2.PointField()
        msg_pf4.name = np.str('intensity')
        msg_pf4.offset = np.uint32(16)
        msg_pf4.datatype = np.uint8(7)
        msg_pf4.count = np.uint32(1)

        if num_field == 4:
            return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]
        
        msg_pf5 = pc2.PointField()
        msg_pf5.name = np.str('ring')
        msg_pf5.offset = np.uint32(20)
        msg_pf5.datatype = np.uint8(4)
        msg_pf5.count = np.uint32(1)

        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]



    def pcmsg_cb(self,data):
        # read points from pointcloud message `data`
        pc = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z","intensity","ring"))
        # print data.fields
        # to conver pc into numpy.ndarray format
        np_p = np.array(list(pc))
        # print np_p
        # perform fov filter by using hv_in_range
        cond = hv_in_range(x=np_p[:,0],y=np_p[:,1],z=np_p[:,2],fov=[-45,45])
        # to rotate points according to calibrated points with velo2cam
        np_p_ranged = np.stack((np_p[:,1],-np_p[:,2],np_p[:,0],np_p[:,3])).T
        np_p_ranged = np_p_ranged[cond]
 
        # get depth map
        dep_map = pto_depth_map(velo_points=np_p_ranged,C=5).astype(np.float32)
        #normalize intensity from [0,255] to [0,1], as shown in KITTI dataset
        #dep_map[:,:,0] = (dep_map[:,:,0]-0)/np.max(dep_map[:,:,0])
        #dep_map = cv2.resize(src=dep_map,dsize=(512,64))

        # to perform prediction
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": dep_map},
            #batch_size=1,
            num_epochs=1,
            shuffle=False)
        predictions = self.squeeze_seg_classifier.predict(input_fn=predict_input_fn)
        #cv2.imshow('test',dep_map[:,:,0])
        #cv2.waitKey(1)

        #predict label
        l=(np.array(list(predictions))[0]['classes'])
        # all non_zeros points are predicted object
        num = np.count_nonzero(l)
        # set all labelled object point into 255, as intensity, for display
        l[l!=0]=255
        l=l.reshape(-1)
        #print np.count_nonzero(l)
        # condition filter for filter out those object points
        cond=(l==255)
        ## to reproject points from depth map into 3D pointcloud
        x=dep_map[:,:,1].reshape(-1)
        y=dep_map[:,:,2].reshape(-1)
        z=dep_map[:,:,3].reshape(-1)
        i=dep_map[:,:,0].reshape(-1)
        # print x.shape,y.shape,z.shape,l.shape
        # r=dep_map[:,:,0].astype(np.uint16)
        

        # Create 4 PointFields as channel description
        # stack points
        ppp = np.stack((-x[cond],y[cond],z[cond],l[cond]))

        # create pointcloud2 message 
        new_header = data.header
        new_header.frame_id = 'class_velodyne'
        new_header.stamp = rospy.Time()
        # point_field = [x,y,z,intensity,ring(optional)]
        pc_ranged = pc2.create_cloud(header=new_header,fields=self.point_field,points=ppp.T)

        # saving PCD file
        # pcd =ppp.reshape(4,-1)
        # print pcd.shape,'hahah'
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
        # print header
        # np.savetxt('./xyz_raw_2/'+str(num)+'_'+str(data.header.stamp)+'.pcd',pcd.T,fmt='%f %f %f %f',header=header,comments='')
        
        
        self.pcmsg_pub.publish(pc_ranged)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pointcloud message semantic classfier')
    parser.add_argument('--lis_topic', type=str, help='the pointcloud message topic to be listenned, default `/velodyne_points`', default='/velodyne_points')
    parser.add_argument('--pub_topic', type=str, help='the pointcloud message topic to be listenned, default `/class_topic`', default='/class_topic')
    #parser.add_argument('--dataset', type=str, choices=('training', 'testing'), help='Which dataset to run on', default='training')
    args = parser.parse_args()
    node = squeezeseg_node(lis_topic=args.lis_topic,pub_topic=args.pub_topic)
    print 'finish'