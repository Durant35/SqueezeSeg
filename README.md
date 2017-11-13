# SqueezeSeg_tf
An on going TF implementation on SqueezeSeg to deal with LiDAR pointcloud.

## Description
`bin2depth.py`: is used to generate depth map and the corresponding ground truth from kitti-velodyne-binary file, by calibration `velo2cam`.    
`squeeze_seg.py`: is the main part of the network(still under construction, the current part only finish the CNN part but the CRF-AS-RNN refinement).   
`pcm2pcm.py`: is a ROS node which listens pointcloud2 message from ROS, and republish pointcloud2 message with `squeeze_seg` predicted labels. Both of the pc2 message topic to be listenned and the pc2 message topic to be published can be defined in the cmd.   

## Videos
[![screencast](https://vthumb.ykimg.com/054304085A0848A000000120A90164B8.png)](http://v.youku.com/v_show/id_XMzE1MzI2MTc1Ng==.html?spm=a2h3j.8428770.3416059.1)
