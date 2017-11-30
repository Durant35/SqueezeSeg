"""Label point clouds using image 2d annotation and calibration files."""

from thirdparty.calib import Calib
import argparse
import os
import cv2
import numpy as np
import sys
#import pcl

#kitti label dictionary, other class are valued by `0`
kitti_dict={'Car':1, 'Van':2, 'Truck':3, 'Pedestrian':4, 'Person_sitting':5, 'Cyclist':6,'Tram':7,
'Misc':8,'DontCare':9}

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

def pto_depth_map(velo_points, inten, H=64, W=512, C=5, dtheta=np.radians(0.4), dphi=np.radians(90/512.0)):
    """
    Project velodyne points into front view depth map.
    
    Args:
    `velo_points`:velodyne points in shape [:,3]    
    `inten`:ponits intensity    
    `H`:the row num of depth map, could be 64(default), 32, 16      
    `W`:the col num of depth map    
    `C`:the channel size of depth map   
    `dtheta`:the delta theta of H, in radian    
    `dphi`:the delta phi of W, in radian   

    Return:
    `depth_map`:the projected depth map of shape[H,W,C]
    """
    x, y, z = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2]
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)#sqrt(x^2+y^2+z^2)
    theta = np.arcsin(y/d)#z/d
    r = np.sqrt(x ** 2 + z ** 2)#sqrt(x^2+y^2)
    phi = np.arcsin(x/r) #y/r

    theta_ = np.abs((theta/dtheta).astype(int)-5)
    theta_[theta_>=64]=63
    # np.savetxt('./dump/'+'theta_'+"dump.txt",(theta/dtheta).astype(int), fmt="%d")
    # np.savetxt('./dump/'+'theta'+"dump.txt",theta_, fmt="%d")
    
    phi_ = np.abs((phi/dphi).astype(int)+255)
    phi_[phi_>=512]=511
    # print(np.min((phi/dphi)),np.max((phi/dphi)))
    # np.savetxt('./dump/'+'phi_'+"dump.txt",(phi/dphi).astype(int), fmt="%d")

    #print(inten.shape)
    
    depth_map = np.zeros((H,W,C))
    # 5 channels according to paper
    if C==5:
        depth_map[theta_,phi_,0] = inten
        depth_map[theta_,phi_,1] = x
        depth_map[theta_,phi_,2] = y
        depth_map[theta_,phi_,3] = z
        depth_map[theta_,phi_,4] = d
    else:
        depth_map[theta_,phi_,0] = inten

    return depth_map

class box3d(object):
    """
    An object represent 3D bounding box in KITTI dataset.
    Simple usage:
    box = box3d()
    box.set_list(str_list)#str_list can ref to `set_list`
    box.get_box()
    box.get_label()
    """
    def __init__(self,p1=0,p2=0,p4=0,p5=0):
        """
        Init 3D bounding box with 4 points
        """
        self.init = True
        self.u = p1-p2
        self.v = p1-p4
        self.w = p1-p5
        self.b_u = [np.dot(self.u,p1),np.dot(self.u,p2)]
        self.b_v = [np.dot(self.v,p1),np.dot(self.v,p4)]
        self.b_w = [np.dot(self.w,p1),np.dot(self.w,p5)]
        #print(self.v,self.v,self.w,self.b_u,self.b_v,self.b_w)
    
    def set_list(self,str_list):
        """
        Init 3D bounding box with KITTI object label.txt.
        Such as ['Truck', '0.00', '0', '-1.57', '599.41', '156.40', '629.75', 
        '189.25', '2.85', '2.63', '12.34', '0.47', '1.49', '69.44', '-1.56']
        The `Values Name Description` can be ref:https://github.com/NVIDIA/DIGITS/issues/992
        """
        self.label = str_list[0]
        h,w,l = np.array(str_list[8:11]).astype(float)
        x,y,z = np.array(str_list[11:14]).astype(float)
        rot = np.array(str_list[14]).astype(float)

        px = np.array([0.5*l,0.5*l,-0.5*l,-0.5*l, 0.5*l,0.5*l,-0.5*l,-0.5*l])
        py = np.array([0,0,0,0, -h,-h,-h,-h])
        pz = np.array([0.5*w,-0.5*w,-0.5*w,0.5*w, 0.5*w, -0.5*w, -0.5*w,0.5*w])
        rot_mat = np.array([
            [np.cos(rot), 0, np.sin(rot)],
            [0, 1, 0],
            [-np.sin(rot), 0, np.cos(rot)],
            
        ])
        
        p_stack = np.array([px,py,pz])
        #np.savetxt('./dump/'+'box'+str(rot)+"dump.xyz",p_stack, fmt="%f %f %f")
        # #print (p_stack)
        rot_p = np.dot(rot_mat,p_stack)
        rot_p[0,:] = rot_p[0,:]+x
        rot_p[1,:] = rot_p[1,:]+y
        rot_p[2,:] = rot_p[2,:]+z
        
        # header1=('# .PCD v.7 - Point Cloud Data file format\n'
        # 'VERSION .7\n'
        # 'FIELDS x y z\n'
        # 'SIZE 4 4 4\n'
        # 'TYPE F F F\n'
        # 'COUNT 1 1 1\n'
        # 'WIDTH '+str(8)+'\n'
        # 'HEIGHT 1\n'
        # 'VIEWPOINT 0 0 0 1 0 0 0\n'
        # 'POINTS '+str(8)+'\n'
        # 'DATA bin\n')
        # np.savetxt('./'+'box'+str(x)+'.pcd',rot_p.T,fmt='%f %f %f',header=header1,comments='')

        p1 = rot_p[:,0]
        p2 = rot_p[:,1]
        p4 = rot_p[:,3]
        p5 = rot_p[:,4]
        self.__init__(p1,p2,p4,p5)
        # TODO: (vincent.cheung.mcer@gmail.com) Adding support to rotate
        #self.__init__(rot_p[0],rot_p[1],rot_p[2],rot_p[3])

    #TODO:(vincent.cheung.mcer@gmail.com) need to add throw-catch for value init detection
    def get_box(self):
        """
        Return u,v,w,bounding_u,bouding_v,bouding_w of predencular 3D bounding box.

        Return:
        `u`:P1-P2
        `v`:P1-P4
        `w`:P1-P5
        `b_u`:a list of dot product i.e.:[u.p1, u.p2]
        `b_v`:a list of dot product i.e.:[u.p1, u.p4]
        `b_w`:a list of dot product i.e.:[u.p1, u.p5]
        """
        if self.init is None:
            print('Error using get_box without init.')
            return None
        return self.u,self.v,self.w,self.b_u,self.b_v,self.b_w
    
    def get_label(self):
        """
        Return the label of the current box.abs

        Return:
        `kitti_dict[self.label]`:a dictionary value of `label` in `kitti_dict`
        """
        if self.label is None:
            print('Error using get_label without init.')
            return kitti_dict['DontCare']

        return kitti_dict[self.label]

    def with_box(points):
        """
        Return points index that within the 3d bounding box.

        `points`:3d velodyne points

        Return:
        `points_ind`:
        """
        # dot product of vector
        u_dot = np.dot(self.u,points.T)
        v_dot = np.dot(self.v,points.T)
        w_dot = np.dot(self.w,points.T)
        # index condition
        con_b_u = np.logical_and(u_dot<self.b_u[0], u_dot>self.b_u[1])
        con_b_v = np.logical_and(v_dot<self.b_v[0], v_dot>self.b_v[1])
        con_b_w = np.logical_and(w_dot<self.b_w[0], w_dot>self.b_w[1])
        # return points within three bouding
        return np.logical_and(np.logical_and(con_b_u,con_b_v),con_b_w)

def within_3d_box(points, box3d):
    """
    Return points index that within the 3d bounding box.

    `points`:3d velodyne points
    `box3d`:box3d object

    Return:
    `points_ind`:
    """
    u,v,w,b_u,b_v,b_w = box3d.get_box()
    u_dot = np.dot(u,points.T)
    v_dot = np.dot(v,points.T)
    w_dot = np.dot(w,points.T)
    con_b_u = np.logical_and(u_dot<b_u[0], u_dot>b_u[1])
    con_b_v = np.logical_and(v_dot<b_v[0], v_dot>b_v[1])
    con_b_w = np.logical_and(w_dot<b_w[0], w_dot>b_w[1])
    return np.logical_and(np.logical_and(con_b_u,con_b_v),con_b_w)

def main():
    parser = argparse.ArgumentParser(description='Converter')
    parser.add_argument('--data-object-velodyne', type=str, help='Path to KITTI `data_object_velodyne`', default='data_object_velodyne/')
    parser.add_argument('--cam-idx', type=int, help='Index of the camera being used', default=2)
    parser.add_argument('--dataset', type=str, choices=('training', 'testing'), help='Which dataset to run on', default='training')
    args = parser.parse_args()

    #velodyne dir
    dirpath = os.path.join(args.data_object_velodyne, args.dataset, 'velodyne')
    
    # process counter
    cnt = 1
    total_cnt = len(os.listdir(dirpath))

    for filename in os.listdir(dirpath):
        # ignore files
        if filename.startswith('.'):
            continue
        # filename is :00xxxx.bin/txt, x is number
        print (filename)
        name = filename.split('.')[0]
        id = name
        
        # Structure look like this:
        #   data_object_velodyne/velodyne/xxx.bin
        #   data_object_velodyne/calib/xxx.txt
        #   data_object_velodyne/label_2/xxx.txt
        velo_path = os.path.join(args.data_object_velodyne, args.dataset, 'velodyne', filename)
        calib_path = os.path.join(args.data_object_velodyne, args.dataset, 'calib', '%s.txt' % name)
        gt_path = os.path.join(args.data_object_velodyne, args.dataset, 'label_2', '%s.txt' % name)

        # n x 4 (x, y, z, intensity), read in velodyne binary data
        velo_data = np.fromfile(velo_path, dtype=np.float32).reshape((-1, 4))
        velo_points = velo_data[:, :3]# (x,y,z)
        inten = velo_data[:,3]# intensity
        
        boxes3d=[]        
        # readin all bounding box from grouth truth txt
        with open(gt_path) as f:
            for line in f.readlines():
                #print (line)
                str_list = line.strip('\n').split(' ')
                if str_list[0]=='Misc' or str_list[0]=='DontCare':
                    continue
                box = box3d()
                box.set_list(str_list)
                boxes3d.append(box)


        # if no interest bounding box in this frame, then skip
        if len(boxes3d) == 0:
            print('\r%s does not have a ground truth file' % filename)
            continue

        # Load calibration
        calib = Calib(calib_path)

        # Get points with FOV
        x, y, z = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2]
        con=hv_in_range(x,y,z,[-45,45])
        # Calibrate velodyne points to camera coordinates
        img_points = calib.velo2cams(velo_points[con])
        img_intens = inten[con]
        # Save points within FOV to xyz file
        #np.savetxt('./dump/'+'velo2cams'+name+id+"dump.xyz",img_points.T, fmt="%f %f %f")


        

        # Get points with 3D bounding box
        gt_points = np.zeros_like(img_points.T[:,0])
        # label all points within bouding box to there label
        for idx in range(len(boxes3d)):
            # get index within 3d box
            v_con = within_3d_box(img_points.T,boxes3d[idx])
            # velodyne points within 3d box
            v_in_b=img_points.T[v_con]
            # label those points with
            gt_points[v_con] = boxes3d[idx].get_label()
            #np.savetxt('./dump/'+'obj_v_in_b'+name+str(idx)+"o.xyz",v_in_b, fmt="%f %f %f")
            #np.savetxt('./dump/'+'con_obj_v_in_b'+name+str(idx)+"o.xyz",velo_points[con], fmt="%f %f %f")

        # Get five channels depth map
        train_depth_map = pto_depth_map(img_points.T, inten=img_intens)
        # Get one channel ground truth, with points within bouding box labeled by kitti_dict[object.label]
        gt_depth_map = pto_depth_map(img_points.T, C=1, inten=gt_points)

        # Uncomment to display depth map
        # cv2.imshow('intensity', train_depth_map[:,:,0])
        # cv2.imshow('x', train_depth_map[:,:,1])
        # cv2.imshow('y', train_depth_map[:,:,2])
        # cv2.imshow('z', train_depth_map[:,:,3])
        # cv2.imshow('range', train_depth_map[:,:,4])
        # cv2.imshow('ground truth', gt_depth_map[:,:,0])
        # cv2.waitKey(1)

        # Write npy files to velo_depth_map folder
        os.makedirs('./velo_depth_map_train', exist_ok=True)
        np.save('./velo_depth_map_train/'+name, train_depth_map)
        os.makedirs('./velo_depth_map_gt', exist_ok=True)
        np.save('./velo_depth_map_gt/'+name, gt_depth_map)

        # process counter and process displaying
        print('saving %s to train and gt'%name+'.npy')
        print ('process %d/%d is %f %%'%(cnt,total_cnt,cnt/total_cnt*100.0))
        cnt += 1

    print('Done, sir.')


if __name__ == '__main__':
    main()