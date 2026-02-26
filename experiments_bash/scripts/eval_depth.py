#!/usr/bin/python

# Modified by Raul Mur-Artal
# Automatically compute the optimal scale factor for monocular VO/SLAM.

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""

import sys
import argparse
import os
# import rerun as rr
import cv2
import numpy as np

def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    
    """
    np.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3, -1))
    data_zerocentered = data - data.mean(1).reshape((3, -1))
    
    W = np.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    rotmodel = rot*model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
        normi = np.linalg.norm(model_zerocentered[:,column])
        norms += normi*normi

    s = float(dots/norms)    

    # print("scale: %f " % s  )
    
    trans = data.mean(1).reshape((3,-1)) - s*rot * model.mean(1).reshape((3,-1))
    
    model_aligned = s*rot * model + trans
    alignment_error = model_aligned - data
    
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error),0)).A[0]
        
    return rot,trans,trans_error, s

def plot_traj(ax,stamps,traj,style,color,label):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = np.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x)>0:
            ax.plot(x,y,style,color=color,label=label)
            label=""
            x=[]
            y=[]
        last= stamps[i]
    if len(x)>0:
        ax.plot(x,y,style,color=color,label=label)
            

if __name__=="__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('gt_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('est_path', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('gt_path', help='est depth path')
    args = parser.parse_args()
    
    # load gt pose
    gt_xyz_whole = []
    with open(args.gt_file, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        gt_xyz_whole.append([c2w[0,3], c2w[1,3], c2w[2,3]])
    gt_xyz_whole = np.array(gt_xyz_whole)  # [n,3]

    # load estimated traj
    est_idxs = []
    est_xyz = []
    est_pose_path = os.path.join(args.est_path, "pose")
    est_pose_file_list = sorted(os.listdir(est_pose_path))
    for est_pose_file in est_pose_file_list:
        est_c2w = np.loadtxt(os.path.join(est_pose_path, est_pose_file))
        est_xyz.append([est_c2w[0,3], est_c2w[1,3], est_c2w[2,3]])
        est_idxs.append((int)(est_pose_file.split('.')[0]))
    est_xyz = np.array(est_xyz)
    
    # associate
    gt_xyz = gt_xyz_whole[est_idxs]

    gt_xyz = gt_xyz.T
    est_xyz = est_xyz.T
    
    rot,trans,trans_error,scale = align(est_xyz,gt_xyz)
    
    est_xyz_aligned = scale * rot * est_xyz + trans
    
    print(f"ATE RMSE: {trans_error.mean()*100}")
    
    # plot
    # rr.init("Evaluation result", spawn=True)
    
    # ## GT
    # rr.log(
    #     "Ground_Truth_Traj",
    #     rr.Points3D(gt_xyz_whole)
    # )
    
    # ## Est
    # rr.log(
    #     "Estimate_Traj",
    #     rr.Points3D(est_xyz_aligned.T)
    # )
    
    # eval_depth
    ## estimated depths
    for item in os.listdir(args.est_path):
        item_path = os.path.join(args.est_path, item)
        
        if os.path.isdir(item_path) and item.endswith('_shutdown'):
            gs_result_path = item_path
            
    est_depth_path = os.path.join(gs_result_path, "depth")
    
    est_depth_files = sorted(os.listdir(est_depth_path))
    
    ## gt depths
    gt_imgs_path = os.path.join(args.gt_path, "results")
    
    gt_depth_files = []
    for file in os.listdir(gt_imgs_path):
        if file.endswith('.png'):
            gt_depth_files.append(file)

    gt_depth_files = sorted(gt_depth_files)

    ## calculate depth L1 error
    depth_l1_erorrs = []
    for est_depth_file in est_depth_files:
        est_depth = cv2.imread(os.path.join(est_depth_path, est_depth_file), cv2.IMREAD_UNCHANGED)
        # est_depth = est_depth[:,:,0]
        est_depth = est_depth / 6553.5

        est_depth_scaled = est_depth * scale
        est_depth_mask = (est_depth > 0.)

        # est_depth_mask = est_depth > 0.
        # est_depth_scaled = np.clip(est_depth_scaled, 0., 10.)

        est_depth_file_ = est_depth_file.split(".")[0]
        est_depth_idx = int(est_depth_file_.split("_")[1])
        
        gt_depth_file = gt_depth_files[est_depth_idx]
                    
        gt_depth = cv2.imread(os.path.join(gt_imgs_path, gt_depth_file), cv2.IMREAD_UNCHANGED)
        gt_depth_mask = gt_depth > 0
        gt_depth = gt_depth/6553.5
    
        # mask = np.bitwise_and(est_depth_mask, gt_depth_mask)
        mask = est_depth_mask & gt_depth_mask
        
        diff_depth_l1_ = np.abs(est_depth_scaled.astype(np.float32) - gt_depth.astype(np.float32))
        diff_depth_l1 = diff_depth_l1_ * mask
        l1_error = diff_depth_l1.sum() / mask.sum()
        depth_l1_erorrs.append(l1_error)
        # print(l1_error)

        # vis
        # diff_depth_l1_vis = diff_depth_l1_
        # # diff_depth_l1_vis = cv2.normalize(diff_depth_l1_vis, None, 0, 255, cv2.NORM_MINMAX)
        # diff_depth_l1_vis = diff_depth_l1_vis / 6.0 * 255
        # diff_depth_l1_vis = np.uint8(diff_depth_l1_vis)
        # diff_depth_l1_vis_colormap = cv2.applyColorMap(diff_depth_l1_vis, cv2.COLORMAP_JET)
        # cv2.imshow("depth_error", diff_depth_l1_vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        # est_depth_vis = cv2.normalize(est_depth_scaled, None, 0, 255, cv2.NORM_MINMAX)
        # gt_depth_vis = cv2.normalize(gt_depth, None, 0, 255, cv2.NORM_MINMAX)
        # est_depth_vis = est_depth_scaled / 10.0 * 255.
        # gt_depth_vis = gt_depth / 10.0 * 255.
        # est_depth_vis = np.uint8(est_depth_vis)
        # gt_depth_vis = np.uint8(gt_depth_vis)
        
        # est_colormap = cv2.applyColorMap(est_depth_vis, cv2.COLORMAP_JET)
        # gt_colormap = cv2.applyColorMap(gt_depth_vis, cv2.COLORMAP_JET)
        
        # cv2.imshow("est", est_depth_vis)
        # cv2.imshow('gt', gt_depth_vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    print(f"Depth L1: {np.mean(depth_l1_erorrs) * 100.}")
