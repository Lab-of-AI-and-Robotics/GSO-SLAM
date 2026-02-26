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
import numpy
import argparse
import os
# import rerun as rr
from natsort import natsorted

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
    numpy.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3, -1))
    data_zerocentered = data - data.mean(1).reshape((3, -1))
    
    W = numpy.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = numpy.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity( 3 ))
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    rotmodel = rot*model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += numpy.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
        normi = numpy.linalg.norm(model_zerocentered[:,column])
        norms += normi*normi

    s = float(dots/norms)    

    # print("scale: %f " % s  )
    
    trans = data.mean(1).reshape((3,-1)) - s*rot * model.mean(1).reshape((3,-1))
    
    model_aligned = s*rot * model + trans
    alignment_error = model_aligned - data
    
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error, alignment_error),0)).A[0]
        
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
    interval = numpy.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
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
    parser.add_argument('gt_path', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('est_path', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('association', help='association file')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)',default=1.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
    parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    args = parser.parse_args()
    
    rgb = []
    depth = []
    
    dataset_path = args.gt_path
    
    rgb_timestamps = []
    with open(args.association, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            if len(parts) == 2:
                timestamp1, sRGB = parts
                rgb_timestamps.append(timestamp1)  
            elif len(parts) == 4:
                timestamp1, sRGB, timestamp2, sD = parts
                rgb_timestamps.append(timestamp1)  
    
    # load gt pose
    if os.path.isfile(os.path.join(dataset_path, 'groundtruth.txt')):
        gt_pose_file = os.path.join(dataset_path, 'groundtruth.txt')
    elif os.path.isfile(os.path.join(dataset_path, 'pose.txt')):
        gt_pose_file = os.path.join(dataset_path, 'pose.txt')
        
    pose_data = numpy.loadtxt(gt_pose_file, delimiter=' ',
                          dtype=numpy.str_, skiprows=1)
    pose_vecs = pose_data[:, 1:].astype(numpy.float64)
    tstamp_pose = pose_data[:, 0].astype(numpy.float64)
    
    gt_xyz_whole_ = pose_vecs[:,:3]
    gt_xyz_whole_ = numpy.array(gt_xyz_whole_)  # [n,3]

    # 
    gt_xyz_whole =  []
    for rgb_timestamp in rgb_timestamps:
        t_errors = (tstamp_pose - (float)(rgb_timestamp))**2
        matched_idx = numpy.argmin(t_errors, axis=0)
        gt_xyz_whole.append([gt_xyz_whole_[matched_idx,0], gt_xyz_whole_[matched_idx,1], gt_xyz_whole_[matched_idx,2]])
    gt_xyz_whole = numpy.array(gt_xyz_whole)
    # print(gt_xyz_whole.shape)

    # load estimated traj
    est_idxs = []
    est_xyz = []
    est_pose_path = os.path.join(args.est_path, "pose")
    est_pose_file_list = natsorted(os.listdir(est_pose_path))
    for est_pose_file in est_pose_file_list:
        est_c2w = numpy.loadtxt(os.path.join(est_pose_path, est_pose_file))
        est_xyz.append([est_c2w[0,3], est_c2w[1,3], est_c2w[2,3]])
        est_idxs.append((int)(est_pose_file.split('.')[0]))
    est_xyz = numpy.array(est_xyz)
    
    # associate (g.t. <-> est)
    gt_xyz = gt_xyz_whole[est_idxs]

    gt_xyz = gt_xyz.T
    est_xyz = est_xyz.T
    
    rot,trans,trans_error,scale = align(est_xyz,gt_xyz)
    
    est_xyz_aligned = scale * rot * est_xyz + trans
    
    # print("%f"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)))
    print(f"ATE RMSE: {trans_error.mean()*100}")
    # print(trans_error.mean()*100)

    # plot
    # rr.init("Evaluation result", spawn=True)
    
    # # ## GT
    # rr.log(
    #     "Ground_Truth_Traj",
    #     rr.Points3D(gt_xyz.T)
    # )
    
    # ## Est
    # rr.log(
    #     "Estimate_Traj",
    #     rr.Points3D(est_xyz_aligned.T)
    # )
        
