#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/register_point_struct.h>

struct PointXYZIndexBW
{
  float x;
  float y;
  float z;
  float gray; // Gray Scale Image Intensity
  int index;  // Keyframe Index

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT (
  PointXYZIndexBW, 
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, gray, gray)
  (int, index, index)
)
