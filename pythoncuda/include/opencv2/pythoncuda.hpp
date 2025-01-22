/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
//################################################################################
//
//                    Created by Neeraj Gulia
//
//################################################################################

#ifndef __OPENCV_PYTHONCUDA_HPP__
#define __OPENCV_PYTHONCUDA_HPP__

#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace cv {
namespace pythoncuda {
CV_EXPORTS_W void cpuOpticalFlowFarneback(InputArray prev, InputArray next, InputOutputArray flow, double pyr_scale,
                                          int levels, int winsize, int iterations, int poly_n, double poly_sigma,
                                          int flags);

CV_EXPORTS_W void gpuOpticalFlowFarneback(InputArray prev, InputArray next, InputOutputArray flow, double pyr_scale,
                                          int levels, int winsize, int iterations, int poly_n, double poly_sigma,
                                          int flags);

CV_EXPORTS_W void cpuOpticalFlowPyrLK(InputArray prevImg, InputArray nextImg, InputArray prevPts,
                                      InputOutputArray nextPts, OutputArray status, OutputArray err,
                                      Size winSize = Size(21, 21), int maxLevel = 3,
                                      TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30,
                                                                           0.01),
                                      int flags = 0, double minEigThreshold = 1e-4);

CV_EXPORTS_W void gpuOpticalFlowPyrLK(InputArray prevImg, InputArray nextImg, InputArray prevPts,
                                      InputOutputArray nextPts, OutputArray status, OutputArray err,
                                      Size winSize = Size(21, 21), int maxLevel = 3, int iterations = 30);

CV_EXPORTS_W void cudaPnP(InputArray obj, InputArray im, int num_points, int num_iterations, float repr,
                          InputArray cammat, OutputArray _returns);
}  // namespace pythoncuda
}  // namespace cv
#endif /* __OPENCV_PYTHONCUDA_HPP__ */
