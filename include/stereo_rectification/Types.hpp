/**
 * @file    Types.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
struct CameraInfo {
    cv::Size imageSize;

    // intrinsic camera matrix for the raw images
    cv::Mat K;

    // distortion parameters
    cv::Mat D;
};
}  // namespace _cv
