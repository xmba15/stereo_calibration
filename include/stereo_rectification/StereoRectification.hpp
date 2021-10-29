/**
 * @file    StereoRectification.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
void stereoRectifyUncalibrated(const std::vector<cv::Point2f>& leftPoints, const std::vector<cv::Point2f>& rightPoints,
                               const cv::Mat& F, const cv::Size& imageSize, cv::Mat& Hl, cv::Mat& Hr);
}  // namespace _cv
