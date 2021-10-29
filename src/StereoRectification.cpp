/**
 * @file    StereoRectification.cpp
 *
 * @author  btran
 *
 */

#include <stereo_rectification/StereoRectification.hpp>

#include "TransformEstimation.hpp"

namespace _cv
{
void stereoRectifyUncalibrated(const std::vector<cv::Point2f>& leftPoints, const std::vector<cv::Point2f>& rightPoints,
                               const cv::Mat& F, const cv::Size& imageSize, cv::Mat& HLeft, cv::Mat& HRight)
{
    estimateProjectiveTransform(leftPoints, rightPoints, F, imageSize, HLeft, HRight);

    cv::Mat HsLeft, HsRight;
    estimateShearingTransform(F, imageSize, HLeft, HRight, HsLeft, HsRight);

    HLeft *= HsLeft;
    HRight *= HsRight;
}
}  // namespace _cv
