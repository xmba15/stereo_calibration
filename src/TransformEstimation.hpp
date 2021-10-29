/**
 * @file    TransformEstimation.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
/**
 *  brief estimate projective transform according to Hartley's rectification algorithm
 *
 *  Multiple View Geometry in Computer Vision p. 307
 *
 */
void estimateProjectiveTransform(const std::vector<cv::Point2f>& leftPoints,
                                 const std::vector<cv::Point2f>& rightPoints, const cv::Mat& F, const cv::Size& size,
                                 cv::Mat& HpLeft, cv::Mat& HpRight, float threshold = 1);

/**
 *  brief estimate similarity transform according to section 7. of
 *
 *  Computing Rectifying Homographies for Stereo Vision, Loop. et al
 *
 */
void estimateShearingTransform(const cv::Mat& F, const cv::Size& size, const cv::Mat& HrHpLeft,
                               const cv::Mat& HrHpRight, cv::Mat& HsLeft, cv::Mat& HsRight);
}  // namespace _cv
