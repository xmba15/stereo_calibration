/**
 * @file    FundamentalMatrix.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
cv::Mat findFundamentalMat(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2);

void computeCorrespondEpilines(const std::vector<cv::Point2f>& points, int whichImage, const cv::Mat& F,
                               std::vector<cv::Vec3f>& lines);

cv::Point3f triangulatePoint(const cv::Point2f& point1, const cv::Point2f& point2, const cv::Mat& P1,
                             const cv::Mat& P2);

std::vector<cv::Point3f> triangulatePoints(const std::vector<cv::Point2f>& points1,
                                           const std::vector<cv::Point2f>& points2, const cv::Mat& P1,
                                           const cv::Mat& P2);

/**
 *  @brief refine fundamental matrix by Gold Standard algorithm
 *
 *  See Multiple View Geometry in Computer Vision (p.285)
 */
void refineFundamentalMat(cv::Mat& F, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2);

void estimateEpipoles(const cv::Mat& F, cv::Vec3f* leftEpipole, cv::Vec3f* rightEpipole);
}  // namespace _cv
