/**
 * @file    ProjectiveTransform.cpp
 *
 * @author  btran
 *
 */

#include <ceres/ceres.h>
#include <opencv2/core/eigen.hpp>

#include <stereo_rectification/FundamentalMatrix.hpp>

#include "TransformEstimation.hpp"

namespace _cv
{
namespace
{
template <typename DataType>
inline bool almostEquals(DataType input, DataType other, DataType rtol = 1e-05, DataType atol = 1e-08)
{
    return std::abs(input - other) <= atol + rtol * std::abs(other);
}

cv::Mat mapToInfinity(const cv::Vec3f& p, const cv::Point2f& u0)
{
    cv::Mat T = (cv::Mat_<float>(3, 3) << 1, 0, -u0.x, 0, 1, -u0.y, 0, 0, 1);

    cv::Vec3f pt = cv::Mat(T * cv::Mat(p));

    float alpha = pt[0] < 0 ? -1 : 1;
    float d = std::sqrt(pt[0] * pt[0] + pt[1] * pt[1]);
    // clang-format off
    cv::Mat R = (cv::Mat_<float>(3, 3) << alpha * pt[0] / d, alpha * pt[1] / d, 0,
                                         -alpha * pt[1] / d, alpha * pt[0] / d, 0,
                                          0, 0, 1);
    // clang-format on

    pt = cv::Mat(R * cv::Mat(pt));

    cv::Mat G = cv::Mat::eye(3, 3, CV_32F);
    if (!almostEquals<float>(std::abs(pt[2] / cv::norm(pt)), 0)) {
        G.at<float>(2, 0) = -pt[2] / pt[0];
    }

    cv::Mat H = G * R * T;

    // map (f, 0, 0) to (1, 0, 0)
    pt = cv::Mat(H * cv::Mat(p));
    H /= pt[0];

    cv::Mat Tinv = (cv::Mat_<float>(3, 3) << 1, 0, u0.x, 0, 1, u0.y, 0, 0, 1);
    H = Tinv * H;

    return H;
}

void getInliers(const cv::Mat& F, const std::vector<cv::Point2f>& leftPoints,
                const std::vector<cv::Point2f>& rightPoints, std::vector<cv::Point2f>& inlierLeftPoints,
                std::vector<cv::Point2f>& inlierRightPoints, float threshold)
{
    int numPoints = leftPoints.size();

    std::vector<cv::Vec3f> leftLines, rightLines;
    _cv::computeCorrespondEpilines(leftPoints, 1, F, rightLines);
    _cv::computeCorrespondEpilines(rightPoints, 2, F, leftLines);

    auto calcDistPointToLine = [](const cv::Point2f& point, const cv::Vec3f& line) -> float {
        return std::abs(point.x * line[0] + point.y * line[1] + line[2]);
    };

    for (int i = 0; i < numPoints; ++i) {
        if (calcDistPointToLine(leftPoints[i], leftLines[i]) <= threshold &&
            calcDistPointToLine(rightPoints[i], rightLines[i]) <= threshold) {
            inlierLeftPoints.emplace_back(leftPoints[i]);
            inlierRightPoints.emplace_back(rightPoints[i]);
        }
    }
}
}  // namespace

void estimateProjectiveTransform(const std::vector<cv::Point2f>& leftPoints,
                                 const std::vector<cv::Point2f>& rightPoints, const cv::Mat& F, const cv::Size& size,
                                 cv::Mat& HpLeft, cv::Mat& HpRight, float threshold)
{
    cv::Vec3f rightEpipole;
    estimateEpipoles(F, nullptr, &rightEpipole);

    HpRight = mapToInfinity(rightEpipole, cv::Point2f((size.width - 1) / 2., (size.height - 1) / 2.));

    auto getSkewSymMatrix = [](const cv::Vec3f& v) -> cv::Mat {
        return (cv::Mat_<float>(3, 3) << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0);
    };

    cv::Mat M = getSkewSymMatrix(rightEpipole) * F + cv::Mat(rightEpipole) * cv::Mat::ones(1, 3, CV_32F);
    cv::Mat H0 = HpRight * M;

    std::vector<cv::Point2f> inlierLeftPoints, inlierRightPoints;
    getInliers(F, leftPoints, rightPoints, inlierLeftPoints, inlierRightPoints, threshold);

    if (inlierLeftPoints.empty()) {
        inlierLeftPoints = leftPoints;
        inlierRightPoints = rightPoints;
    }

    std::vector<cv::Point2f> leftPointsHat, rightPointsHat;
    cv::perspectiveTransform(inlierLeftPoints, leftPointsHat, H0);
    cv::perspectiveTransform(inlierRightPoints, rightPointsHat, HpRight);

    cv::Mat A = cv::Mat::ones(inlierLeftPoints.size(), 3, CV_32F);
    cv::Mat b(inlierLeftPoints.size(), 1, CV_32F);
    for (std::size_t i = 0; i < inlierLeftPoints.size(); ++i) {
        A.at<float>(i, 0) = leftPointsHat[i].x;
        A.at<float>(i, 1) = leftPointsHat[i].y;
        b.at<float>(i, 0) = rightPointsHat[i].x;
    }

    cv::Mat coeffs;
    cv::solve(A, b, coeffs, cv::DECOMP_SVD);
    cv::Mat Ha = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat(coeffs.t()).copyTo(Ha.row(0));
    HpLeft = Ha * H0;
}
}  // namespace _cv
