/**
 * @file    ShearingTransform.cpp
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
struct ShearingTransformEstimationResidual {
    ShearingTransformEstimationResidual(const cv::Size& size, const cv::Mat& x, const cv::Mat& y)
        : m_size(size)
    {
        m_x << x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0);
        m_y << y.at<float>(0, 0), y.at<float>(1, 0), y.at<float>(2, 0);
    }

    template <typename T> bool operator()(const T* const coeffs, T* residuals) const
    {
        Eigen::Matrix<T, 3, 3> S = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(3, 3);
        S(0, 0) = coeffs[0];
        S(0, 1) = coeffs[1];

        auto Sx = S * m_x.cast<T>();
        auto Sy = S * m_y.cast<T>();
        residuals[0] = (Sx.transpose() * Sy)(0, 0);
        residuals[1] = (Sx.transpose() * Sx)(0, 0) / (Sy.transpose() * Sy)(0, 0) -
                       static_cast<T>(m_size.width * m_size.width / (m_size.height * m_size.height));

        return true;
    }

 private:
    const cv::Size& m_size;
    Eigen::Matrix<double, 3, 1> m_x;
    Eigen::Matrix<double, 3, 1> m_y;
};

void estimateSingleShearingTransform(const cv::Mat& F, const cv::Size& size, const cv::Mat& HrHp, cv::Mat& Hs)
{
    std::vector<cv::Point2f> midPoints = {
        cv::Point2f(0, (size.width - 1) / 2.), cv::Point2f((size.height - 1) / 2., size.width - 1),
        cv::Point2f(size.height - 1, (size.width - 1) / 2.), cv::Point2f((size.height - 1) / 2., 0)};

    std::vector<cv::Point2f> projectedMidPoints;
    cv::perspectiveTransform(midPoints, projectedMidPoints, HrHp);

    cv::Point2f diff1 = projectedMidPoints[1] - projectedMidPoints[3];
    cv::Point2f diff2 = projectedMidPoints[2] - projectedMidPoints[0];

    double a = (size.height * size.height * diff1.y * diff1.y + size.width * size.width * diff2.y * diff2.y) /
               (size.height * size.width * (diff1.y * diff2.x - diff1.x * diff2.y));
    double b = (size.height * size.height * diff1.x * diff1.y + size.width * size.width * diff2.x * diff2.y) /
               (size.height * size.width * (-diff1.y * diff2.x + diff1.x * diff2.y));

    cv::Mat xMat = (cv::Mat_<float>(3, 1) << diff1.x, diff1.y, 0);
    cv::Mat yMat = (cv::Mat_<float>(3, 1) << diff2.x, diff2.y, 0);

    double coeffs[2] = {a, b};

    ceres::Problem prob;
    prob.AddResidualBlock(new ceres::AutoDiffCostFunction<ShearingTransformEstimationResidual, 2, 2>(
                              new ShearingTransformEstimationResidual(size, xMat, yMat)),
                          nullptr, coeffs);

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &prob, &summary);

    Hs = cv::Mat::eye(3, 3, CV_32F);
    Hs.at<float>(0, 0) = coeffs[0];
    Hs.at<float>(0, 1) = coeffs[1];
}
}  // namespace

void estimateShearingTransform(const cv::Mat& F, const cv::Size& size, const cv::Mat& HrHpLeft,
                               const cv::Mat& HrHpRight, cv::Mat& HsLeft, cv::Mat& HsRight)
{
    estimateSingleShearingTransform(F, size, HrHpLeft, HsLeft);
    estimateSingleShearingTransform(F, size, HrHpRight, HsRight);
}
}  // namespace _cv
