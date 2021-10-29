/**
 * @file    FundamentalMatrix.cpp
 *
 * @author  btran
 *
 */

#include <numeric>

#include <ceres/ceres.h>
#include <opencv2/core/eigen.hpp>

#include <stereo_rectification/FundamentalMatrix.hpp>

namespace _cv
{
namespace
{
std::pair<float, float> calculateMeanVariance(const std::vector<float>& xs)
{
    if (xs.empty()) {
        throw std::runtime_error("empty vector");
    }

    const float mean = std::accumulate(std::begin(xs), std::end(xs), 0.0) / xs.size();
    const float variance = std::accumulate(std::begin(xs), std::end(xs), 0.0,
                                           [mean](float sum, const float elem) {
                                               const float tmp = elem - mean;
                                               return sum + tmp * tmp;
                                           }) /
                           xs.size();
    return std::make_pair(mean, variance);
}

cv::Mat calculateNormalizationMatrix(const std::vector<cv::Point2f>& points)
{
    if (points.empty()) {
        throw std::runtime_error("empty points");
    }

    std::vector<float> xs, ys;
    xs.reserve(points.size());
    ys.reserve(points.size());
    for (const auto& point : points) {
        xs.emplace_back(point.x);
        ys.emplace_back(point.y);
    }

    float xsMean, xsVariance, ysMean, ysVariance;
    std::tie(xsMean, xsVariance) = calculateMeanVariance(xs);
    std::tie(ysMean, ysVariance) = calculateMeanVariance(ys);

    float sx = std::sqrt(2 / (xsVariance + std::numeric_limits<float>::epsilon()));
    float sy = std::sqrt(2 / (ysVariance + std::numeric_limits<float>::epsilon()));

    // clang-format off
    cv::Mat normMatrix = (cv::Mat_<float>(3, 3) << sx , 0.0, -sx * xsMean,
                                                   0.0, sy , -sy * ysMean,
                                                   0.0, 0.0, 1.0
                         );
    // clang-format on

    return normMatrix;
}

cv::Mat toHomogenous(const cv::Mat& mat)
{
    if (mat.channels() != 1) {
        throw std::runtime_error("invalid number of channels");
    }

    cv::Mat concatMat;
    cv::hconcat(mat, cv::Mat::ones(mat.rows, 1, mat.type()), concatMat);

    return concatMat;
}

cv::Mat toInHomogenous(const cv::Mat& mat)
{
    if (mat.channels() != 1) {
        throw std::runtime_error("invalid number of channels");
    }

    if (mat.cols < 2) {
        throw std::runtime_error("matrix's number of columns must be at least 2");
    }

    cv::Mat divisor = cv::repeat(mat.col(mat.cols - 1), 1, mat.cols - 1);
    return mat.colRange(0, mat.cols - 1) / divisor;
}

struct RefineFundamentalMatResidual {
    RefineFundamentalMatResidual(const cv::Point2f& xl, const cv::Point2f& xr)
        : m_xl(xl)
        , m_xr(xr)
    {
    }

    template <typename T> bool operator()(const T* const PrCoeffs, const T* const X, T* residuals) const
    {
        Eigen::Matrix<T, 3, 4> Pr(PrCoeffs);

        Eigen::Matrix<T, 2, 1> xl(static_cast<T>(m_xl.x), static_cast<T>(m_xl.y));
        Eigen::Matrix<T, 2, 1> xr(static_cast<T>(m_xr.x), static_cast<T>(m_xr.y));
        Eigen::Matrix<T, 2, 1> xlhat(static_cast<T>(X[0] / X[2]), X[1] / X[2]);
        Eigen::Matrix<T, 4, 1> homogenousX(X[0], X[1], X[2], static_cast<T>(1));

        Eigen::Matrix<T, 3, 1> homogenousXrhat = Pr * homogenousX;
        Eigen::Matrix<T, 2, 1> xrhat(static_cast<T>(homogenousXrhat[0] / homogenousXrhat[2]),
                                     static_cast<T>(homogenousXrhat[1] / homogenousXrhat[2]));

        auto diffl = xl - xlhat;
        auto diffr = xr - xrhat;

        residuals[0] = diffl[0];
        residuals[1] = diffl[1];
        residuals[2] = diffr[0];
        residuals[3] = diffr[1];

        return true;
    }

 private:
    const cv::Point2f& m_xl;
    const cv::Point2f& m_xr;
};
}  // namespace

cv::Mat findFundamentalMat(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2)
{
    int numImagePoints = points1.size();

    cv::Mat pointsMat1(numImagePoints, 2, CV_32FC1), pointsMat2(numImagePoints, 2, CV_32FC1);
    std::memcpy(pointsMat1.data, points1.data(), pointsMat1.total() * pointsMat1.elemSize());
    std::memcpy(pointsMat2.data, points2.data(), pointsMat2.total() * pointsMat2.elemSize());

    cv::Mat normMat1 = calculateNormalizationMatrix(points1);
    cv::Mat normMat2 = calculateNormalizationMatrix(points2);

    cv::Mat normalizedPoints1 = toInHomogenous((normMat1 * toHomogenous(pointsMat1).t()).t());
    cv::Mat normalizedPoints2 = toInHomogenous((normMat2 * toHomogenous(pointsMat2).t()).t());

    cv::Mat M = cv::Mat::zeros(numImagePoints, 9, CV_32FC1);
    for (int i = 0; i < numImagePoints; ++i) {
        auto p1 = normalizedPoints1.ptr<float>(i);
        auto p2 = normalizedPoints2.ptr<float>(i);

        cv::Mat curRow = (cv::Mat_<float>(1, 9) << p1[0] * p2[0], p1[1] * p2[0], p2[0], p1[0] * p2[1], p1[1] * p2[1],
                          p2[1], p1[0], p1[1], 1);
        curRow.copyTo(M.row(i));
    }

    cv::Mat W, U, Vt;
    cv::SVD::compute(M, W, U, Vt);
    cv::Mat F = Vt.row(8).reshape(1, 3);

    cv::Mat FW, FU, FVt;
    cv::SVD::compute(F, FW, FU, FVt);

    FW.ptr<float>(0)[2] = 0;
    F = FU * cv::Mat::diag(FW) * FVt;
    F = normMat2.t() * F * normMat1;

    return F / F.ptr<float>(2)[2];
}

void computeCorrespondEpilines(const std::vector<cv::Point2f>& points, int whichImage, const cv::Mat& F,
                               std::vector<cv::Vec3f>& lines)
{
    int numImagePoints = points.size();
    lines.reserve(numImagePoints);

    cv::Mat pointsMat(numImagePoints, 2, CV_32FC1);
    std::memcpy(pointsMat.data, points.data(), pointsMat.total() * pointsMat.elemSize());
    cv::Mat linesMat = whichImage == 1 ? F * toHomogenous(pointsMat).t() : F.t() * toHomogenous(pointsMat).t();
    linesMat = linesMat.t();

    for (int i = 0; i < numImagePoints; ++i) {
        auto rowPtr = linesMat.ptr<float>(i);
        double nu = rowPtr[0] * rowPtr[0] + rowPtr[1] * rowPtr[1];
        nu = nu ? 1. / std::sqrt(nu) : 1;
        lines.emplace_back(cv::Vec3f(rowPtr[0] * nu, rowPtr[1] * nu, rowPtr[2] * nu));
    }
}

cv::Point3f triangulatePoint(const cv::Point2f& point1, const cv::Point2f& point2, const cv::Mat& P1, const cv::Mat& P2)
{
    cv::Mat A(4, 4, CV_32FC1);

    cv::Mat(point1.x * P1.row(2) - P1.row(0)).copyTo(A.row(0));
    cv::Mat(point1.y * P1.row(2) - P1.row(1)).copyTo(A.row(1));
    cv::Mat(point2.x * P2.row(2) - P2.row(0)).copyTo(A.row(2));
    cv::Mat(point2.y * P2.row(2) - P2.row(1)).copyTo(A.row(3));

    cv::Mat W, U, Vt;
    cv::SVD::compute(A, W, U, Vt);

    cv::Mat homogenousOutput = Vt.row(3);
    homogenousOutput /= homogenousOutput.ptr<float>(0)[3];

    return cv::Point3f(homogenousOutput.colRange(0, 3));
}

std::vector<cv::Point3f> triangulatePoints(const std::vector<cv::Point2f>& points1,
                                           const std::vector<cv::Point2f>& points2, const cv::Mat& P1,
                                           const cv::Mat& P2)
{
    std::vector<cv::Point3f> output;
    output.reserve(points1.size());
    for (std::size_t i = 0; i < points1.size(); ++i) {
        output.emplace_back(triangulatePoint(points1[i], points2[i], P1, P2));
    }
    return output;
}

void refineFundamentalMat(cv::Mat& F, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2)
{
    cv::Mat W, U, Vt;
    cv::SVD::compute(F, W, U, Vt);

    // epipole in the right camera. F^T*er = 0
    cv::Mat er = U.col(2).t();

    auto getSkewSymMatrix = [](const cv::Mat& v) -> cv::Mat {
        assert(v.size() == cv::Size(3, 1));
        const float* data = v.ptr<float>();
        cv::Mat output = (cv::Mat_<float>(3, 3) << 0, -data[2], data[1], data[2], 0, -data[0], -data[1], data[0], 0);
        return output;
    };

    cv::Mat Pl = cv::Mat::zeros(3, 4, CV_32FC1), Pr = cv::Mat::zeros(3, 4, CV_32FC1);
    Pl.colRange(0, 3) = cv::Mat::eye(3, 3, CV_32FC1);
    Pr.colRange(0, 3) = getSkewSymMatrix(er) * F;
    cv::Mat(er.t()).copyTo(Pr.col(3));
    std::vector<cv::Point3f> Xs = _cv::triangulatePoints(points1, points2, Pl, Pr);

    Pr.convertTo(Pr, CV_64FC1);
    Eigen::Matrix<double, 3, 4> eigenPr;
    cv::cv2eigen(Pr, eigenPr);
    ceres::Problem prob;

    std::vector<std::array<double, 3>> XsV(Xs.size());
    for (std::size_t i = 0; i < points1.size(); ++i) {
        XsV[i] = {Xs[i].x, Xs[i].y, Xs[i].z};
        prob.AddResidualBlock(new ceres::AutoDiffCostFunction<RefineFundamentalMatResidual, 4, 12, 3>(
                                  new RefineFundamentalMatResidual(points1[i], points2[i])),
                              nullptr, eigenPr.data(), XsV[i].data());
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_SCHUR;

#ifdef DEBUG
    options.minimizer_progress_to_stdout = true;
#endif

    ceres::Solver::Summary summary;
    ceres::Solve(options, &prob, &summary);

#ifdef DEBUG
    std::cout << summary.BriefReport() << std::endl;
#endif

    cv::eigen2cv(eigenPr, Pr);
    Pr.convertTo(Pr, CV_32FC1);
    F = getSkewSymMatrix(Pr.col(3).t()) * Pr.colRange(0, 3);

    F.convertTo(F, CV_32FC1);
    F /= F.ptr<float>(2)[2];
}

void estimateEpipoles(const cv::Mat& F, cv::Vec3f* leftEpipole, cv::Vec3f* rightEpipole)
{
    if (!leftEpipole && !rightEpipole) {
        return;
    }

    cv::Mat FW, FU, FVt;
    cv::SVD::compute(F, FW, FU, FVt);
    if (leftEpipole) {
        *leftEpipole = FVt.row(2);
        *leftEpipole /= (*leftEpipole)[2];
    }

    if (rightEpipole) {
        *rightEpipole = FU.col(2);
        *rightEpipole /= (*rightEpipole)[2];
    }
}
}  // namespace _cv
