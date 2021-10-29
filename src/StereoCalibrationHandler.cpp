/**
 * @file    StereoCalibrationHandler.cpp
 *
 * @author  btran
 *
 */

#include <numeric>

#include <stereo_rectification/stereo_rectification.hpp>

#include "TransformEstimation.hpp"

namespace _cv
{
namespace
{
template <typename DataType> std::vector<DataType> concatData(const std::vector<std::vector<DataType>>& data)
{
    std::vector<DataType> output;
    for (const auto& curData : data) {
        output.insert(output.end(), curData.begin(), curData.end());
    }
    return output;
}

void getRectangles(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const cv::Mat& R,
                   const cv::Mat& newCameraMatrix, const cv::Size& imgSize, cv::Rect_<float>& inner,
                   cv::Rect_<float>& outer, int numSamples = 9)
{
    std::vector<cv::Point2f> pts(numSamples * numSamples);
    for (int i = 0; i < numSamples; ++i) {
        for (int j = 0; j < numSamples; ++j) {
            pts[i * numSamples + j] = cv::Point2f(1. * j * (imgSize.width - 1) / (numSamples - 1),
                                                  1. * i * (imgSize.height - 1) / (numSamples - 1));
        }
    }
    cv::undistortPoints(pts, pts, cameraMatrix, distCoeffs, R, newCameraMatrix);

    float iX0 = -FLT_MAX, iX1 = FLT_MAX, iY0 = -FLT_MAX, iY1 = FLT_MAX;
    float oX0 = FLT_MAX, oX1 = -FLT_MAX, oY0 = FLT_MAX, oY1 = -FLT_MAX;

    for (int i = 0; i < numSamples; ++i) {
        for (int j = 0; j < numSamples; ++j) {
            const auto& p = pts[i * numSamples + j];
            oX0 = std::min(oX0, p.x);
            oX1 = std::max(oX1, p.x);
            oY0 = std::min(oY0, p.y);
            oY1 = std::max(oY1, p.y);

            if (j == 0) {
                iX0 = std::max(iX0, p.x);
            }
            if (j == numSamples - 1) {
                iX1 = std::min(iX1, p.x);
            }
            if (i == 0) {
                iY0 = std::max(iY0, p.y);
            }
            if (i == numSamples - 1) {
                iY1 = std::min(iY1, p.y);
            }
        }
    }
    inner = cv::Rect_<float>(iX0, iY0, iX1 - iX0, iY1 - iY0);
    outer = cv::Rect_<float>(oX0, oY0, oX1 - oX0, oY1 - oY0);
}
}  // namespace

StereoCalibrationHandler::StereoCalibrationHandler(const Param& param)
    : m_param(param)
    , m_patternSize(m_param.numCol, m_param.numRow)
    , m_imageSize(0, 0)
{
    validate(param);
    this->getRawIntrinsicParams();
    m_imageList = parseMetaDataFile(param.imageListFile);
    m_objectPoints = this->generateObjectPoints(m_param.numRow, m_param.numCol, m_param.squareSize);
    this->getImagePoints(m_allLeftImagePoints, m_allRightImagePoints);
}

cv::Mat StereoCalibrationHandler::findFundamentalMat()
{
    auto allLeft = concatData(m_allLeftImagePoints);
    auto allRight = concatData(m_allRightImagePoints);

    cv::Mat F = _cv::findFundamentalMat(allLeft, allRight);
    _cv::refineFundamentalMat(F, allLeft, allRight);

    return F;
}

void StereoCalibrationHandler::drawEpipolarLines(const cv::Mat& F) const
{
    std::string outputPath = "/tmp";
    for (std::size_t i = 0; i < m_imageList.size(); ++i) {
        cv::Mat leftImage = cv::imread(m_param.leftImagePath + "/" + m_imageList[i]);
        cv::Mat rightImage = cv::imread(m_param.rightImagePath + "/" + m_imageList[i]);

        const auto& leftPoints = m_allLeftImagePoints[i];
        const auto& rightPoints = m_allRightImagePoints[i];
        cv::drawChessboardCorners(leftImage, m_patternSize, cv::Mat(leftPoints), true);
        cv::drawChessboardCorners(rightImage, m_patternSize, cv::Mat(rightPoints), true);

        std::vector<cv::Vec3f> leftLines, rightLines;
        _cv::computeCorrespondEpilines(leftPoints, 1, F, rightLines);
        _cv::computeCorrespondEpilines(rightPoints, 2, F, leftLines);

        for (const auto& line : rightLines) {
            auto& image = rightImage;
            cv::line(image, cv::Point(0, -line[2] / line[1]),
                     cv::Point(image.cols, -(line[2] + line[0] * image.cols) / line[1]), cv::Scalar(0, 0, 255));
        }
        for (const auto& line : leftLines) {
            auto& image = leftImage;
            cv::line(image, cv::Point(0, -line[2] / line[1]),
                     cv::Point(image.cols, -(line[2] + line[0] * image.cols) / line[1]), cv::Scalar(0, 0, 255));
        }
        cv::Mat concat;
        cv::hconcat(leftImage, rightImage, concat);
        cv::imwrite(outputPath + "/" + m_imageList[i], concat);
    }
}

double StereoCalibrationHandler::computeEpipolarErr(const cv::Mat& F) const
{
    double err = 0.;
    for (std::size_t i = 0; i < m_imageList.size(); ++i) {
        const auto& leftPoints = m_allLeftImagePoints[i];
        const auto& rightPoints = m_allRightImagePoints[i];

        std::vector<cv::Vec3f> leftLines, rightLines;
        _cv::computeCorrespondEpilines(leftPoints, 1, F, rightLines);
        _cv::computeCorrespondEpilines(rightPoints, 2, F, leftLines);

        for (std::size_t i = 0; i < leftPoints.size(); ++i) {
            const auto& line = leftLines[i];
            const auto& point = leftPoints[i];
            err += std::abs(point.x * line[0] + point.y * line[1] + line[2]);
        }

        for (std::size_t i = 0; i < rightPoints.size(); ++i) {
            const auto& line = rightLines[i];
            const auto& point = rightPoints[i];
            err += std::abs(point.x * line[0] + point.y * line[1] + line[2]);
        }
    }

    return err / (m_allLeftImagePoints.size() * m_allLeftImagePoints[0].size());
}

void StereoCalibrationHandler::rectifyUncalibrated(const cv::Mat& F, cv::Mat& Hl, cv::Mat& Hr) const
{
    auto allLeft = concatData(m_allLeftImagePoints);
    auto allRight = concatData(m_allRightImagePoints);

    _cv::stereoRectifyUncalibrated(allLeft, allRight, F, m_imageSize, Hl, Hr);

    Hl /= Hl.at<float>(2, 2);
    Hr /= Hr.at<float>(2, 2);
}

void StereoCalibrationHandler::getNewCameraInfo(const cv::Mat& Hl, const cv::Mat& Hr, CameraInfo& newCamInfo,
                                                cv::Mat& Rl, cv::Mat& Rr, float alpha) const
{
    Rl = m_leftCamInfo.K.inv() * Hl * m_leftCamInfo.K;
    Rr = m_rightCamInfo.K.inv() * Hr * m_rightCamInfo.K;

    newCamInfo.K = cv::Mat::eye(3, 3, CV_32F);
    newCamInfo.K.at<float>(1, 1) = (m_leftCamInfo.K.at<float>(1, 1) + m_rightCamInfo.K.at<float>(1, 1)) / 2.;
    newCamInfo.K.at<float>(0, 0) = newCamInfo.K.at<float>(1, 1);

    int width = m_imageSize.width;
    int height = m_imageSize.height;

    int numSamples = 2;
    std::vector<cv::Point2f> observedPoints(numSamples * numSamples);
    for (int i = 0; i < numSamples; ++i) {
        for (int j = 0; j < numSamples; ++j) {
            observedPoints[i * numSamples + j] = cv::Point2f(1. * j * (m_imageSize.width - 1) / (numSamples - 1),
                                                             1. * i * (m_imageSize.height - 1) / (numSamples - 1));
        }
    }

    float newCx, newCy;
    newCx = newCy = 0;
    for (int i = 0; i < 2; ++i) {
        const auto& K = i == 0 ? m_leftCamInfo.K : m_rightCamInfo.K;
        const auto& D = i == 0 ? m_leftCamInfo.D : m_rightCamInfo.D;
        std::vector<cv::Point2f> dst2D(observedPoints.size());
        cv::Mat dst3D;
        cv::undistortPoints(observedPoints, dst2D, K, D);
        cv::convertPointsToHomogeneous(cv::Mat(dst2D), dst3D);
        cv::Mat rvec, tvec;
        tvec = cv::Mat::zeros(3, 1, CV_32F);
        cv::Rodrigues(i == 0 ? Rl : Rr, rvec);
        cv::projectPoints(dst3D, rvec, tvec, newCamInfo.K, cv::Mat::zeros(5, 1, CV_32F), dst2D);
        cv::Scalar avg = cv::mean(cv::Mat(dst2D));
        newCx += (width - 1) / 2. - avg[0];
        newCy += (height - 1) / 2. - avg[1];
    }
    newCx /= 2;
    newCy /= 2;
    newCamInfo.K.at<float>(0, 2) = newCx;
    newCamInfo.K.at<float>(1, 2) = newCy;

    cv::Rect_<float> inner1, inner2, outer1, outer2;
    getRectangles(m_leftCamInfo.K, m_leftCamInfo.D, Rl, newCamInfo.K, m_imageSize, inner1, outer1);
    getRectangles(m_rightCamInfo.K, m_rightCamInfo.D, Rr, newCamInfo.K, m_imageSize, inner2, outer2);

    double s = 1.;
    double cx1 = newCamInfo.K.at<float>(0, 2);
    double cy1 = newCamInfo.K.at<float>(1, 2);
    if (alpha >= 0) {
        double s0 = std::max(std::max(std::max((double)cx1 / (cx1 - inner1.x), (double)cy1 / (cy1 - inner1.y)),
                                      (double)(m_imageSize.width - cx1) / (inner1.x + inner1.width - cx1)),
                             (double)(m_imageSize.height - cy1) / (inner1.y + inner1.height - cy1));
        s0 = std::max(std::max(std::max(std::max((double)cx1 / (cx1 - inner2.x), (double)cy1 / (cy1 - inner2.y)),
                                        (double)(m_imageSize.width - cx1) / (inner2.x + inner2.width - cx1)),
                               (double)(m_imageSize.height - cy1) / (inner2.y + inner2.height - cy1)),
                      s0);

        double s1 = std::min(std::min(std::min((double)cx1 / (cx1 - outer1.x), (double)cy1 / (cy1 - outer1.y)),
                                      (double)(m_imageSize.width - cx1) / (outer1.x + outer1.width - cx1)),
                             (double)(m_imageSize.height - cy1) / (outer1.y + outer1.height - cy1));
        s1 = std::min(std::min(std::min(std::min((double)cx1 / (cx1 - outer2.x), (double)cy1 / (cy1 - outer2.y)),
                                        (double)(m_imageSize.width - cx1) / (outer2.x + outer2.width - cx1)),
                               (double)(m_imageSize.height - cy1) / (outer2.y + outer2.height - cy1)),
                      s1);

        s = s0 * (1 - alpha) + s1 * alpha;
    }
    newCamInfo.K.at<float>(0, 0) *= s;
    newCamInfo.K.at<float>(1, 1) *= s;
}

std::vector<cv::Point3f> StereoCalibrationHandler::generateObjectPoints(int numRow, int numCol, float squareSize) const
{
    std::vector<cv::Point3f> targetPoints;
    targetPoints.reserve(numRow * numCol);

    for (int i = 0; i < numRow; ++i) {
        for (int j = 0; j < numCol; ++j) {
            targetPoints.emplace_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }

    return targetPoints;
}

void StereoCalibrationHandler::getImagePoints(std::vector<std::vector<cv::Point2f>>& allLeftImagePoints,
                                              std::vector<std::vector<cv::Point2f>>& allRightImagePoints)
{
    allLeftImagePoints.clear();
    allRightImagePoints.clear();
    allLeftImagePoints.reserve(m_imageList.size());
    allRightImagePoints.reserve(m_imageList.size());

    const cv::Size subPixWinSize(5, 5);
    const cv::TermCriteria termimateCrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);

    for (auto it = m_imageList.begin(); it != m_imageList.end();) {
        cv::Mat gray[2];
        gray[0] = cv::imread(m_param.leftImagePath + "/" + *it, 0);
        gray[1] = cv::imread(m_param.rightImagePath + "/" + *it, 0);

        if (gray[0].empty()) {
            throw std::runtime_error("failed to read " + m_param.leftImagePath + *it);
        }
        if (gray[1].empty()) {
            throw std::runtime_error("failed to read " + m_param.rightImagePath + *it);
        }
        if (m_imageSize.area() == 0) {
            m_imageSize = gray[0].size();
        }

        cv::GaussianBlur(gray[0], gray[0], cv::Size(3, 3), 0.);
        cv::GaussianBlur(gray[1], gray[1], cv::Size(3, 3), 0.);

        std::vector<cv::Point2f> corners[2];
        bool found[2];
        found[0] = cv::findChessboardCorners(gray[0], m_patternSize, corners[0]);
        found[1] = cv::findChessboardCorners(gray[1], m_patternSize, corners[1]);

        if (!(found[0] && found[1])) {
            std::cerr << "failed to find corners on image pair: " + *it << std::endl;
            it = m_imageList.erase(it);
            continue;
        }

        cv::cornerSubPix(gray[0], corners[0], subPixWinSize, cv::Size(-1, -1), termimateCrit);
        cv::cornerSubPix(gray[1], corners[1], subPixWinSize, cv::Size(-1, -1), termimateCrit);
        allLeftImagePoints.emplace_back(corners[0]);
        allRightImagePoints.emplace_back(corners[1]);
        ++it;
    }
}

StereoCalibrationHandler::Param StereoCalibrationHandler::getParam(const std::string& jsonPath)
{
    rapidjson::Document jsonDoc = readFromJsonFile(jsonPath);
    StereoCalibrationHandler::Param param;
    param.leftImagePath = getValueAs<std::string>(jsonDoc, "left_image_path");
    param.rightImagePath = getValueAs<std::string>(jsonDoc, "right_image_path");
    param.leftCalibParamsPath = getValueAs<std::string>(jsonDoc, "left_calib_params_path");
    param.rightCalibParamsPath = getValueAs<std::string>(jsonDoc, "right_calib_params_path");

    param.imageListFile = getValueAs<std::string>(jsonDoc, "image_list_file");
    param.numRow = getValueAs<int>(jsonDoc, "num_row");
    param.numCol = getValueAs<int>(jsonDoc, "num_col");
    param.squareSize = getValueAs<float>(jsonDoc, "square_size");

    return param;
}

void StereoCalibrationHandler::getRawIntrinsicParams()
{
    cv::FileStorage fs;

    fs.open(m_param.leftCalibParamsPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("failed to open left calibration params");
    }
    fs["camera_matrix"] >> m_leftCamInfo.K;
    fs["distortion_coefficients"] >> m_leftCamInfo.D;
    fs["image_width"] >> m_leftCamInfo.imageSize.width;
    fs["image_height"] >> m_leftCamInfo.imageSize.height;

    fs.open(m_param.rightCalibParamsPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("failed to open right calibration params");
    }
    fs["camera_matrix"] >> m_rightCamInfo.K;
    fs["distortion_coefficients"] >> m_rightCamInfo.D;
    fs["image_width"] >> m_rightCamInfo.imageSize.width;
    fs["image_height"] >> m_rightCamInfo.imageSize.height;

    m_leftCamInfo.K.convertTo(m_leftCamInfo.K, CV_32FC1);
    m_rightCamInfo.K.convertTo(m_rightCamInfo.K, CV_32FC1);
}

void StereoCalibrationHandler::validate(const StereoCalibrationHandler::Param& param)
{
    if (param.leftImagePath.empty() || param.rightImagePath.empty()) {
        throw std::runtime_error("empty path to images");
    }

    if (param.leftCalibParamsPath.empty() || param.rightCalibParamsPath.empty()) {
        throw std::runtime_error("empty path to calibration params");
    }

    if (param.imageListFile.empty()) {
        throw std::runtime_error("empty file that stores image list");
    }

    if (param.numRow <= 0) {
        throw std::runtime_error("invalid number of rows");
    }

    if (param.numCol <= 0) {
        throw std::runtime_error("invalid number of columns");
    }

    if (param.squareSize <= 0) {
        throw std::runtime_error("invalid square size");
    }
}
}  // namespace _cv
