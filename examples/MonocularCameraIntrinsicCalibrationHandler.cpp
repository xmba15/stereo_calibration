/**
 * @file    MonocularCameraIntrinsicCalibrationHandler.cpp
 *
 * @author  btran
 *
 */

#include "MonocularCameraIntrinsicCalibrationHandler.hpp"

#include <stereo_rectification/stereo_rectification.hpp>

namespace _cv
{
MonocularCameraIntrinsicCalibrationHandler::MonocularCameraIntrinsicCalibrationHandler(const Param& param)
    : m_param(param)
    , m_patternSize(m_param.numCol, m_param.numRow)
    , m_imageSize(0, 0)
{
    validate(param);
    m_imageList = parseMetaDataFile(param.imageListFile);
    m_objectPoints = generateObjectPoints(m_param.numRow, m_param.numCol, m_param.squareSize);
    m_allImagePoints = this->getImagePoints();
}

void MonocularCameraIntrinsicCalibrationHandler::run(cv::Mat& K, cv::Mat& distortionParams, std::vector<cv::Mat>& rVecs,
                                                     std::vector<cv::Mat>& tVecs) const
{
    int flag = 0;
#if (CV_VERSION_MAJOR >= 4)
    flag |= cv::CALIB_FIX_K4;
    flag |= cv::CALIB_FIX_K5;
#else
    flag |= CV_CALIB_FIX_K4;
    flag |= CV_CALIB_FIX_K5;
#endif

    std::vector<std::vector<cv::Point3f>> allObjectPoints(m_allImagePoints.size(), m_objectPoints);
    cv::calibrateCamera(allObjectPoints, m_allImagePoints, m_imageSize, K, distortionParams, rVecs, tVecs, flag);
}

double MonocularCameraIntrinsicCalibrationHandler::computeReprojectionErrors(
    const std::vector<cv::Point3f>& objectPoints, const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const std::vector<cv::Mat>& rvecs,
    const std::vector<cv::Mat>& tvecs) const
{
    std::vector<cv::Point2f> reprojectedImagePoints;
    int totalPoints = imagePoints.size() * objectPoints.size();
    double totalErr = 0;
    double err;

    for (std::size_t i = 0; i < imagePoints.size(); ++i) {
        cv::projectPoints(cv::Mat(objectPoints), rvecs[i], tvecs[i], cameraMatrix, distCoeffs, reprojectedImagePoints);
#if (CV_VERSION_MAJOR >= 4)
        err = cv::norm(cv::Mat(imagePoints[i]).clone(), cv::Mat(reprojectedImagePoints).clone(), cv::NORM_L2);
#else
        err = cv::norm(cv::Mat(imagePoints[i]), cv::Mat(reprojectedImagePoints), CV_L2);
#endif
        totalErr += err * err;
    }
    return std::sqrt(totalErr / totalPoints);
}

std::vector<cv::Point3f> MonocularCameraIntrinsicCalibrationHandler::generateObjectPoints(int numRow, int numCol,
                                                                                          float squareSize) const
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

std::vector<std::vector<cv::Point2f>> MonocularCameraIntrinsicCalibrationHandler::getImagePoints()
{
    std::vector<std::vector<cv::Point2f>> imagePoints;
    imagePoints.reserve(m_imageList.size());

    const cv::Size subPixWinSize(5, 5);
    const cv::TermCriteria termimateCrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    for (auto it = m_imageList.begin(); it != m_imageList.end();) {
        cv::Mat gray = cv::imread(m_param.imagePath + "/" + *it, 0);

        if (m_imageSize.area() == 0) {
            m_imageSize = gray.size();
        }
        if (gray.empty()) {
            throw std::runtime_error("failed to read " + *it);
        }
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, m_patternSize, corners);
        if (!found) {
            std::cerr << "failed to find corners on image " + *it << std::endl;
            it = m_imageList.erase(it);
            continue;
        }

        cv::cornerSubPix(gray, corners, subPixWinSize, cv::Size(-1, -1), termimateCrit);
        imagePoints.emplace_back(corners);
        ++it;
    }

    return imagePoints;
}

MonocularCameraIntrinsicCalibrationHandler::Param
MonocularCameraIntrinsicCalibrationHandler::getParam(const std::string& jsonPath)
{
    rapidjson::Document jsonDoc = readFromJsonFile(jsonPath);
    MonocularCameraIntrinsicCalibrationHandler::Param param;
    param.imagePath = getValueAs<std::string>(jsonDoc, "image_path");
    param.imageListFile = getValueAs<std::string>(jsonDoc, "image_list_file");
    param.numRow = getValueAs<int>(jsonDoc, "num_row");
    param.numCol = getValueAs<int>(jsonDoc, "num_col");
    param.squareSize = getValueAs<float>(jsonDoc, "square_size");

    return param;
}

void MonocularCameraIntrinsicCalibrationHandler::validate(
    const MonocularCameraIntrinsicCalibrationHandler::Param& param)
{
    if (param.imagePath.empty()) {
        throw std::runtime_error("empty path to images");
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
