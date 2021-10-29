/**
 * @file    StereoCalibrationHandler.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <string>

#include <opencv2/opencv.hpp>

#include "Types.hpp"

namespace _cv
{
class StereoCalibrationHandler
{
 public:
    struct Param {
        std::string leftImagePath;         // path to the directory that stores images from left camera
        std::string rightImagePath;        // path to the directory that stores images from right camera
        std::string leftCalibParamsPath;   // path to the intrinsic calibration param file of the left camera
        std::string rightCalibParamsPath;  // path to the intrinsic calibration param file of the right camera
        std::string imageListFile;         // file that stores image list
        int numRow;                        // num (inner) row of the checkerboard
        int numCol;                        // num (inner) column of the checkerboard
        float squareSize;
    };

    static Param getParam(const std::string& jsonPath);
    static void validate(const StereoCalibrationHandler::Param& param);

    explicit StereoCalibrationHandler(const Param& param);

    const auto& allLeftImagePoints() const
    {
        return m_allLeftImagePoints;
    }

    const auto& allRightImagePoints() const
    {
        return m_allRightImagePoints;
    }

    const auto& leftCamInfo() const
    {
        return m_leftCamInfo;
    }

    const auto& rightCamInfo() const
    {
        return m_rightCamInfo;
    }

    const auto& imageSize() const
    {
        return m_imageSize;
    }

    cv::Mat findFundamentalMat();

    void drawEpipolarLines(const cv::Mat& F) const;

    double computeEpipolarErr(const cv::Mat& F) const;

    void rectifyUncalibrated(const cv::Mat& F, cv::Mat& Hl, cv::Mat& Hr) const;

    void getNewCameraInfo(const cv::Mat& Hl, const cv::Mat& Hr, CameraInfo& newCameraInfo, cv::Mat& Rl, cv::Mat& Rr,
                          float alpha = 0) const;

 private:
    std::vector<cv::Point3f> generateObjectPoints(int numRow, int numCol, float squareSize) const;
    void getImagePoints(std::vector<std::vector<cv::Point2f>>& allLeftImagePoints,
                        std::vector<std::vector<cv::Point2f>>& allRightImagePoints);
    void getRawIntrinsicParams();

 private:
    Param m_param;

    cv::Size m_patternSize;

    cv::Size m_imageSize;
    std::vector<std::string> m_imageList;
    std::vector<cv::Point3f> m_objectPoints;  // object points for one image
    std::vector<std::vector<cv::Point2f>> m_allLeftImagePoints;
    std::vector<std::vector<cv::Point2f>> m_allRightImagePoints;

    CameraInfo m_leftCamInfo;
    CameraInfo m_rightCamInfo;
};

}  // namespace _cv
