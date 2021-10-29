/**
 * @file    MonocularCameraIntrinsicCalibrationHandler.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
class MonocularCameraIntrinsicCalibrationHandler
{
 public:
    struct Param {
        std::string imagePath;      // path to the directory that stores images
        std::string imageListFile;  // file that stores image list
        int numRow;                 // num (inner) row of the checkerboard
        int numCol;                 // num (inner) column of the checkerboard
        float squareSize;
    };

    static Param getParam(const std::string& jsonPath);
    static void validate(const Param& param);

 public:
    explicit MonocularCameraIntrinsicCalibrationHandler(const Param& param);

    void run(cv::Mat& K, cv::Mat& distortionParams, std::vector<cv::Mat>& rVecs, std::vector<cv::Mat>& tVecs) const;

    const auto& allImagePoints() const
    {
        return m_allImagePoints;
    }

    const auto& objectPoints() const
    {
        return m_objectPoints;
    }

    const auto& imageSize() const
    {
        return m_imageSize;
    }

    double computeReprojectionErrors(const std::vector<cv::Point3f>& objectPoints,
                                     const std::vector<std::vector<cv::Point2f>>& imagePoints,
                                     const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                                     const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs) const;

 private:
    std::vector<cv::Point3f> generateObjectPoints(int numRow, int numCol, float squareSize) const;
    std::vector<std::vector<cv::Point2f>> getImagePoints();

 private:
    Param m_param;
    cv::Size m_patternSize;

    cv::Size m_imageSize;
    std::vector<std::string> m_imageList;
    std::vector<cv::Point3f> m_objectPoints;  // object points for one image
    std::vector<std::vector<cv::Point2f>> m_allImagePoints;
};
}  // namespace _cv
