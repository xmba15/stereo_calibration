/**
 * @file    MonocularCameraIntrinsicCalibration.cpp
 *
 * @author  btran
 *
 */

#include <opencv2/opencv.hpp>

#include "MonocularCameraIntrinsicCalibrationHandler.hpp"

namespace
{
using CalibrationHandler = _cv::MonocularCameraIntrinsicCalibrationHandler;

void saveCameraParams(const std::string& fileName, const cv::Size& imageSize, const cv::Mat& cameraMatrix,
                      const cv::Mat& distortionParams);
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [app] [path/to/json/config] [path/to/output/params]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string CONFIG_PATH = argv[1];
    CalibrationHandler::Param param = CalibrationHandler::getParam(CONFIG_PATH);
    CalibrationHandler calibHandler(param);
    if (calibHandler.allImagePoints().empty()) {
        std::cerr << "failed to get enough image points" << std::endl;
        return EXIT_FAILURE;
    }
    cv::Mat K;
    cv::Mat distortionParams;
    std::vector<cv::Mat> rVecs;
    std::vector<cv::Mat> tVecs;
    calibHandler.run(K, distortionParams, rVecs, tVecs);

    std::string outputCalibFile = argv[2];
    ::saveCameraParams(outputCalibFile, calibHandler.imageSize(), K, distortionParams);

    std::cout << K << "\n";
    std::cout << distortionParams << "\n";

    double reprojectionErr = calibHandler.computeReprojectionErrors(
        calibHandler.objectPoints(), calibHandler.allImagePoints(), K, distortionParams, rVecs, tVecs);

    std::cout << "reprojection error: " << reprojectionErr << "\n";

    cv::Mat optimalK =
        cv::getOptimalNewCameraMatrix(K, distortionParams, calibHandler.imageSize(), 1, calibHandler.imageSize(), 0);
    std::cout << "optimal K: " << optimalK << "\n";

    return 0;
}

namespace
{
void saveCameraParams(const std::string& fileName, const cv::Size& imageSize, const cv::Mat& cameraMatrix,
                      const cv::Mat& distortionParams)
{
    cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distortionParams;
}
}  // namespace
