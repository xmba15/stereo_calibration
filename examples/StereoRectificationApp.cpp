/**
 * @file    StereoRectificationApp.cpp
 *
 * @author  btran
 *
 */

#include <opencv2/opencv.hpp>

#include <stereo_rectification/stereo_rectification.hpp>

namespace
{
using StereoRectifier = _cv::StereoCalibrationHandler;

void drawHorizontalLines(cv::Mat& image, const cv::Scalar& color, int step = 20)
{
    int height = image.rows;
    int width = image.cols;

    if (height * width == 0) {
        return;
    }

    int numLines = height / step;
    for (int i = 0; i < numLines; ++i) {
        int y = (i + 1) * step;
        cv::line(image, cv::Point(0, y), cv::Point(width - 1, y), color);
    }
}

const cv::Scalar BLUE(255, 0, 0);
const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar RED(0, 0, 255);
const cv::Scalar PINK(255, 192, 203);
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: [app] [path/to/json/config] [sample/left/image] [sample/right/image]" << std::endl;
        return EXIT_FAILURE;
    }
    std::string CONFIG_PATH = argv[1];
    std::string SAMPLE_LEFT_PATH = argv[2];
    std::string SAMPLE_RIGHT_PATH = argv[3];

    StereoRectifier::Param param = StereoRectifier::getParam(CONFIG_PATH);
    StereoRectifier stereoRectifier(param);

    if (stereoRectifier.allLeftImagePoints().empty()) {
        std::cerr << "failed to get enough image points" << std::endl;
        return EXIT_FAILURE;
    }

    // estimate fundamental matrix

    cv::Mat F = stereoRectifier.findFundamentalMat();
    stereoRectifier.drawEpipolarLines(F);

    double epipolarErr = stereoRectifier.computeEpipolarErr(F);

    std::cout << "epipolar error: " << epipolarErr << "\n";

    // estimate homographies for rectification
    cv::Mat H[2];
    stereoRectifier.rectifyUncalibrated(F, H[0], H[1]);

    // undistort rectified images
    _cv::CameraInfo newCameraInfo;
    cv::Mat R[2];
    stereoRectifier.getNewCameraInfo(H[0], H[1], newCameraInfo, R[0], R[1]);

    cv::Mat rmap[2][2];
    cv::initUndistortRectifyMap(stereoRectifier.leftCamInfo().K, stereoRectifier.leftCamInfo().D, R[0], newCameraInfo.K,
                                stereoRectifier.imageSize(), CV_16SC2, rmap[0][0], rmap[0][1]);

    cv::initUndistortRectifyMap(stereoRectifier.rightCamInfo().K, stereoRectifier.rightCamInfo().D, R[1],
                                newCameraInfo.K, stereoRectifier.imageSize(), CV_16SC2, rmap[1][0], rmap[1][1]);

    // -------------------------------------------------------------------------
    // test estimated calibration parameters on a pair of images
    cv::Mat images[2], hconcatImages[3], vconcatImages[2];
    cv::Mat allConcat;
    images[0] = cv::imread(SAMPLE_LEFT_PATH);
    if (images[0].empty()) {
        std::cerr << "failed to read sample left image" << std::endl;
        return EXIT_FAILURE;
    }
    images[1] = cv::imread(SAMPLE_RIGHT_PATH);
    if (images[1].empty()) {
        std::cerr << "failed to read sample right image" << std::endl;
        return EXIT_FAILURE;
    }

    // draw original distorted, not rectified images
    cv::hconcat(images[0], images[1], hconcatImages[0]);
    ::drawHorizontalLines(hconcatImages[0], BLUE);
    cv::putText(hconcatImages[0], "NOT RECTIFIED/ DISTORTED", cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, RED);

    // draw distorted, rectified images
    cv::Mat warpedImages[2];
    for (int i = 0; i < 2; ++i) {
        cv::warpPerspective(images[i], warpedImages[i], H[i], images[i].size(), cv::INTER_LINEAR);
    }

    cv::hconcat(warpedImages[0], warpedImages[1], hconcatImages[1]);
    ::drawHorizontalLines(hconcatImages[1], GREEN);
    cv::putText(hconcatImages[1], "RECTIFIED/ DISTORTED", cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, RED);

    // draw undistorted, rectified images
    cv::Mat undistortedImages[2];
    for (int i = 0; i < 2; ++i) {
        cv::remap(images[i], undistortedImages[i], rmap[i][0], rmap[i][1], cv::INTER_LINEAR);
    }
    cv::hconcat(undistortedImages[0], undistortedImages[1], hconcatImages[2]);
    ::drawHorizontalLines(hconcatImages[2], PINK);
    cv::putText(hconcatImages[2], "RECTIFIED/ UNDISTORTED", cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, RED);

    cv::vconcat(hconcatImages[0], hconcatImages[1], vconcatImages[0]);
    cv::vconcat(hconcatImages[1], hconcatImages[2], vconcatImages[1]);
    cv::imshow("not_rectified_vs_rectified", vconcatImages[0]);
    cv::imshow("undistorted_rectified", vconcatImages[1]);
    cv::waitKey(0);

    return 0;
}
