cmake_minimum_required(VERSION 3.10)

add_executable(stereo_rectification_app
  ${CMAKE_CURRENT_LIST_DIR}/StereoRectificationApp.cpp
)

target_link_libraries(stereo_rectification_app
  PUBLIC
    ${LIBRARY_NAME}
)

add_executable(mono_cam_intrinsic_calib
  ${CMAKE_CURRENT_LIST_DIR}/MonocularCameraIntrinsicCalibration.cpp
  ${CMAKE_CURRENT_LIST_DIR}/MonocularCameraIntrinsicCalibrationHandler.cpp
)

target_link_libraries(mono_cam_intrinsic_calib
  PUBLIC
    ${LIBRARY_NAME}
)
