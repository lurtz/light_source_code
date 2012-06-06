#ifndef SOLVER_H_
#define SOLVER_H_

#include "lights.h"
#include <cv.hpp>
#include <vector>
#include <lpsolve/lp_lib.h>

template<typename T>
void optimize_lights(cv::Mat& original_image, cv::Mat& image, cv::Mat& normals, std::vector<typename Light<T>::properties>& lights) {
  int new_channel_count = std::max(original_image.channels(), image.channels());
  original_image.reshape(new_channel_count);
  image.reshape(new_channel_count);

  cv::Mat correct_format_image;
  original_image.convertTo(correct_format_image, CV_32F);

  cv::Mat diff = image - correct_format_image;
  cv::imshow("differenz", diff);
  cv::waitKey(1000);
}

#endif /* SOLVER_H_ */
