#ifndef SOLVER_H_
#define SOLVER_H_

#include "lights.h"
#include <cv.hpp>
#include <vector>
#include <lpsolve/lp_lib.h>

template<typename T>
void optimize_lights(cv::Mat original_image , cv::Mat image, std::vector<typename Light<T>::properties>& lights) {
  int original_depth = original_image.depth();
  int image_depth = image.depth();
  int new_channel_count = std::max(original_image.channels(), image.channels());
  original_image.reshape(new_channel_count);
  image.reshape(new_channel_count);

  cv::Mat correct_format_image;
  image.convertTo(correct_format_image, CV_8U);
//  image.assignTo(correct_format_image, CV_8U);

  /*
  for (int x = 0; x < image.cols; x++) {
    for (int y = 0; y < image.rows; y++) {
      std::cout << x << ", " << y << std::endl;
      float tmp_val = image.at<float>(y, x);
      correct_format_image.at<unsigned char>(y, x) = static_cast<unsigned char>(tmp_val);
    }
  }
  */

  cv::Mat diff = original_image - correct_format_image;
  cv::imshow("differenz", diff);
  cv::waitKey(0);
}

#endif /* SOLVER_H_ */
