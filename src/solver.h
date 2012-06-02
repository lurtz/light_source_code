#ifndef SOLVER_H_
#define SOLVER_H_

#include "lights.h"
#include <cv.hpp>
#include <vector>

template<typename T>
void optimize_lights(const cv::Mat& original_image , const cv::Mat& image, std::vector<typename Light<T>::properties>& lights) {

}

#endif /* SOLVER_H_ */
