#ifndef __TESTS_H__
#define __TESTS_H__

#ifdef OPENCV_OLD_INCLUDES
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/highgui/highgui.hpp>
#endif
#include <limits>
#include "lights.h"
#include "gsl.h"

template<typename T>
bool check_bounds_of_value(const T value, const std::string& valuename, const T min = 0, const T max = 1) {
  bool ret_val = value >= min && value <= max;
  if (!ret_val) {
    std::cout << valuename << " " << value << " too ";
    if (value < min)
      std::cout << "small";
    if (value > max)
      std::cout << "big";
    std::cout << std::endl;
  }
  return ret_val;
}

template<typename T, int dim>
bool check_pixel(const cv::Vec<T, dim>& pixel, const std::string& name, const int x, const int y, const T min = 0.0, const T max = 1.0) {
  bool status = true;
  for (unsigned int i = 0; i < dim; i++)
    status &= check_bounds_of_value(pixel[i], name, min, max);
  if (!status) {
    std::cout << "point at: " << x << "/" << y << std::endl;
    std::cout << "point value: ";
    for (unsigned int j = 0; j < dim; j++)
      std::cout << pixel[j] << ", ";
    std::cout << std::endl;
  }
  return status;
}
template<typename X, int dim>
std::pair<X, X> get_min_max(const cv::Mat_<cv::Vec<X, dim> >& mat) {
  cv::Mat_<X> one_dim(cv::Mat(mat).reshape(1));
  std::pair<cv::MatIterator_<X>, cv::MatIterator_<X> > tmp = std::minmax_element(std::begin(one_dim), std::end(one_dim));
  return std::make_pair(*tmp.first, *tmp.second);
}

template<typename X>
std::pair<X, X> get_min_max(const std::vector<X>& mat) {
  auto tmp = std::minmax_element(std::begin(mat), std::end(mat));
  return std::make_pair(*tmp.first, *tmp.second);
}

template<typename X, int dim>
std::pair<X, X> get_min_max_and_print(const cv::Mat_<cv::Vec<X, dim> >& mat) {
  std::pair<X, X> ret_val = get_min_max<X, dim>(mat);
  std::cout << "min/max value of a pixel is " << static_cast<double>(ret_val.first) << " / " << static_cast<double>(ret_val.second) << std::endl;
  return ret_val;
}

template<typename T>
void test_reflect() {
  cv::Mat_<T> vector = (cv::Mat_<T>(2, 1) << -1, 0);
  cv::Mat_<T> normal = (cv::Mat_<T>(2, 1) << 1, 1)/sqrt(2);
  cv::Mat_<T> r = reflect(normal, vector);
  assert(fabs(r(0)) <= std::numeric_limits<T>::epsilon());
  assert(fabs(r(1) - 1) <= std::numeric_limits<T>::epsilon());
}

template<typename T, int dim>
void test_normals(const cv::Mat_<cv::Vec<T, dim> >& normals_, const cv::Vec<T, dim> offset = cv::Vec<T, dim>(), const T eps = std::numeric_limits<float>::epsilon()) {
  cv::Mat_<cv::Vec<T, dim> > normals;
  normals_.copyTo(normals);
  for (auto& normal : normals) {
    if (fabs(cv::norm(normal - offset) - 1) > eps) {
      normal = T();
    }
  }
  cv::imshow("geteste normalen1", normals);
}

template<typename T>
bool is_scalar(const cv::Mat_<T>& mat) {
  const bool dims = mat.dims == 2;
  const bool cols = mat.cols == 1;
  const bool rows = mat.rows == 1;
  return dims && cols && rows;
}

template<typename T, int dim>
void test_modelview_matrix_and_light_positions(const cv::Mat_<GLfloat>& model_view_matrix, const Lights::Lights<T, dim>& lights) {
  std::cout << "modelviewmatrix:\n" << model_view_matrix << std::endl;

  for (const auto& light : lights.lights) {
    auto position_object_space = light.get_position();
    std::cout << "light source: "
      << light.number << ", "
      << position_object_space
      << std::endl;
    auto position_world_space = transform(model_view_matrix, position_object_space);
    std::cout << "light source: "
      << light.number << ", "
      << position_world_space
      << std::endl;
  }
}

template<int colors_per_light, int components_per_light>
bool check_solution(const gsl::vector<colors_per_light, components_per_light> &sol, double min = 0.0, double max = 1.0) {
  bool ret_val = true;
  for (unsigned int i = 0; i < sol.size(); i++) {
    ret_val &= check_bounds_of_value(sol.get(i), "light solution", min, max);
  }
  return ret_val;
}

bool test_gsl_vector_iterator();

#endif /* __TESTS_H__ */