#ifndef __TESTS_H__
#define __TESTS_H__

#include "lights.h"
#include "gsl.h"

#ifdef OPENCV_OLD_INCLUDES
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/highgui/highgui.hpp>
#endif
#include <limits>

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
    auto position_world_space = model_view_matrix * position_object_space;
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

bool test_all();

template<typename T>
std::tuple<unsigned int, unsigned int> vector_to_pixel_coordinate(const cv::Vec3f& pos, const cv::Vec<T, 2>& min, const cv::Vec<T, 2>& max, const unsigned int width, const unsigned int height) {
  auto pos_in_center = cv::Vec<T, 2>(pos[0], pos[1]) - min;
  auto max_in_center = max - min;
  return std::make_tuple(width * (pos_in_center[0] / max_in_center[0]), height * (pos_in_center[1] / max_in_center[1]));
}

template<typename T, int dim>
void show_sky(const cv::Mat_<cv::Vec3f>& position, const cv::Mat_<cv::Vec3f>& normals, const Lights::Lights<T, dim> &outer_lights, const cv::Mat_<T>& model_view_matrix) {
  using namespace output_operators;
  auto lights = outer_lights.lights;
  cv::Vec<T, 2> min_lights{std::numeric_limits< int >::max(), std::numeric_limits< int >::max()};
  cv::Vec<T, 2> max_lights{std::numeric_limits< int >::min(), std::numeric_limits< int >::min()};
  for (const auto& light : lights) {
    const cv::Mat_<T> pos = model_view_matrix * light.template get<Lights::Properties::POSITION>();
    for (unsigned int i = 0; i < 2; i++) {
      if (min_lights[i] > pos(i))
        min_lights[i] = pos(i);
      if (max_lights[i] < pos(i))
        max_lights[i] = pos(i);
    }
  }
  std::cout << "Minimum: " << min_lights << ", maximum: " << max_lights << std::endl;
  
  float aspect_ratio = (max_lights[0] - min_lights[0]) / (max_lights[1] - min_lights[1]);
  const unsigned int width = 640;
  const unsigned int height = width/aspect_ratio;
  cv::Mat_<cv::Vec3b> sky(width, height, cv::Vec3b(0, 0, 0));
  
  for (auto iter_pos = std::begin(position), iter_normal = std::begin(normals); iter_pos != std::end(position) && iter_normal != std::end(normals); iter_pos++, iter_normal++) {
    if (is_sample_point(*iter_normal)) {
      cv::Vec3f pos = *iter_pos;
      unsigned int x;
      unsigned int y;
      std::tie(x, y) = vector_to_pixel_coordinate(pos, min_lights, max_lights, width, height);
      sky(x, y) = cv::Vec3b(std::numeric_limits< char >::max(), 0, 0);
    }
  }
  
  for (const auto& light : lights) {
    const cv::Mat_<T> pos = model_view_matrix * light.template get<Lights::Properties::POSITION>();
    unsigned int x;
    unsigned int y;
    std::tie(x, y) = vector_to_pixel_coordinate(cv::Vec3f(pos(0), pos(1), pos(2)), min_lights, max_lights, width, height);
    sky(x, y) = cv::Vec3b(0, std::numeric_limits< char >::max(), 0);
  }
  
  cv::imshow("scene from view of the sovler", sky);
}

#endif /* __TESTS_H__ */