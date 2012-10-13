#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <iterator>
#include <limits>
#include <chrono>
#include <array>
#ifdef OPENCV_OLD_INCLUDES
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include<opencv2/highgui/highgui.hpp>
  #include <opencv2/imgproc/imgproc.hpp>
#endif

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, char delim);

namespace output_operators {

template<typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec) {
  stream << "(";
  std::copy(std::begin(vec), std::end(vec)-1, std::ostream_iterator<T>(stream, ", "));
  stream << vec.back();
  stream << ")";
  return stream;
}

template<typename T, int dim>
std::ostream& operator<<(std::ostream& out, const cv::Vec<T, dim>& vec) {
  out << "cv::Vec<" << typeid(T).name() << ", " << dim << ">(";
  for (unsigned int i = 0; i < dim-1; i++)
    out << vec[i] << ", ";
  out << vec[dim-1] << ")";
  return out;
}

template<typename K, typename V>
std::ostream& operator<<(std::ostream& out, const std::pair<K,V>& pair) {
  out << pair.first << ", " << pair.second;
  return out;
}

template<typename K, typename V>
std::ostream& operator<<(std::ostream& out, const std::map<K,V>& map) {
  for (const auto& iter : map)
    out << iter << std::endl;
  return out;
}

template<class Rep, class Period>
std::ostream& operator<<(std::ostream& out, const std::chrono::duration<Rep, Period>& tp) {
  out << std::chrono::duration_cast<std::chrono::seconds>(tp).count() << "s";
  return out;
}

}

template<typename T>
void print(T t) {
  using namespace output_operators;
  std::cout << t << std::endl;
}

template<typename T, std::size_t dim>
cv::Vec<T, dim> create_vector_from_array(const std::array<T, dim> &array) {
  cv::Vec<T, dim> ret_val;
  for (unsigned int i = 0; i < dim; i++)
    ret_val[i] = array[i];
  return ret_val;
}

template<typename T, int D>
T distFromPlane(const cv::Vec<T, D>& x, const cv::Vec<T, D>& normal, const cv::Vec<T, D>& point) {
  return normal.dot(x-point);
}

template<typename T>
double sum(const T& v) {
  double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
  return sum;
}

// TODO how to tell std::accumulate to use fabs() ?
template<typename T>
double abssum(const T& v) {
  double sum = 0;
  for (auto val : v)
    sum += fabs(val);
  return sum;
}

template<typename T, int dim>
double sum(const cv::Vec<T, dim>& v) {
  double sum = 0;
  for (int i = 0; i < dim; i++)
    sum += v[i];
  return sum;
}

template<typename T, int dim, typename T1>
bool has_length(const cv::Vec<T, dim>& vec, T1 length, T eps = std::numeric_limits<T>::epsilon()) {
  return std::fabs(cv::norm(vec) - length) < eps;
}

template<class RandomAccessIterator>
void flipImage(RandomAccessIterator first_row, RandomAccessIterator past_last_row, const unsigned int width) {
  for (; first_row < past_last_row; first_row+=width, past_last_row-=width) {
    std::swap_ranges(first_row, first_row + width, past_last_row - width);
  }
}

// this nice friend works with std::vector<> as well as for cv::Mat_<>
template<class T>
void flipImage(T& image, const unsigned int width) {
  flipImage(std::begin(image), std::end(image), width);
}

// TODO better tests is this behaves like standard c++11 PRNG
typedef struct halton_sequence {
  const unsigned int m_base;
  float m_number;
  halton_sequence(const unsigned int base = 2, const unsigned int number = 1);
  float operator()();
  void discard(unsigned long long z);
  void seed(const unsigned int i = 1);
  static float min();
  static float max();
} halton_sequence;

template<typename T>
cv::Mat_<T> reflect(const cv::Mat_<T>& normal, const cv::Mat_<T>& vector) {
  assert(normal.cols == vector.cols);
  assert(normal.rows == vector.rows);
  assert(normal.rows == 1 || normal.cols == 1);
  const cv::Mat_<T> L_m_N = vector.t() * normal;
  const T cos = L_m_N(0,0);
  const cv::Mat_<T> R_m = vector - 2 * cos * normal;
  return R_m;
}

template<typename T>
void show_rgb_image(std::string name, cv::Mat_<cv::Vec<T, 3>> image) {
  decltype(image) bla;
  cv::cvtColor(image, bla, CV_RGB2BGR);
  cv::imshow(name, bla);
}

// TODO return type may be cv::Vec<T, dim>
template<typename T, typename T1, int dim>
cv::Mat_<T> operator*(const cv::Mat_<T>& model_view_matrix, const cv::Vec<T1, dim>& light_pos_in_object_space_vector) {
  const cv::Mat_<T1> light_pos_in_object_space_mat(light_pos_in_object_space_vector, false);
  // durch vierte komponente teilen
  const cv::Mat_<T> light_pos_vec4(model_view_matrix * light_pos_in_object_space_mat);
  const cv::Mat_<T> light_pos(light_pos_vec4 / light_pos_vec4(3), cv::Range(0, 3));
  
  return light_pos;
}

template<typename T, typename T1>
void mark_if(cv::Mat_<T>& mat, const T1 row, const T1 col, const T val) {
  if (row >= 0 && row < mat.rows && col >= 0 && col < mat.cols)
    mat(row, col) = val;
}

template<typename T, typename T1>
void mark(cv::Mat_<T>& mat, cv::Vec<T1, 2> pos) {
  for (int i = -1; i < 2; i++)
    for (int j = -1; j < 2; j++) {
      mark_if(mat, pos[0]+i, pos[1]+j, 0.0);
    }
  mark_if(mat, pos[0], pos[1], 1.0);
}

#endif /* __UTILS_H__ */