#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <iterator>
#include <chrono>
#ifdef OPENCV_OLD_INCLUDES
  #include <cv.h>
#else
  #include <opencv2/core/core.hpp>
#endif

template<typename T>
void print(T t) {
  std::cout << t << std::endl;
}

template<typename T, int dim>
std::vector<T> create_vector_from_array(const T (&array)[dim]) {
  std::vector<T> tmp(array, array + dim);
  return tmp;
}

template<typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec) {
  stream << "(";
  std::copy(std::begin(vec), std::end(vec)-1, std::ostream_iterator<T>(stream, ", "));
  stream << vec.back();
  stream << ")";
  return stream;
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

template<typename T, int D>
T distFromPlane(const cv::Vec<T, D>& x, const cv::Vec<T, D>& normal, const cv::Vec<T, D>& point) {
  return normal.dot(x-point);
}

template<typename T, int D>
T distFromPlane(const std::vector<T>& x, const cv::Vec<T, D>& normal, const cv::Vec<T, D>& point) {
  cv::Mat mat(x);
  return distFromPlane(mat.at<cv::Vec<T,D>>(0), normal, point);
}

template<class Rep, class Period>
std::ostream& operator<<(std::ostream& out, const std::chrono::duration<Rep, Period>& tp) {
  out << std::chrono::duration_cast<std::chrono::seconds>(tp).count() << "s";
  return out;
}

template<typename T>
double sum(const T& v) {
  double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
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

#endif /* __UTILS_H__ */