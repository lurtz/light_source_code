#ifndef SOLVER_H_
#define SOLVER_H_

#include "lights.h"
#include <cv.hpp>
#include <vector>
#include <gsl/gsl_multifit.h>
#include <iomanip>
#include <limits>

void print_gsl_matrix_row(const gsl_matrix& m, const unsigned int row) {
  for (unsigned int col = 0; col < m.size2 - 1; col++)
    std::cout << std::setw(10) << gsl_matrix_get(&m, row, col) << ", ";
  std::cout << std::setw(10) << gsl_matrix_get(&m, row, m.size2-1);
}

void print_gsl_matrix(const gsl_matrix& m) {
  for (unsigned int i = 0; i < m.size1; i++) {
    print_gsl_matrix_row(m, i);
    std::cout << std::endl;
  }
}

void print_gsl_linear_system(const gsl_matrix& m, const gsl_vector& c, const gsl_vector& y) {
  // print row by row
  std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(4) << std::setfill(' ');
  for (unsigned int row = 0; row < y.size; row++) {
    print_gsl_matrix_row(m, row);
    std::cout << "  ";
    if (row < c.size)
      std::cout << std::setw(10) << gsl_vector_get(&c, row);
    else
      std::cout << "          ";
    std::cout << "  " << std::setw(10) << gsl_vector_get(&y, row) << std::endl;
  }
}

void sample_linear_problem() {
  const unsigned int rows = 6;
  const unsigned int cols = 3;
  gsl_matrix *x = gsl_matrix_alloc(rows, cols);
  gsl_matrix *cov = gsl_matrix_alloc(cols, cols);
  gsl_vector *y = gsl_vector_alloc(rows);
  gsl_vector *c = gsl_vector_alloc(cols);

  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++)
      gsl_matrix_set(x, i, j, (j + 1) * (i + 1));
    gsl_vector_set(y, i, cols * (cols + 1) / 2);
  }

  double chisq;
  gsl_multifit_linear_workspace * problem = gsl_multifit_linear_alloc(rows, cols);
  gsl_multifit_linear(x, y, c, cov, &chisq, problem);
  gsl_multifit_linear_free(problem);

  print_gsl_linear_system(*x, *c, *y);

  gsl_matrix_free(x);
  gsl_matrix_free(cov);
  gsl_vector_free(y);
  gsl_vector_free(c);
}

template<typename T>
bool check_bounds_of_value(const T value, const std::string& valuename, const T min = 0.0f, const T max = 1.0f) {
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

template<typename T, int dim>
bool update_min_max(const cv::Vec<T, dim>& item, T& min, T& max) {
  bool status = false;
  for (unsigned int i = 0; i < dim; i++) {
    if (item[i] < min) {
      status = true;
      min = item[i];
    }
    if (item[i] > max) {
      status = true;
      max = item[i];
    }
  }
  return status;
}

template<typename X, int dim>
std::pair<X, X> get_min_max(const cv::Mat& mat) {
  X min = std::numeric_limits<X>::max();
  X max = std::numeric_limits<X>::lowest();
  for (auto iter = mat.begin<cv::Vec<X, dim> >(); iter != mat.end<cv::Vec<X, dim> >(); iter++) {
    cv::Vec<X, dim> item = *iter;
    update_min_max(item, min, max);
  }
  return std::make_pair(min, max);
}

template<typename X, int dim>
std::pair<X, X> get_min_max_and_print(const cv::Mat& mat) {
  std::pair<X, X> ret_val = get_min_max<X, dim>(mat);
  std::cout << "min/max value of a pixel is " << static_cast<double>(ret_val.first) << " / " << static_cast<double>(ret_val.second) << std::endl;
  return ret_val;
}

template<typename T>
cv::Mat reflect(const cv::Mat& normal, const cv::Mat& vector) {
  assert(normal.cols == vector.cols);
  assert(normal.rows == vector.rows);
  assert(normal.rows == 1 || normal.cols == 1);
  const cv::Mat L_m_N = vector.t() * normal;
  T cos = L_m_N.at<T>(0,0);
  const cv::Mat R_m = vector - 2 * cos * normal;
  return R_m;
}

template<typename T>
void test_reflect() {
  cv::Mat vector = (cv::Mat_<T>(2, 1) << -1, 0);
  cv::Mat normal = (cv::Mat_<T>(2, 1) << 1, 1)/sqrt(2);
  cv::Mat r = reflect<T>(normal, vector);
  assert(fabs(r.at<T>(0)) <= std::numeric_limits<T>::epsilon());
  assert(fabs(r.at<T>(1) - 1) <= std::numeric_limits<T>::epsilon());
}

template<typename T>
void optimize_lights(cv::Mat& original_image, cv::Mat& image, cv::Mat& normals, cv::Mat& position, cv::Mat& model_view_matrix, float clear_color, std::vector<T> &ambient, std::vector<typename Light<T>::properties>& lights, const int alpha = 50) {
//  cv::imshow("FBO texture", image);

  // CV_8UC3  16
  get_min_max_and_print<unsigned char, 3>(original_image);

  int new_channel_count = std::max(original_image.channels(), image.channels());
  original_image.reshape(new_channel_count);
  image.reshape(new_channel_count);
//  cv::imshow("original image right channel count", original_image);

  cv::Mat correct_format_image;
  original_image.convertTo(correct_format_image, CV_32F, 1.0/std::numeric_limits<unsigned char>::max());
  cv::imshow("correct format image", correct_format_image);

//  CV_32FC3  21
  get_min_max_and_print<float, 3>(correct_format_image);

  assert(correct_format_image.type() == image.type());

  cv::Mat diff = image - correct_format_image;
//  cv::imshow("differenz", diff);

//  cv::imshow("normals", normals);
//  cv::imshow("position", position);

  print_lights(lights, ambient);

  // do not take all points of the image
  // TODO calculate this value somehow, maybe specify the number of samples and
  //      distribute them over the mesh in the image

  const unsigned int div = 100;
  const unsigned int colors_per_light = 3;
  const unsigned int rows = correct_format_image.rows * correct_format_image.cols / div * colors_per_light;
  const unsigned int components_per_light = 2;
  const unsigned int cols = (1 + lights.size() * components_per_light) * colors_per_light;
  gsl_matrix *x = gsl_matrix_alloc (rows, cols);
  gsl_matrix *cov = gsl_matrix_alloc(cols, cols);
  gsl_vector *y = gsl_vector_alloc(rows);
  gsl_vector *c = gsl_vector_alloc(cols);

  cv::Mat used_pixels(image.rows, image.cols, CV_8U, cv::Scalar(0));

  for (unsigned int row = 0; row < rows; row+=colors_per_light) {
    // 1. find a good pixel
    int _x = 0;
    int _y = 0;
    while (_x == 0 && _y == 0) {
      const int x = image.cols * drand48();
      const int y = image.rows * drand48();

      // skip if already taken
      if (used_pixels.at<unsigned char>(y, x))
        continue;

      // smallest possible eps is 0.01 for float and opengl (precision)
      const float eps = 0.01;

      // skip if no object
      // assume that a normal is (clear_color,clear_color,clear_color) where no object is
      cv::Vec<float, 3> normal = normals.at<cv::Vec<float, 3> >(y, x);
      if (fabs(normal[0] - clear_color) < eps && fabs(normal[1] - clear_color) < eps && fabs(normal[2] - clear_color) < eps)
        continue;

      _x = x;
      _y = y;
      used_pixels.at<unsigned char>(y,x) = std::numeric_limits<unsigned char>::max();
    }

    // 2. set matrix parameter for pixel
    // set value of pixel in the image to the vector
    assert(_x < correct_format_image.cols);
    assert(_y < correct_format_image.rows);
    const cv::Vec<float, colors_per_light> pixel = correct_format_image.at<cv::Vec<float, colors_per_light> >(_y, _x);
    check_pixel(pixel, "target", _x, _y);
    for (unsigned int i = 0; i < colors_per_light; i++) {
      gsl_vector_set(y, row + i, pixel[i]);
    }
    // set shading parameter for a pixel in the matrix
    // ambient term
    for (unsigned int i = 0; i < colors_per_light; i++)
      for (unsigned int j = 0; j < colors_per_light; j++)
        if (i == j)
          gsl_matrix_set(x, row + i, j, 1);
        else
          gsl_matrix_set(x, row + i, j, 0);

    const cv::Mat pos_vec(position.at<cv::Vec3f>(_y, _x));
    const cv::Mat normal_(normals.at<cv::Vec<float, 3> >(_y, _x), false);
    cv::Mat normal(normal_.rows, normal_.cols, normal_.type());
    cv::normalize(normal_, normal);

    for (unsigned int col = colors_per_light; col < cols; col+=components_per_light*colors_per_light) {
      typename Light<T>::properties& props = lights.at(col/components_per_light/colors_per_light);
      // TODO need to transform light_pos to image_space
      const std::vector<T>& light_pos_in_world_space_vector = props[get_position_name(col/components_per_light/colors_per_light)];
      const cv::Mat light_pos_in_world_space_mat(light_pos_in_world_space_vector, false);
      // TODO fail right here
      //      need position of vertex and light in world space for phong shading
      //      maybe achievable by inverting the projection matrix
      const cv::Mat light_pos(model_view_matrix * light_pos_in_world_space_mat, cv::Range(0, 3));

      const cv::Mat L_m_ = light_pos - pos_vec;
      cv::Mat L_m(L_m_.rows, L_m_.cols, L_m_.type());
      cv::normalize(L_m_, L_m);
      // should be a scalar
      const cv::Mat L_m_N = L_m.t() * normal;
      assert(L_m_N.dims == 2);
      assert(L_m_N.cols == 1);
      assert(L_m_N.rows == 1);
      float diffuse = L_m_N.at<float>(0,0);
      if (diffuse < 0.0f)
        diffuse = 0.0f;
      check_bounds_of_value(diffuse, "diffuse");

      float specular = 0.0f;
      if (diffuse > 0.0f) {
        // R =  I - 2.0 * dot(N, I) * N
        const cv::Mat R_m = reflect<float>(normal, -L_m);
        // should be a scalar
        cv::Mat E (pos_vec.rows, pos_vec.cols, pos_vec.type());
        cv::normalize(-pos_vec, E);
        const cv::Mat R_m_V = R_m.t() * E;
        assert(R_m_V.dims == 2);
        assert(R_m_V.cols == 1);
        assert(R_m_V.rows == 1);
        const float base = R_m_V.at<float>(0,0);
        specular = std::pow(base, alpha);
        check_bounds_of_value(specular, "specular");
      }

      //   global  each light
      //   amb     diff   spec
      //   r g b   r g b  r g b
      // r 1 0 0   1 0 0  1 0 0
      // g 0 1 0   0 1 0  0 1 0
      // b 0 0 1   0 0 1  0 0 1
      for (unsigned int i = 0; i < colors_per_light; i++) {
        for (unsigned int j = 0; j < colors_per_light; j++) {
          if (i == j) {
            // diffuse
            gsl_matrix_set(x, row+i, col+j, diffuse);
            // specular
            gsl_matrix_set(x, row+i, col+colors_per_light+j, specular);
          } else {
            // diffuse
            gsl_matrix_set(x, row+i, col+j, 0);
            // specular
            gsl_matrix_set(x, row+i, col+colors_per_light+j, 0);
          }
        }
      }
    }
  }

  // get solution
  double chisq;
  gsl_multifit_linear_workspace * problem = gsl_multifit_linear_alloc(rows, cols);
  gsl_multifit_linear (x, y, c, cov, &chisq, problem);
  gsl_multifit_linear_free(problem);

  // ambient
  for (unsigned int col = 0; col < colors_per_light; col++) {
    ambient.at(col) = gsl_vector_get(c, col);
  }

  // diffuse and specular
  for (unsigned int col = colors_per_light; col < cols; col+=components_per_light*colors_per_light) {
    typename Light<T>::properties& props = lights.at(col/components_per_light/colors_per_light);
    std::vector<T>& diff = props[get_diffuse_name(col/components_per_light/colors_per_light)];
    std::vector<T>& spec = props[get_specular_name(col/components_per_light/colors_per_light)];
    for (unsigned int i = 0; i < colors_per_light; i++) {
      diff.at(i) = gsl_vector_get(c, col + i);
      spec.at(i) = gsl_vector_get(c, col + colors_per_light + i);
    }
  }

//  print_gsl_linear_system(*x, *c, *y);

  gsl_matrix_free(x);
  gsl_matrix_free(cov);
  gsl_vector_free(y);
  gsl_vector_free(c);

  cv::imshow("used_pixels", used_pixels);
  cv::waitKey(100);

  print_lights(lights, ambient);
}

#endif /* SOLVER_H_ */
