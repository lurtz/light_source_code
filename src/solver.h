#ifndef SOLVER_H_
#define SOLVER_H_

#include "lights.h"

#ifdef OPENCV_OLD_INCLUDES
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include<opencv2/highgui/highgui.hpp>
#endif

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

template<typename X, int dim>
std::pair<X, X> get_min_max(const cv::Mat_<cv::Vec<X, dim> >& mat) {
  cv::Mat_<X> one_dim(cv::Mat(mat).reshape(1));
  std::pair<cv::MatIterator_<X>, cv::MatIterator_<X> > tmp = std::minmax_element(std::begin(one_dim), std::end(one_dim));
  return std::make_pair(*tmp.first, *tmp.second);
}

template<typename X, int dim>
std::pair<X, X> get_min_max_and_print(const cv::Mat_<cv::Vec<X, dim> >& mat) {
  std::pair<X, X> ret_val = get_min_max<X, dim>(mat);
  std::cout << "min/max value of a pixel is " << static_cast<double>(ret_val.first) << " / " << static_cast<double>(ret_val.second) << std::endl;
  return ret_val;
}

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
void test_reflect() {
  cv::Mat_<T> vector = (cv::Mat_<T>(2, 1) << -1, 0);
  cv::Mat_<T> normal = (cv::Mat_<T>(2, 1) << 1, 1)/sqrt(2);
  cv::Mat_<T> r = reflect(normal, vector);
  assert(fabs(r(0)) <= std::numeric_limits<T>::epsilon());
  assert(fabs(r(1) - 1) <= std::numeric_limits<T>::epsilon());
}

template<typename T, int dim>
void test_normals(const cv::Mat_<cv::Vec<T, dim> >& normals_, const cv::Vec<T, dim> offset = cv::Vec<T, dim>(), const T eps = 0.01f) {
  cv::Mat_<cv::Vec<T, dim> > normals;
  normals_.copyTo(normals);
  for (auto& normal : normals) {
    if (fabs(cv::norm(normal - offset) - 1) > eps) {
      normal = T();
    }
  }
  cv::imshow("geteste normalen1", normals);
}

void test_normals2(const cv::Mat_<cv::Vec3f>& normals_, const int x_ = -1, const int y_ = -1, const float eps = 0.01f) {
  const cv::Vec3f offset(0.0f, 0.0f, 0.0f);
  cv::Mat_<cv::Vec3f> normals;
  normals_.copyTo(normals);
  for (int y = 0; y < normals.rows; y++)
    for (int x = 0; x < normals.cols; x++) {
      cv::Vec3f normal = normals(y, x);
      if (fabs(cv::norm(normal - offset) - 1) > eps) {
        normals(y, x) = cv::Vec3f(0, 0, 0);
      } else if (x != -1 && y != -1 && x == x_ && y == y_) {
        std::cout << "normale ok an stelle " << x << ", " << y << std::endl;
      }
    }
  cv::imshow("geteste normalen2", normals);
}

template<typename T>
bool is_scalar(const cv::Mat_<T>& mat) {
  const bool dims = mat.dims == 2;
  const bool cols = mat.cols == 1;
  const bool rows = mat.rows == 1;
  return dims && cols && rows;
}


// smallest possible eps is 0.01 for float and opengl (precision)
const float eps = 0.01;

template<typename T>
void optimize_lights(cv::Mat_<cv::Vec3f >& image, cv::Mat_<cv::Vec3f>& normals, cv::Mat_<cv::Vec3f>& position, cv::Mat_<GLfloat>& model_view_matrix, float clear_color, std::vector<T> &ambient, std::vector<typename Light<T>::properties>& lights, const int alpha = 50) {
  cv::imshow("target image", image);

//  CV_32FC3  21

  cv::imshow("normals", normals);
//  cv::imshow("position", position);

  print_lights(lights, ambient);

  test_normals(normals);
//  test_normals2(normals);
  get_min_max_and_print(normals);

  // do not take all points of the image
  // calculate this value somehow, maybe specify the number of samples and
  // distribute them over the mesh in the image

  const unsigned int div = 100;
  const unsigned int colors_per_light = 3;
  const unsigned int rows = image.rows * image.cols / div * colors_per_light;
  const unsigned int components_per_light = 2;
  const unsigned int cols = (1 + lights.size() * components_per_light) * colors_per_light;
  gsl_matrix *x = gsl_matrix_alloc (rows, cols);
  gsl_matrix *cov = gsl_matrix_alloc(cols, cols);
  gsl_vector *y = gsl_vector_alloc(rows);
  gsl_vector *c = gsl_vector_alloc(cols);

  cv::Mat_<unsigned char> used_pixels(cv::Mat(image.rows, image.cols, CV_8U, cv::Scalar(0)));

  for (unsigned int row = 0; row < rows; row+=colors_per_light) {
    // 1. find a good pixel
    int _x = 0;
    int _y = 0;
    while (_x == 0 && _y == 0) {
      const int x = image.cols * drand48();
      const int y = image.rows * drand48();

      // skip if already taken
      if (used_pixels(y, x))
        continue;

      // skip if no object
      // assume that a normal is (clear_color,clear_color,clear_color) where no object is
      // TODO normalen nochmal testen
      cv::Vec<float, 3> normal = normals(y, x);
      if (fabs(normal[0] - clear_color) < eps && fabs(normal[1] - clear_color) < eps && fabs(normal[2] - clear_color) < eps)
        continue;

      if (!(fabs(cv::norm(normal) - 1) < eps))
        continue;
//      assert(fabs(cv::norm(normal) - 1) < eps);

      _x = x;
      _y = y;
      used_pixels(y,x) = std::numeric_limits<unsigned char>::max();
    }

    // 2. set matrix parameter for pixel
    // set value of pixel in the image to the vector
    assert(_x < image.cols);
    assert(_y < image.rows);
    const cv::Vec<float, colors_per_light> pixel = image(_y, _x);
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

    const cv::Mat_<float> pos_vec(position(_y, _x));
    const cv::Mat_<float> normal(normals(_y, _x), false);
    assert(fabs(cv::norm(normal) - 1) < eps);

    for (unsigned int col = colors_per_light; col < cols; col+=components_per_light*colors_per_light) {
      typename Light<T>::properties& props = lights.at(col/components_per_light/colors_per_light);
      const std::vector<T>& light_pos_in_world_space_vector = props[get_position_name(col/components_per_light/colors_per_light)];
      const cv::Mat_<float> light_pos_in_world_space_mat(light_pos_in_world_space_vector, false);
      // durch vierte komponente teilen
      const cv::Mat_<float> light_pos_vec4(model_view_matrix * light_pos_in_world_space_mat);
      const cv::Mat_<float> light_pos(light_pos_vec4 / light_pos_vec4(3), cv::Range(0, 3));

      const cv::Mat_<float> L_ = light_pos - pos_vec;
      cv::Mat_<float> L(L_.rows, L_.cols, L_.type());
      cv::normalize(L_, L);
      // should be a scalar
      const cv::Mat_<float> LN = L.t() * normal;
      assert(is_scalar(LN));
      float diffuse = LN(0,0);
      if (diffuse < 0.0f)
        diffuse = 0.0f;
      check_bounds_of_value(diffuse, "diffuse");

      float specular = 0.0f;
      if (diffuse > 0.0f) {
        // R =  I - 2.0 * dot(N, I) * N
        const cv::Mat_<float> R = reflect<float>(normal, -L);
        cv::Mat_<float> E (pos_vec.rows, pos_vec.cols, pos_vec.type());
        cv::normalize(-pos_vec, E);
        // should be a scalar
        const cv::Mat_<float> RE = R.t() * E;
        assert(is_scalar(RE));
        const float base = RE(0,0);
        assert(base <= 1.0f);
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
