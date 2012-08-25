#ifndef SOLVER_H_
#define SOLVER_H_

#include "lights.h"

#ifdef OPENCV_OLD_INCLUDES
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include<opencv2/highgui/highgui.hpp>
  #include <opencv2/imgproc/imgproc.hpp>
#endif

#include <vector>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <tuple>

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

template<typename T>
void show_rgb_image(std::string name, cv::Mat_<cv::Vec<T, 3>> image) {
  decltype(image) bla;
  cv::cvtColor(image, bla, CV_RGB2BGR);
  cv::imshow(name, bla);
}

template<typename T>
cv::Mat_<float> transform(const cv::Mat_<GLfloat>& model_view_matrix, const std::vector<T>& light_pos_in_object_space_vector) {
  const cv::Mat_<float> light_pos_in_object_space_mat(light_pos_in_object_space_vector, false);
  // durch vierte komponente teilen
  const cv::Mat_<float> light_pos_vec4(model_view_matrix * light_pos_in_object_space_mat);
  const cv::Mat_<float> light_pos(light_pos_vec4 / light_pos_vec4(3), cv::Range(0, 3));
  
  return light_pos;
}

template<typename T>
void test_modelview_matrix_and_light_positions(const cv::Mat_<GLfloat>& model_view_matrix, const Lights<T>& lights) {
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

std::tuple<unsigned int, unsigned int> get_sample_point(const cv::Mat_<cv::Vec3f>& normals, cv::Mat_<unsigned char>& used_pixels, const float clear_color) {
  int _x = 0;
  int _y = 0;
  while (_x == 0 && _y == 0) {
    const int x = normals.cols * drand48();
    const int y = normals.rows * drand48();

    // skip if already taken
    if (used_pixels(y, x))
      continue;

    const float eps = std::numeric_limits<float>::epsilon();
    
    // skip if no object
    // assume that a normal is (clear_color,clear_color,clear_color) where no object is
    cv::Vec<float, 3> normal = normals(y, x);
    if (fabs(normal[0] - clear_color) < eps && fabs(normal[1] - clear_color) < eps && fabs(normal[2] - clear_color) < eps)
      continue;

    // skip if length is not 1
    if (!(fabs(cv::norm(normal) - 1) < eps))
      continue;
    assert(fabs(cv::norm(normal) - 1) < eps);
    assert(_x < normals.cols);
    assert(_y < normals.rows);

    _x = x;
    _y = y;
    used_pixels(y,x) = std::numeric_limits<unsigned char>::max();
  }
  return std::make_tuple(_x, _y);
}

template<typename T>
std::tuple<float, float> get_diffuse_specular(const cv::Mat_<float> &pos_vec, const cv::Mat_<float> &normal, const Light<T> &light, const cv::Mat_<GLfloat>& model_view_matrix, const int alpha) {
  const cv::Mat_<float> light_pos = transform(model_view_matrix, light.get_position());

  const cv::Mat_<float> L_ = light_pos - pos_vec;
  cv::Mat_<float> L(L_.rows, L_.cols, L_.type());
  cv::normalize(L_, L);
  // should be a scalar
  const cv::Mat_<float> LN = L.t() * normal;
  assert(is_scalar(LN));
  float diffuse = LN(0,0);
  if (diffuse < 0.0f)
    diffuse = 0.0f;
  assert(check_bounds_of_value(diffuse, "diffuse"));

  float specular = 0.0f;
  if (diffuse > 0.0f) {
    // R =  I - 2.0 * dot(N, I) * N
    const cv::Mat_<float> R = reflect<float>(normal, -L);
    cv::Mat_<float> E (pos_vec.rows, pos_vec.cols, pos_vec.type());
    cv::normalize(-pos_vec, E);
    // should be a scalar
    const cv::Mat_<float> RE = R.t() * E;
    assert(is_scalar(RE));
    assert(RE(0,0) <= 1.0f + std::numeric_limits< float >::epsilon());
    const float base = RE(0,0) > 1.0f ? 1.0f : RE(0,0);
    specular = std::pow(base, alpha);
    assert(check_bounds_of_value(specular, "specular"));
  }
  
  return std::make_tuple(diffuse, specular);
}

namespace gsl {
  // TODO overload operators
  // TODO iterators
  
  template<int colors_per_light, int components_per_light>
  class matrix {
    std::unique_ptr<gsl_matrix, void (*)(gsl_matrix *)> m;

    public:
    matrix() : m(0, gsl_matrix_free) {
    }
    
    matrix(const unsigned int rows, const unsigned int cols) : m(gsl_matrix_alloc(rows, cols), gsl_matrix_free) {
      if (!m)
        throw;
    }

    gsl_matrix * get() {
      return m.get();
    }

    const gsl_matrix * get() const {
      return m.get();
    }
    
    void set(const size_t i, const size_t j, const double x) {
      gsl_matrix_set(m.get(), i, j, x);
    }

    size_t get_rows() const {
      return m.get()->size1;
    }

    size_t get_cols() const {
      return m.get()->size2;
    }

    //   global  each light
    //   amb     diff   spec
    //   r g b   r g b  r g b
    // r 1 0 0   1 0 0  1 0 0
    // g 0 1 0   0 1 0  0 1 0
    // b 0 0 1   0 0 1  0 0 1
    void set_ambient(const unsigned int row) {
      for (unsigned int i = 0; i < colors_per_light; i++)
        for (unsigned int j = 0; j < colors_per_light; j++)
          if (i == j)
            set(colors_per_light * row + i, j, 1);
          else
            set(colors_per_light * row + i, j, 0);
    }

    void set_diffuse_specular(const unsigned int row, const unsigned int col, const std::tuple<float, float>& vals) {
      float diffuse, specular;
      std::tie(diffuse, specular) = vals;
      for (unsigned int i = 0; i < colors_per_light; i++) {
        for (unsigned int j = 0; j < colors_per_light; j++) {
          if (i == j) {
            // diffuse
            set(colors_per_light * row+i, colors_per_light + colors_per_light * components_per_light * col+j, diffuse);
            // specular
            set(colors_per_light * row+i, colors_per_light + colors_per_light * components_per_light * col+colors_per_light+j, specular);
          } else {
            // diffuse
            set(colors_per_light * row+i, colors_per_light + colors_per_light * components_per_light * col+j, 0);
            // specular
            set(colors_per_light * row+i, colors_per_light + colors_per_light * components_per_light * col+colors_per_light+j, 0);
          }
        }
      }
    }
  };

  template<int colors_per_light, int components_per_light>
  class vector {
    enum {AMBIENT = 0, DIFFUSE, SPECULAR};
    
    std::unique_ptr<gsl_vector, void (*)(gsl_vector *)> v;

    public:
    vector() : v(0, gsl_vector_free) {}
    
    vector(const unsigned int rows, double default_value = 0.0f) : v(gsl_vector_alloc(rows), gsl_vector_free) {
      if (!v)
        throw;
      set_all(default_value);
    }
    
    vector(const vector& rhs) : v(nullptr, gsl_vector_free) {
      *this = rhs;
    }
    
    vector(gsl_vector const * const rhs) : v(gsl_vector_alloc(rhs->size), gsl_vector_free) {
      if (gsl_vector_memcpy(v.get(), rhs))
        throw;
    }
    
    vector& operator=(const vector& rhs) {
      if (this != &rhs) {
        v = std::unique_ptr<gsl_vector, void (*)(gsl_vector *)>(gsl_vector_alloc(rhs.v->size), gsl_vector_free);
        if (!v)
          throw;
	if (gsl_vector_memcpy(v.get(), rhs.v.get()))
	  throw;
      }
      return *this;
    };
    
    vector(vector&& rhs) : v(0, gsl_vector_free) {
      *this = std::move(rhs);
    }
    
    vector& operator=(vector&& rhs) {
      std::cout << "operator=(vector&&)" << std::endl;
      if (this != &rhs)
        v = std::move(rhs.v);
      return *this;
    }
    
    void set_raw(const size_t i, const double x) {
      gsl_vector_set(v.get(), i, x);
    }

    gsl_vector * get() {
      return v.get();
    }

    const gsl_vector * get() const {
      return v.get();
    }
    
    double get(const size_t i) const {
      return gsl_vector_get(v.get(), i);
    }

    size_t size() const {
      return v.get()->size;
    }
    
    void set_all(double x) {
      gsl_vector_set_all(v.get(), x);
    }
    
    template<typename T>
    std::vector<T> get(const size_t i, const int offset) const {
      std::vector<T> tmp(colors_per_light+1);
      for (unsigned j = 0; j < colors_per_light; j++)
        tmp.at(j) = get(offset*colors_per_light + i*colors_per_light*components_per_light + j);
      tmp.at(colors_per_light) = 0;
      return tmp;
    }
    
    template<typename T>
    std::vector<T> get_ambient() const {
      return get<T>(0, AMBIENT);
    }
    
    template<typename T>
    std::vector<T> get_diffuse(const size_t i) const {
      return get<T>(i, DIFFUSE);
    }
    
    template<typename T>
    std::vector<T> get_specular(const size_t i) const {
      return get<T>(i, SPECULAR);
    }
    
    void set(const unsigned int row, const cv::Vec<float, colors_per_light>& pixel) {
      for (unsigned int i = 0; i < colors_per_light; i++) {
        set_raw(colors_per_light*row + i, pixel[i]);
      }
    }
    
  };

  typedef struct workspace {
    std::unique_ptr<gsl_multifit_linear_workspace, void (*)(gsl_multifit_linear_workspace*)> w;
    workspace(size_t rows, size_t cols) : w(gsl_multifit_linear_alloc(rows, cols), gsl_multifit_linear_free) {
      if (!w)
        throw;
    }
    template<int colors_per_light, int components_per_light>
    void solve(const matrix<colors_per_light, components_per_light>& x, const vector<colors_per_light, components_per_light>& y, vector<colors_per_light, components_per_light>& c, matrix<colors_per_light, components_per_light> &cov, double &chisq) {
      if (gsl_multifit_linear(x.get(), y.get(), c.get(), cov.get(), &chisq, w.get()))
	throw;
    }
  } workspace;
}

template<typename T, unsigned int colors_per_light, unsigned int components_per_light, unsigned int div>
std::tuple<gsl::matrix<colors_per_light, components_per_light>, gsl::vector<colors_per_light, components_per_light>> create_linear_system(const cv::Mat_<cv::Vec3f >& image, const cv::Mat_<cv::Vec3f>& normals, const cv::Mat_<cv::Vec3f>& position, const cv::Mat_<GLfloat>& model_view_matrix, const float clear_color, const Lights<T>& lights, const int alpha = 50) {
  const unsigned int rows = image.rows * image.cols / div * colors_per_light;
  const unsigned int cols = (1 + lights.lights.size() * components_per_light) * colors_per_light;

  gsl::matrix<colors_per_light, components_per_light> x(rows, cols);
  gsl::vector<colors_per_light, components_per_light> y(rows);

  // do not take all points of the image
  // calculate this value somehow, maybe specify the number of samples and
  // distribute them over the mesh in the image
  cv::Mat_<unsigned char> used_pixels(cv::Mat(image.rows, image.cols, CV_8U, cv::Scalar(0)));

  for (unsigned int row = 0; row < rows/colors_per_light; row++) {
    // 1. find a good pixel
    int _x = 0;
    int _y = 0;
    std::tie(_x, _y) = get_sample_point(normals, used_pixels, clear_color);

    // 2. set matrix parameter for pixel
    // set value of pixel in the image to the vector
    const cv::Vec<float, colors_per_light>& pixel = image(_y, _x);
    assert(check_pixel(pixel, "target", _x, _y));
    y.set(row, pixel);
    // set shading parameter for a pixel in the matrix
    // ambient term

    x.set_ambient(row);

    const cv::Mat_<float> pos_vec(position(_y, _x));
    const cv::Mat_<float> normal(normals(_y, _x), false);

    for (unsigned int col = 0; col < lights.lights.size(); col++) {
      const Light<T>& light = lights.lights.at(col);
      x.set_diffuse_specular(row, col, get_diffuse_specular(pos_vec, normal, light, model_view_matrix, alpha));
    }
  }

  return std::make_tuple(std::move(x), std::move(y));
}

template<typename T, int colors_per_light, int components_per_light>
void set_solution(const gsl::vector<colors_per_light, components_per_light>& c, Lights<T>& lights) {
  // ambient
  lights.ambient = c.template get_ambient<T>();

  // diffuse and specular
  for (unsigned int i = 0; i < lights.lights.size(); i++) {
    Light<T>& light = lights.lights.at(i);
    std::vector<T>& diff = light.get_diffuse();
    std::vector<T>& spec = light.get_specular();
    diff = c.template get_diffuse<T>(i);
    spec = c.template get_specular<T>(i);
  }
}

template<typename T>
void optimize_lights(const cv::Mat_<cv::Vec3f >& image, const cv::Mat_<cv::Vec3f>& normals, const cv::Mat_<cv::Vec3f>& position, const cv::Mat_<GLfloat>& model_view_matrix, const float clear_color, Lights<T>& lights, const int alpha = 50) {
  show_rgb_image("target image", image);
//  cv::imshow("normals", normals);
//  cv::imshow("position", position);
  
  //  order of images is: xyz, RGB
  
//  test_modelview_matrix_and_light_positions<T>(model_view_matrix, lights);

  print(lights);

//  test_normals(normals);
  //get_min_max_and_print(normals);

  // do not take all points of the image
  // calculate this value somehow, maybe specify the number of samples and
  // distribute them over the mesh in the image
  
  const unsigned int div = 100;
  const unsigned int colors_per_light = 3;
  const unsigned int components_per_light = 2;

  gsl::matrix<colors_per_light, components_per_light> x;
  gsl::vector<colors_per_light, components_per_light> y;

  std::tie(x, y) = create_linear_system<T, colors_per_light, components_per_light, div>(image, normals, position, model_view_matrix, clear_color, lights, alpha);

  // get solution
  
  gsl::matrix<colors_per_light, components_per_light> cov(x.get_cols(), x.get_cols());
  gsl::vector<colors_per_light, components_per_light> c(x.get_cols());
  double chisq;
  gsl::workspace problem(x.get_rows(), x.get_cols());
  problem.solve(x, y, c, cov, chisq);

  set_solution<float>(c, lights);

  cv::waitKey(100);

  print(lights);
}

namespace gsl {
  template<int colors_per_light, int components_per_light>
  class minimizer {
    std::unique_ptr<gsl_multimin_fminimizer, void (*)(gsl_multimin_fminimizer*)> problem;
    gsl::vector<colors_per_light, components_per_light> start_point;
    gsl::vector<colors_per_light, components_per_light> step_size;
    
    public:
      
    minimizer(const gsl_multimin_fminimizer_type * T, gsl_multimin_function& f) 
    : problem(gsl_multimin_fminimizer_alloc(T, f.n), gsl_multimin_fminimizer_free) {
      gsl::vector<colors_per_light, components_per_light> start_point_(f.n, 0.0);
      gsl::vector<colors_per_light, components_per_light> step_size_(f.n, 1.0);
      if (!problem)
        throw;
      set(f, std::move(start_point_), std::move(step_size_));
    }
    
    minimizer(const gsl_multimin_fminimizer_type * T, gsl_multimin_function& f, const gsl::vector<colors_per_light, components_per_light> start_point, const gsl::vector<colors_per_light, components_per_light> step_size)
    : problem(gsl_multimin_fminimizer_alloc(T, f.n), gsl_multimin_fminimizer_free) {
      if (!problem)
        throw;
      set(f, std::move(start_point), std::move(step_size));
    }
    
    gsl::vector<colors_per_light, components_per_light> get_solution() const {
      return gsl::vector<colors_per_light, components_per_light>(problem.get()->x);
    }
    
    double get_function_value() const {
      return problem.get()->fval;
    }
    
    void set(gsl_multimin_function& f, const gsl::vector<colors_per_light, components_per_light> x, const gsl::vector<colors_per_light, components_per_light> ss) {
      start_point = std::move(x);
      step_size = std::move(ss);
      if (gsl_multimin_fminimizer_set(problem.get(), &f, start_point.get(), step_size.get()))
        throw;
    }
    
    int iterate() {
      if (gsl_multimin_fminimizer_iterate(problem.get()))
        throw;
      double size = gsl_multimin_fminimizer_size (problem.get());
      int status = gsl_multimin_test_size (size, 1e-2);
      return status;
    }
    
  };
  
  template<int colors_per_light, int components_per_light>
  void matrix_vector_mult(double alpha, const matrix<colors_per_light, components_per_light>& A, const gsl_vector * X, double beta, vector<colors_per_light, components_per_light>& Y) {
    if(gsl_blas_dgemv(CblasNoTrans, alpha, A.get(), X, beta, Y.get()))
      throw;
  }
}

template<int colors_per_light, int components_per_light>
double cost(const gsl_vector *v, void *params) {
  if (v == nullptr) {
    std::cerr << "cost(const gsl_vector*, void*): const gsl_vector* parameter is nullptr" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (params == nullptr) {
    std::cerr << "cost(const gsl_vector*, void*): void* parameter is nullptr" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto tuple = static_cast<std::tuple<gsl::matrix<colors_per_light, components_per_light>, gsl::vector<colors_per_light, components_per_light>> *>(params);
  const gsl::matrix<colors_per_light, components_per_light>& x = std::get<0>(*tuple);
  const gsl::vector<colors_per_light, components_per_light>& y = std::get<1>(*tuple);

  gsl::vector<colors_per_light, components_per_light> y_copy = y;
  gsl::matrix_vector_mult(1, x, v, -1, y_copy);
  double cost = 0;
  for (unsigned int i = 0; i < y_copy.size(); i++) {
    cost += fabs(y_copy.get(i));
  }
  for (unsigned int i = 0; i < v->size; i++) {
    double val = gsl_vector_get(v, i);
    if (val < 0.0) {
      cost += std::exp2(std::ceil(-val));
    }
    if (val > 1.0) {
      cost += std::exp2(std::ceil(val-1.0));
    }
  }
  return cost;
}

template<typename T>
void optimize_lights_multi_dim_fit(const cv::Mat_<cv::Vec3f >& image, const cv::Mat_<cv::Vec3f>& normals, const cv::Mat_<cv::Vec3f>& position, const cv::Mat_<GLfloat>& model_view_matrix, const float clear_color, Lights<T>& lights, const int alpha = 50) {
  show_rgb_image("target image", image);
  const unsigned int div = 100;
  const unsigned int colors_per_light = 3;
  const unsigned int components_per_light = 2;
  
  auto linear_system = create_linear_system<T, colors_per_light, components_per_light, div>(image, normals, position, model_view_matrix, clear_color, lights, alpha);

  gsl_multimin_function f{&cost<colors_per_light, components_per_light>, std::get<0>(linear_system).get_cols(), &linear_system};
  gsl::minimizer<colors_per_light, components_per_light> minimizer(gsl_multimin_fminimizer_nmsimplex2, f);
//  gsl::vector<colors_per_light, components_per_light> v(f.n);
//  gsl::vector<colors_per_light, components_per_light> ss(f.n, 1.0);
  
  int status = GSL_CONTINUE;
  for (size_t iter = 0; iter < 10000 && status == GSL_CONTINUE; iter++) {
    status = minimizer.iterate();
    std::cout << "step " << iter << ", " << minimizer.get_function_value() << std::endl;
  }
  
  set_solution<float>(minimizer.get_solution(), lights);
  print(lights);
  
  cv::waitKey(100);
  
}

#endif /* SOLVER_H_ */
