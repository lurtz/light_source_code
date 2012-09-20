#ifndef SOLVER_H_
#define SOLVER_H_

#include "lights.h"
extern "C" {
#include "libtpc/include/nnls.h"
}

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

template<typename T, int dim>
bool is_sample_point(const cv::Vec<T, dim>& normal, const T clear_color, const T eps = std::numeric_limits<T>::epsilon()) {
  bool ret_val = true;
  ret_val &= !(fabs(normal[0] - clear_color) < eps && fabs(normal[1] - clear_color) < eps && fabs(normal[2] - clear_color) < eps);
  // skip if length is not 1
  ret_val &= fabs(cv::norm(normal) - 1) < eps;
  return ret_val;
}

template<typename T, int dim>
unsigned int get_maximum_number_of_sample_points(const cv::Mat_<cv::Vec<T, dim>>& normals, const T clear_color) {
  unsigned int sum = 0;
  for (const auto& normal: normals)
    sum += is_sample_point(normal, clear_color);
  return sum;
}

template<typename T, int dim>
struct sample_point_deterministic {
  typename cv::Mat_<cv::Vec<T, dim>>::const_iterator pos;
  const typename cv::Mat_<cv::Vec<T, dim>>::const_iterator end;
  const T clear_color;
  const double step_size;
  double points_to_skip;
  sample_point_deterministic(const cv::Mat_<cv::Vec<T, dim>>& normals, const unsigned int points_to_deliver, const T clear_color) : pos(std::begin(normals)), end(std::end(normals)), clear_color(clear_color), step_size(static_cast<double>(get_maximum_number_of_sample_points(normals, clear_color))/points_to_deliver), points_to_skip(0) {
    assert(step_size >= 1.0);
  }
  std::tuple<unsigned int, unsigned int> operator()() {
    bool loop = true;
    while (loop) {
      loop = false;
      while (!is_sample_point(*pos), clear_color)
        pos++;
      if (points_to_skip > 1.0) {
        pos++;
        points_to_skip -= 1;
        loop = true;
      }
    }
    assert(is_sample_point(*pos, clear_color));
    points_to_skip += step_size;
    cv::Point p = pos.pos();
    return std::make_tuple(p.y, p.x);
  }
};

template<typename T, int dim>
struct sample_point_random {
  const cv::Mat_<cv::Vec<T, dim>>& normals;
  cv::Mat_<unsigned char> used_pixels;
  const T clear_color;
  sample_point_random(const cv::Mat_<cv::Vec<T, dim>>& normals, const unsigned int points_to_deliver, const T clear_color) : normals(normals), used_pixels(normals.rows, normals.cols, static_cast<unsigned char>(0)), clear_color(clear_color) {
  }
  std::tuple<unsigned int, unsigned int> operator()() {
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
      cv::Vec<T, dim> normal = normals(y, x);
      if (!is_sample_point(normal, clear_color))
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
};

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

    void set_diffuse_specular(const unsigned int row, const unsigned int col, const std::tuple<float, float>& vals, const cv::Mat_<float> diffuse_tex, const cv::Mat_<float> specular_tex) {
      float diffuse, specular;
      std::tie(diffuse, specular) = vals;
      for (unsigned int i = 0; i < colors_per_light; i++) {
        for (unsigned int j = 0; j < colors_per_light; j++) {
          if (i == j) {
            // diffuse
            set(colors_per_light * row+i, colors_per_light + colors_per_light * components_per_light * col+j, diffuse*diffuse_tex(i));
            // specular
            set(colors_per_light * row+i, colors_per_light + colors_per_light * components_per_light * col+colors_per_light+j, specular*specular_tex(i));
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

    template<typename T>
    vector(const Lights<T>& lights) : v(gsl_vector_alloc(colors_per_light + lights.lights.size()*components_per_light*colors_per_light), gsl_vector_free) {
      set_ambient(lights.ambient);

      for (unsigned int i = 0; i < lights.lights.size(); i++) {
        const Light<T>& light = lights.lights.at(i);
        set_diffuse(i, light.get_diffuse());
        set_specular(i, light.get_specular());
      }
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
      if (this != &rhs)
        v = std::move(rhs.v);
      return *this;
    }

    bool operator==(const vector &rhs) const {
      for (size_t i = 0; i < v->size; i++)
        if (std::abs(get(i) - rhs.get(i)) > std::numeric_limits<float>::epsilon())
          return false;
      return true;
    }

    bool operator!=(const vector &rhs) const {
      return !(*this == rhs);
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
    
    template<typename T, int offset>
    std::vector<T> get(const size_t i) const {
      std::vector<T> tmp(colors_per_light+1);
      for (unsigned j = 0; j < colors_per_light; j++)
        tmp.at(j) = get(offset*colors_per_light + i*colors_per_light*components_per_light + j);
      tmp.at(colors_per_light) = 0;
      return tmp;
    }
    
    template<typename T>
    std::vector<T> get_ambient() const {
      return get<T, AMBIENT>(0);
    }
    
    template<typename T>
    std::vector<T> get_diffuse(const size_t i) const {
      return get<T, DIFFUSE>(i);
    }
    
    template<typename T>
    std::vector<T> get_specular(const size_t i) const {
      return get<T, SPECULAR>(i);
    }

    void set(const size_t i, const double x) {
      gsl_vector_set(v.get(), i, x);
    }

    /** set target color */
    void set(const unsigned int row, const cv::Vec<float, colors_per_light>& pixel) {
      for (size_t i = 0; i < colors_per_light; i++) {
        set(colors_per_light*row + i, pixel[i]);
      }
    }

    /** set light properties for initial guess */
    template<int offset, typename T>
    void set(const size_t i, const std::vector<T> &val) {
      for (size_t j = 0; j < colors_per_light; j++) {
        set(offset*colors_per_light + i*colors_per_light*components_per_light + j, val.at(j));
      }
    }

    template<typename T>
    void set_ambient(const std::vector<T> &ambient) {
      set<AMBIENT>(0, ambient);
    }

    template<typename T>
    void set_diffuse(const size_t i, const std::vector<T> &diffuse) {
      set<DIFFUSE>(i, diffuse);
    }

    template<typename T>
    void set_specular(const size_t i, const std::vector<T> &specular) {
      set<SPECULAR>(i, specular);
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

template<unsigned int colors_per_light, unsigned int components_per_light, template <typename, int> class point_selector, typename T, typename T1, int dim>
std::tuple<gsl::matrix<colors_per_light, components_per_light>, gsl::vector<colors_per_light, components_per_light>> create_linear_system(const cv::Mat_<cv::Vec<T, dim>>& image, const cv::Mat_<cv::Vec<T, dim>>& normals, const cv::Mat_<cv::Vec<T, dim>>& position, const cv::Mat_<cv::Vec<T, dim>>& diffuse_texture, const cv::Mat_<cv::Vec<T, dim>>& specular_texture, const cv::Mat_<GLfloat>& model_view_matrix, const float clear_color, const Lights<T1>& lights, const int alpha) {
  std::cout << "creating linear system" << std::endl;

  const unsigned int sample_max = get_maximum_number_of_sample_points(normals, clear_color);
  assert(sample_max >= lights.lights.size());
  const double fraction_of_points_to_take = std::log2((static_cast<double>(lights.lights.size())*components_per_light / sample_max) + 1);
  const double min_fraction_of_points_to_take = 0.1;
  
  const unsigned int rows = sample_max * std::max(fraction_of_points_to_take, min_fraction_of_points_to_take) * colors_per_light;
  const unsigned int cols = (1 + lights.lights.size() * components_per_light) * colors_per_light;

  std::cout << "linear_system will have " << rows << " rows and " << cols << " columns" << std::endl;

  gsl::matrix<colors_per_light, components_per_light> x(rows, cols);
  gsl::vector<colors_per_light, components_per_light> y(rows);

  // do not take all points of the image
  // calculate this value somehow, maybe specify the number of samples and
  // distribute them over the mesh in the image
  point_selector<T, dim> ps(normals, rows/colors_per_light, clear_color);
//  sample_point_random<T, dim> ps(normals, rows/colors_per_light, clear_color);

  for (unsigned int row = 0; row < rows/colors_per_light; row++) {
    std::cout << "at row " << row*colors_per_light << "/" << rows << "\r";
    // 1. find a good pixel
    int _x = 0;
    int _y = 0;
    std::tie(_x, _y) = ps();

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
    const cv::Mat_<float> diffuse_tex(diffuse_texture(_y, _x));
    const cv::Mat_<float> specular_tex(specular_texture(_y, _x));

    for (unsigned int col = 0; col < lights.lights.size(); col++) {
      const Light<T>& light = lights.lights.at(col);
      x.set_diffuse_specular(row, col, get_diffuse_specular(pos_vec, normal, light, model_view_matrix, alpha), diffuse_tex, specular_tex);
    }
  }
  std::cout << std::endl;

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

  const unsigned int colors_per_light = 3;
  const unsigned int components_per_light = 2;

  gsl::matrix<colors_per_light, components_per_light> x;
  gsl::vector<colors_per_light, components_per_light> y;

  std::tie(x, y) = create_linear_system<colors_per_light, components_per_light, sample_point_random>(image, normals, position, model_view_matrix, clear_color, lights, alpha);

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
    gsl_multimin_function function;
    gsl::vector<colors_per_light, components_per_light> start_point;
    gsl::vector<colors_per_light, components_per_light> step_size;
    
    public:
     
    minimizer(const gsl_multimin_fminimizer_type * T, gsl_multimin_function f, const gsl::vector<colors_per_light, components_per_light> start_point)
    : problem(gsl_multimin_fminimizer_alloc(T, f.n), gsl_multimin_fminimizer_free), function(std::move(f)), start_point(std::move(start_point)) {
      if (!problem)
        throw;
      step_size = gsl::vector<colors_per_light, components_per_light>(f.n, 1.0);
      reset();
    }
    
    gsl::vector<colors_per_light, components_per_light> get_solution() const {
      return gsl::vector<colors_per_light, components_per_light>(problem.get()->x);
    }
    
    double get_function_value() const {
      return problem.get()->fval;
    }
    
    void set(gsl_multimin_function f, const gsl::vector<colors_per_light, components_per_light> x, const gsl::vector<colors_per_light, components_per_light> ss) {
      function = std::move(f);
      start_point = std::move(x);
      step_size = std::move(ss);
      reset();
    }

    void set_function_and_start_point(gsl_multimin_function f, const gsl::vector<colors_per_light, components_per_light> pos) {
      function = std::move(f);
      start_point = std::move(pos);
      reset();
    } 

    void reset() {
      if (gsl_multimin_fminimizer_set(problem.get(), &function, start_point.get(), step_size.get()))
        throw;
    }
    
    int iterate(double epsabs = 1e-2) {
      if (gsl_multimin_fminimizer_iterate(problem.get()))
        throw;
      double size = gsl_multimin_fminimizer_size(problem.get());
      int status = gsl_multimin_test_size(size, epsabs);
      return status;
    }
    
  };
  
  template<int colors_per_light, int components_per_light>
  void matrix_vector_mult(double alpha, const matrix<colors_per_light, components_per_light>& A, const gsl_vector * X, double beta, vector<colors_per_light, components_per_light>& Y) {
    if(gsl_blas_dgemv(CblasNoTrans, alpha, A.get(), X, beta, Y.get()))
      throw;
  }
}

template<int colors_per_light, int components_per_light, bool free_variables = true>
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
  // how good we got the picture
  for (unsigned int i = 0; i < y_copy.size(); i++) {
    cost += fabs(y_copy.get(i));
  }
  for (unsigned int i = 0; i < v->size; i++) {
    double val = gsl_vector_get(v, i);
    // lights we don't see should stay at zero, hopefully clustering works then
    // when sphere cut by plane: visible changes are: less specular reflection, darker image, no better solution
    if (val > 0)
      cost += val;
    // values for lights properties only range from 0 to 1
    if (!free_variables) {
      if (val < 0.0 || val > 1.0) {
        if (val < 0.0)
          val = -val;
        else
          val = val-1.0;
        cost += std::exp2(std::exp2(5*val));
      }
    }
  }
  return cost;
}

template<int colors_per_light, int components_per_light>
bool check_solution(const gsl::vector<colors_per_light, components_per_light> &sol, double min = 0.0, double max = 1.0) {
  bool ret_val = true;
  for (unsigned int i = 0; i < sol.size(); i++) {
    ret_val &= check_bounds_of_value(sol.get(i), "light solution", min, max);
  }
  return ret_val;
}

template<typename T>
void optimize_lights_multi_dim_fit(const cv::Mat_<cv::Vec3f >& image, const cv::Mat_<cv::Vec3f>& normals, const cv::Mat_<cv::Vec3f>& position, const cv::Mat_<cv::Vec3f>& diffuse, const cv::Mat_<cv::Vec3f>& specular, const cv::Mat_<GLfloat>& model_view_matrix, const float clear_color, Lights<T>& lights, const size_t max_iter = 0, const int alpha = 50) {
  show_rgb_image("target image", image);
  const unsigned int colors_per_light = 3;
  const unsigned int components_per_light = 2;

  auto linear_system = create_linear_system<colors_per_light, components_per_light, sample_point_random>(image, normals, position, diffuse, specular, model_view_matrix, clear_color, lights, alpha);
  assert(std::get<0>(linear_system).get_rows() > std::get<0>(linear_system).get_cols());

  std::cout << "creating problem to minimize" << std::endl;
  gsl_multimin_function f{&cost<colors_per_light, components_per_light>, std::get<0>(linear_system).get_cols(), &linear_system};
  gsl::minimizer<colors_per_light, components_per_light> minimizer(gsl_multimin_fminimizer_nmsimplex2, f, lights);

  std::cout << "starting" << std::endl;
  for (size_t iter = 0; minimizer.iterate() == GSL_CONTINUE && (max_iter == 0 || iter < max_iter); iter++) {
    std::cout << "first stage " << iter <<  ", " << minimizer.get_function_value() << "\r";
  }
  std::cout << std::endl;
  
  f.f = &cost<colors_per_light, components_per_light, false>;
  minimizer.set_function_and_start_point(f, minimizer.get_solution());
  for (size_t iter = 0; minimizer.iterate() == GSL_CONTINUE && (max_iter == 0 || iter < max_iter); iter++) {
    std::cout << "second stage " << iter <<  ", " << minimizer.get_function_value() << "\r";
  }
  std::cout << std::endl;
  
  auto solution = minimizer.get_solution();
//  check_solution(solution);
  set_solution<float>(solution, lights);
//  print(lights);

  assert(solution == (gsl::vector<colors_per_light, components_per_light>(lights)));
  
  cv::waitKey(100);
  
}

template<typename T>
void optimize_lights_nnls(const cv::Mat_<cv::Vec3f >& image, const cv::Mat_<cv::Vec3f>& normals, const cv::Mat_<cv::Vec3f>& position, const cv::Mat_<cv::Vec3f>& diffuse, const cv::Mat_<cv::Vec3f>& specular, const cv::Mat_<GLfloat>& model_view_matrix, const float clear_color, Lights<T>& lights, const int alpha = 50) {
  show_rgb_image("target image", image);
  const unsigned int colors_per_light = 3;
  const unsigned int components_per_light = 2;
  gsl::matrix <colors_per_light, components_per_light> M;
  gsl::vector <colors_per_light, components_per_light> b;
  std::tie(M, b) = create_linear_system<colors_per_light, components_per_light, sample_point_random>(image, normals, position, diffuse, specular, model_view_matrix, clear_color, lights, alpha);
  
  nnls(0, 0, 0, 0, 0, 0, 0, 0, 0);
}

#endif /* SOLVER_H_ */
