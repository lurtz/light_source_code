#ifndef SOLVER_H_
#define SOLVER_H_

#include "lights.h"
#include "gsl.h"
#include "tests.h"
#include "kmeansw.h"
extern "C" {
#include "libtpc/include/nnls.h"
}
#include "utils.h"

#ifdef OPENCV_OLD_INCLUDES
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include<opencv2/highgui/highgui.hpp>
  #include <opencv2/imgproc/imgproc.hpp>
#endif

#include <vector>
#include <limits>
#include <memory>
#include <tuple>
#include <chrono>
//#include <algorithm>

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

template<typename T, int dim>
cv::Mat_<float> transform(const cv::Mat_<GLfloat>& model_view_matrix, const cv::Vec<T, dim>& light_pos_in_object_space_vector) {
  const cv::Mat_<float> light_pos_in_object_space_mat(light_pos_in_object_space_vector, false);
  // durch vierte komponente teilen
  const cv::Mat_<float> light_pos_vec4(model_view_matrix * light_pos_in_object_space_mat);
  const cv::Mat_<float> light_pos(light_pos_vec4 / light_pos_vec4(3), cv::Range(0, 3));
  
  return light_pos;
}

template<typename T, int dim>
bool is_sample_point(const cv::Vec<T, dim>& normal) {
  // skip if length is not 1
  return has_length(normal, 1);
}

template<typename T, int dim>
unsigned int get_maximum_number_of_sample_points(const cv::Mat_<cv::Vec<T, dim>>& normals) {
  return std::accumulate(std::begin(normals), std::end(normals), 0, [](unsigned int sum, const cv::Vec<T, dim>& normal) { return sum + is_sample_point(normal); });
}

template<typename T, int dim>
struct sample_point_deterministic {
  typename cv::Mat_<cv::Vec<T, dim>>::const_iterator pos;
  const double step_size;
  double points_to_skip;
  sample_point_deterministic(const cv::Mat_<cv::Vec<T, dim>>& normals, const unsigned int points_to_deliver) : pos(std::begin(normals)), step_size(static_cast<double>(get_maximum_number_of_sample_points(normals))/points_to_deliver), points_to_skip(0) {
    assert(step_size >= 1.0);
  }
  std::tuple<unsigned int, unsigned int> operator()() {
    bool loop = true;
    while (loop) {
      loop = false;
      while (!is_sample_point(*pos))
        pos++;
      if (points_to_skip > 1.0) {
        pos++;
        points_to_skip -= 1;
        loop = true;
      }
    }
    assert(is_sample_point(*pos));
    points_to_skip += step_size;
    cv::Point p = pos.pos();
    return std::make_tuple(p.y, p.x);
  }
};

// do not take all points of the image
// distribute them over the mesh in the image
template<typename T, int dim>
struct sample_point_random {
  const cv::Mat_<cv::Vec<T, dim>>& normals;
  cv::Mat_<unsigned char> used_pixels;
  sample_point_random(const cv::Mat_<cv::Vec<T, dim>>& normals, const unsigned int points_to_deliver) : normals(normals), used_pixels(normals.rows, normals.cols, static_cast<unsigned char>(0)) {
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
      if (!is_sample_point(normal))
        continue;

      assert(std::fabs(cv::norm(normal) - 1) < eps);
      assert(_x < normals.cols);
      assert(_y < normals.rows);

      _x = x;
      _y = y;
      used_pixels(y,x) = std::numeric_limits<unsigned char>::max();
    }
    return std::make_tuple(_x, _y);
  }
};

template<typename T, int dim>
std::tuple<float, float> get_diffuse_specular(const cv::Mat_<float> &pos_vec, const cv::Mat_<float> &normal, const Lights::Light<T, dim> &light, const cv::Mat_<GLfloat>& model_view_matrix, const int alpha) {
  const cv::Mat_<float> light_pos = transform(model_view_matrix, light.template get<Lights::Properties::POSITION>());

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

// TODO reduce template parameters T, dim
template<unsigned int colors_per_light, unsigned int components_per_light, template <typename, int> class point_selector, typename T, typename T1, int dim, int dim1>
std::tuple<gsl::matrix<colors_per_light, components_per_light>, gsl::vector<colors_per_light, components_per_light>> create_linear_system(const cv::Mat_<cv::Vec<T, dim>>& image, const cv::Mat_<cv::Vec<T, dim>>& normals, const cv::Mat_<cv::Vec<T, dim>>& position, const cv::Mat_<cv::Vec<T, dim>>& diffuse_texture, const cv::Mat_<cv::Vec<T, dim>>& specular_texture, const cv::Mat_<GLfloat>& model_view_matrix, const Lights::Lights<T1, dim1>& lights, const int alpha) {
  std::cout << "creating linear system" << std::endl;

  const unsigned int sample_max = get_maximum_number_of_sample_points(normals);
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
  point_selector<T, dim> ps(normals, rows/colors_per_light);

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
      const Lights::Light<T1, dim1>& light = lights.lights.at(col);
      x.set_diffuse_specular(row, col, get_diffuse_specular(pos_vec, normal, light, model_view_matrix, alpha), diffuse_tex, specular_tex);
    }
  }
  std::cout << std::endl;

  return std::make_tuple(std::move(x), std::move(y));
}

// TODO maybe method of object
template<typename T, int colors_per_light, int components_per_light, int dim>
void set_solution(const gsl::vector<colors_per_light, components_per_light>& c, Lights::Lights<T, dim>& lights) {
  // ambient
  lights.ambient = c.template get_cv_vec<T, gsl::AMBIENT>(0);

  // diffuse and specular
  for (unsigned int i = 0; i < lights.lights.size(); i++) {
    Lights::Light<T, dim>& light = lights.lights.at(i);
    cv::Vec<T, dim>& diff = light.template get<Lights::Properties::DIFFUSE>();
    cv::Vec<T, dim>& spec = light.template get<Lights::Properties::SPECULAR>();
    diff = c.template get_cv_vec<T, gsl::DIFFUSE>(i);
    spec = c.template get_cv_vec<T, gsl::SPECULAR>(i);
  }
}

template<int colors_per_light, int components_per_light>
struct ls {
  gsl::vector<colors_per_light, components_per_light> operator()(gsl::matrix<colors_per_light, components_per_light>& x, gsl::vector<colors_per_light, components_per_light>& y, const gsl::vector<colors_per_light, components_per_light>& initial_guess) {
    // get solution
    gsl::matrix<colors_per_light, components_per_light> cov(x.get_cols(), x.get_cols());
    gsl::vector<colors_per_light, components_per_light> c(x.get_cols());
    double chisq;
    gsl::workspace problem(x.get_rows(), x.get_cols());
    problem.solve(x, y, c, cov, chisq);

    return c;
  }
};

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
  // how good we got the picture
  double cost = abssum(y_copy);
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
struct multi_dim_fit {
    gsl::vector<colors_per_light, components_per_light> operator()(gsl::matrix<colors_per_light, components_per_light>& a, gsl::vector<colors_per_light, components_per_light>& b, const gsl::vector<colors_per_light, components_per_light>& initial_guess) {
    auto linear_system = std::make_tuple(std::move(a), std::move(b));
    assert(std::get<0>(linear_system).get_rows() > std::get<0>(linear_system).get_cols());

    std::cout << "creating problem to minimize" << std::endl;
    gsl_multimin_function f{&cost<colors_per_light, components_per_light>, std::get<0>(linear_system).get_cols(), &linear_system};
    gsl::minimizer<colors_per_light, components_per_light> minimizer(gsl_multimin_fminimizer_nmsimplex2, f, initial_guess);

    std::cout << "starting" << std::endl;
    for (size_t iter = 0; minimizer.iterate() == GSL_CONTINUE; iter++) {
      std::cout << "first stage " << iter <<  ", " << minimizer.get_function_value() << "\r";
    }
    std::cout << std::endl;

    f.f = &cost<colors_per_light, components_per_light, false>;
    minimizer.set_function_and_start_point(f, minimizer.get_solution());
    for (size_t iter = 0; minimizer.iterate() == GSL_CONTINUE; iter++) {
      std::cout << "second stage " << iter <<  ", " << minimizer.get_function_value() << "\r";
    }
    std::cout << std::endl;

    return minimizer.get_solution();
  }
};

template<int colors_per_light, int components_per_light>
struct nnls_struct {
  gsl::vector<colors_per_light, components_per_light> operator()(gsl::matrix<colors_per_light, components_per_light>& a, gsl::vector<colors_per_light, components_per_light>& b, const gsl::vector<colors_per_light, components_per_light>& initial_guess) {
    // nnls uses colum major matrix
    a.transpose();
    std::unique_ptr<double*[]> a_nnls = a.get_nnls_matrix();

    gsl::vector<colors_per_light, components_per_light> x(a.get_rows());

    nnls(a_nnls.get(), a.get_cols(), a.get_rows(), b.get()->data, x.get()->data, nullptr, nullptr, nullptr, nullptr);

    return x;
  }
};

template<int dim, typename T>
Lights::Lights<T, dim> reduce_lights(const Lights::Lights<T, dim>& lights, const unsigned int k) {
  cv::Mat_<cv::Vec<T, dim>> positions(lights.lights.size(), 1);
  std::vector<double> weight(positions.rows);
  for (int i = 0; i < positions.rows; i++) {
    const Lights::Light<T, dim>& light = lights.lights.at(i);
    cv::Vec<T, dim> pos;
    for (unsigned int j = 0; j < dim; j++)
      pos[j] = light.template get<Lights::Properties::POSITION>()[j];
    positions(i) = pos;
    // RGB for diffuse and specular -> 6 values from 0 to 1
    // let sum range from 0 to 2
    weight.at(i) = std::pow(20, 2.0/6*(sum(light.template get<Lights::Properties::DIFFUSE>()) + sum(light.template get<Lights::Properties::SPECULAR>())));

//    std::cout << "light position: " << pos << ", weight: " << weight.at(i) << std::endl;
  }
  cv::Mat labels;
  cv::TermCriteria termcrit(cv::TermCriteria::EPS, 1000, 0.01);
  cv::Mat centers;
  cv::kmeansw(positions, k, labels, termcrit, 1, cv::KMEANS_RANDOM_CENTERS, centers, weight);

  cv::Mat_<cv::Vec<T, dim>> centers_templ(k, 1);
  for (int i = 0; i < centers.rows; i++) {
    centers_templ(i) = centers.at<cv::Vec<T, dim>>(i);
  }

  return Lights::Lights<T, dim>(centers_templ);
}

template<template <int, int> class optimizer, template <typename, int> class point_selector, typename T, int dim>
void optimize_lights(const cv::Mat_<cv::Vec3f >& image, const cv::Mat_<cv::Vec3f>& normals, const cv::Mat_<cv::Vec3f>& position, const cv::Mat_<cv::Vec3f>& diffuse, const cv::Mat_<cv::Vec3f>& specular, const cv::Mat_<GLfloat>& model_view_matrix, Lights::Lights<T, dim>& lights, const int alpha = 50) {
  //  order of images is: xyz, RGB
  const unsigned int colors_per_light = 3;
  const unsigned int components_per_light = 2;

  gsl::matrix<colors_per_light, components_per_light> a;
  gsl::vector<colors_per_light, components_per_light> b;
  std::tie(a, b) = create_linear_system<colors_per_light, components_per_light, point_selector>(image, normals, position, diffuse, specular, model_view_matrix, lights, alpha);

  gsl::vector<colors_per_light, components_per_light> x = optimizer<colors_per_light, components_per_light>()(a, b, lights);
  
  check_solution(x);
  
  set_solution<float>(x, lights);

  assert(x == (gsl::vector<colors_per_light, components_per_light>(lights)));
}

template<template <int, int> class optimizer, template <typename, int> class point_selector, typename T, int dim>
Lights::Lights<T, dim> calc_lights(const std::tuple<cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<float>, cv::Mat_<float>>& image_data, Lights::Lights<T, dim>& a_lot_of_lights, const bool single_pass, const unsigned int small_num_lights = 10) {
//  testkmeansall();
  assert(test_gsl_vector_iterator());

  const auto start_time = std::chrono::high_resolution_clock::now();

  cv::Mat_<cv::Vec3f> image;
  cv::Mat_<cv::Vec3f> normals;
  cv::Mat_<cv::Vec3f> position;
  cv::Mat_<cv::Vec3f> diffuse;
  cv::Mat_<cv::Vec3f> specular;
  cv::Mat_<float> model_view_matrix;

  std::tie(image, normals, position, diffuse, specular, std::ignore, model_view_matrix) = image_data;

  show_rgb_image("target image", image);
//  cv::imshow("normals", normals);
//  cv::imshow("position", position);

//  test_modelview_matrix_and_light_positions<T>(model_view_matrix, lights);

//  test_normals(normals);
//  get_min_max_and_print(normals);

  optimize_lights<optimizer, point_selector>(image, normals, position, diffuse, specular, model_view_matrix.t(), a_lot_of_lights);
  const auto time_after_huge_lights_run = std::chrono::high_resolution_clock::now();
  std::cout << "a lot of lights optimized" << std::endl;

  Lights::Lights<T, dim> lights;
  
  if (!single_pass) {
    lights = reduce_lights(a_lot_of_lights, small_num_lights);
    std::cout << "a lot of lights reduced" << std::endl;

    optimize_lights<optimizer, point_selector>(image, normals, position, diffuse, specular, model_view_matrix.t(), lights);

    std::cout << "small number of lights reduced" << std::endl;
  } else {
    lights = a_lot_of_lights;
  }

  const auto finish_time = std::chrono::high_resolution_clock::now();

  std::cout << "complete run: " << finish_time - start_time << std::endl;
  std::cout << "  light estimation huge light number: " << time_after_huge_lights_run - start_time << std::endl;
  std::cout << "  light estimation smaller light number: " << finish_time - time_after_huge_lights_run << std::endl;

  cv::waitKey(100);

  return lights;
}

#endif /* SOLVER_H_ */
