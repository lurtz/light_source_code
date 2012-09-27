#ifndef __GSL_RAII_H__
#define __GSL_RAII_H__

#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>
#include <memory>
#include <limits>
#include "lights.h"

namespace gsl {
  // TODO overload operators
  // TODO iterators
  enum {AMBIENT = 0, DIFFUSE, SPECULAR};

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
      return m->size1;
    }

    size_t get_cols() const {
      return m->size2;
    }

    std::unique_ptr<double*[]> get_nnls_matrix() const {
      std::unique_ptr<double*[]> array(new double*[get_rows()]);
      for (size_t i = 0; i < get_rows(); i++) {
        array[i] = gsl_matrix_ptr(m.get(), i, 0);
      }
      return array;
    }

    void transpose() {
      gsl::matrix<colors_per_light, components_per_light> dest(get_cols(), get_rows());
      if (gsl_matrix_transpose_memcpy(dest.m.get(), m.get()))
        throw;
      *this = std::move(dest);
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
          set(colors_per_light * row + i, j, i == j);
    }

    void set_diffuse_specular(const unsigned int row, const unsigned int col, const std::tuple<float, float>& vals, const cv::Mat_<float> diffuse_tex, const cv::Mat_<float> specular_tex) {
      float diffuse, specular;
      std::tie(diffuse, specular) = vals;
      for (unsigned int i = 0; i < colors_per_light; i++) {
        const size_t row_pos = colors_per_light * row + i;
        for (unsigned int j = 0; j < colors_per_light; j++) {
          const size_t col_pos = colors_per_light * (DIFFUSE + components_per_light * col) + j;
          if (i == j) {
            // diffuse
            set(row_pos, col_pos, diffuse*diffuse_tex(i));
            // specular
            set(row_pos, col_pos + (SPECULAR-DIFFUSE) * colors_per_light, specular*specular_tex(i));
          } else {
            // diffuse
            set(row_pos, col_pos, 0);
            // specular
            set(row_pos, col_pos + (SPECULAR-DIFFUSE) * colors_per_light, 0);
          }
        }
      }
    }
  };

  template<int colors_per_light, int components_per_light>
  class vector {
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

    template<typename T, int dim>
    vector(const Lights<T, dim>& lights) : v(gsl_vector_alloc(colors_per_light + lights.lights.size()*components_per_light*colors_per_light), gsl_vector_free) {
      set<AMBIENT>(0, lights.ambient);

      for (unsigned int i = 0; i < lights.lights.size(); i++) {
        const Light<T, dim>& light = lights.lights.at(i);
        set<DIFFUSE>(i, light.get_diffuse());
        set<SPECULAR>(i, light.get_specular());
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

    vector(vector&&) = default;

    vector& operator=(vector&&) = default;

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
    cv::Vec<T, colors_per_light+1> get(const size_t i) const {
      cv::Vec<T, colors_per_light+1> tmp;
      for (unsigned j = 0; j < colors_per_light; j++)
        tmp[j] = get(offset*colors_per_light + i*colors_per_light*components_per_light + j);
      tmp[colors_per_light] = 0;
      return tmp;
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
    template<int offset, typename T, int dim>
    void set(const size_t i, const cv::Vec<T, dim> &val) {
      static_assert(colors_per_light <= dim, "val contains less items than needed to set");
      for (size_t j = 0; j < colors_per_light; j++) {
        set(offset*colors_per_light + i*colors_per_light*components_per_light + j, val[j]);
      }
    }
  };

  template<int colors_per_light, int components_per_light>
  void matrix_vector_mult(double alpha, const matrix<colors_per_light, components_per_light>& A, const gsl_vector * X, double beta, vector<colors_per_light, components_per_light>& Y) {
    if(gsl_blas_dgemv(CblasNoTrans, alpha, A.get(), X, beta, Y.get()))
      throw;
  }

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
}


#endif /* __GSL_RAII_H__ */