#ifndef __GSL_RAII_H__
#define __GSL_RAII_H__

#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>
#include <memory>
#include <limits>
#include <iterator>
#include "lights.h"

namespace gsl {
  // TODO overload operators
  // TODO iterators
  enum Properties {AMBIENT = 0, DIFFUSE, SPECULAR};

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
    
    double get(const size_t i, const size_t j) const {
      assert(i < get_rows());
      assert(j < get_cols());
      return gsl_matrix_get(m.get(), i, j);
    }
    
    double& get(const size_t i, const size_t j) {
      assert(i < get_rows());
      assert(j < get_cols());
      return *gsl_matrix_ptr(m.get(), i, j);
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
    template<Properties prop, typename T>
    void set(const unsigned int row, const unsigned int col, const cv::Mat_<T>& values) {
      static_assert(prop <= components_per_light, "template parameter prop is bigger than number of components_per_light");
      for (unsigned int i = 0; i < colors_per_light; i++) {
        const size_t row_pos = colors_per_light * row + i;
        for (unsigned int j = 0; j < colors_per_light; j++) {
          size_t col_pos = colors_per_light * components_per_light * col + j;
          // add depending on the component we are times colors_per_light
          col_pos += colors_per_light * prop;
          get(row_pos, col_pos) = (i==j) * values(i);
        }
      }
    }
    
    bool operator==(const gsl::matrix<colors_per_light, components_per_light>& rhs) const {
      for (unsigned int i = 0; i < get_rows(); i++)
        for (unsigned int j = 0; j < get_cols(); j++)
          if (std::fabs(get(i,j) - rhs.get(i,j)) > std::numeric_limits<double>::epsilon()) {
            return false;
          }
      return true;
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
    vector(const Lights::Lights<T, dim>& lights) : v(gsl_vector_alloc(colors_per_light + lights.lights.size()*components_per_light*colors_per_light), gsl_vector_free) {
      set<AMBIENT>(0, lights.ambient);

      if (components_per_light > 0)
        for (unsigned int i = 0; i < lights.lights.size(); i++) {
          const Lights::Light<T, dim>& light = lights.lights.at(i);
          set<DIFFUSE>(i, light.template get<Lights::Properties::DIFFUSE>());
          if (components_per_light > 1)
            set<SPECULAR>(i, light.template get<Lights::Properties::SPECULAR>());
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

    double& get(const size_t i) {
      return *gsl_vector_ptr(v.get(), i);
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

    template<typename T, Properties offset>
    cv::Vec<T, colors_per_light+1> get_cv_vec(const size_t i) const {
      cv::Vec<T, colors_per_light+1> tmp;
      for (unsigned j = 0; j < colors_per_light; j++)
        tmp[j] = get(offset*colors_per_light + i*colors_per_light*components_per_light + j);
      tmp[colors_per_light] = 0;
      return tmp;
    }

    /** set target color */
    void set(const unsigned int row, const cv::Vec<float, colors_per_light>& pixel) {
      for (size_t i = 0; i < colors_per_light; i++) {
        get(colors_per_light*row + i) = pixel[i];
      }
    }

    /** set light properties for initial guess */
    template<Properties offset, typename T, int dim>
    void set(const size_t i, const cv::Vec<T, dim> &val) {
      static_assert(colors_per_light <= dim, "val contains less items than needed to set");
      for (size_t j = 0; j < colors_per_light; j++) {
        get(offset*colors_per_light + i*colors_per_light*components_per_light + j) = val[j];
      }
    }
    
    template<typename T, typename Y>
    struct abstract_iterator {
      unsigned int pos;
      typedef double value_type;
      typedef typename std::remove_reference<Y>::type* pointer;
      typedef typename std::remove_reference<Y>::type& reference;
      typedef std::bidirectional_iterator_tag iterator_category;
      typedef std::ptrdiff_t difference_type;
      
      abstract_iterator(unsigned int pos) : pos(pos) {}
      
      void check_pos() {
        if (pos > static_cast<T*>(this)->v.size())
          throw;
      }
      
      T operator++() {
        T copy = *static_cast<T*>(this);
        pos++;
        check_pos();
        return copy;
      }
      
      T& operator++(int) {
        pos++;
        check_pos();
        return *static_cast<T*>(this);
      }
      
      T operator--() {
        T copy = *static_cast<T*>(this);
        pos--;
        check_pos();
        return copy;
      }
      
      T& operator--(int) {
        pos--;
        check_pos();
        return *static_cast<T*>(this);
      }
      
      T operator-(const int step) {
        T copy = *static_cast<T*>(this);
        copy.pos -= step;
        check_pos();
        return copy;
      }
      
      Y operator*() {
        return static_cast<T*>(this)->v.get(pos);
      }
      
      template<typename V, typename X>
      bool operator==(const abstract_iterator<V, X>& rhs) const {
        const vector<colors_per_light, components_per_light>& our_vector = static_cast<const T&>(*this).v;
        const vector<colors_per_light, components_per_light>& their_vector = static_cast<const V&>(rhs).v;
        return pos == rhs.pos && &our_vector == &their_vector;
      }
      
      template<typename V, typename X>
      bool operator!=(const abstract_iterator<V, X>& rhs) const {
        return !(*this == rhs);
      }
    };
    
    struct iterator : public abstract_iterator<iterator, double&> {
      vector<colors_per_light, components_per_light>& v;
      
      iterator(vector<colors_per_light, components_per_light>& v, unsigned int pos = 0) : abstract_iterator<iterator, double&>(pos), v(v) {}
    };
    
    struct const_iterator : public abstract_iterator<const_iterator, double> {
      const vector<colors_per_light, components_per_light>& v;
      
      const_iterator(const vector<colors_per_light, components_per_light>& v, unsigned int pos = 0) : abstract_iterator<const_iterator, double>(pos), v(v) {}
    };
    
    iterator begin() {
      return iterator(*this);
    }
    
    const_iterator begin() const {
      return const_iterator(*this);
    }
    
    iterator end() {
      return iterator(*this, size());
    }
    
    const_iterator end() const {
      return const_iterator(*this, size());
    }
  };
  
  template<int A, int B>
  std::ostream& operator<<(std::ostream& out, const vector<A, B>& v) {
    out << "gsl::vector<" << A << ", " << B << "> = ";
    std::copy(std::begin(v), std::end(v)-1, std::ostream_iterator<double>(out, ", "));
    out << *(std::end(v)-1);
    return out;
  }

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