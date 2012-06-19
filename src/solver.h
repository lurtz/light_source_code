#ifndef SOLVER_H_
#define SOLVER_H_

#include "lights.h"
#include <cv.hpp>
#include <vector>
#include <gsl/gsl_multifit.h>
#include <iomanip>

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
void optimize_lights(cv::Mat& original_image, cv::Mat& image, cv::Mat& normals, cv::Mat& depth, cv::Mat& modelview_projection_matrix, std::vector<typename Light<T>::properties>& lights, const int alpha = 50) {
  int new_channel_count = std::max(original_image.channels(), image.channels());
  original_image.reshape(new_channel_count);
  image.reshape(new_channel_count);

  cv::Mat correct_format_image;
  original_image.convertTo(correct_format_image, CV_32F);

  cv::Mat diff = image - correct_format_image;
  cv::imshow("differenz", diff);
  cv::waitKey(100);

//  gsl_multifit_linear_workspace * problem = gsl_multifit_linear_alloc(1000, lights.size()*9);
  // do not take all points of the image
  // TODO calculate this value somehow, maybe specify the number of samples and
  //      distribute them over the mesh in the image

  const unsigned int div = 1000;
  const unsigned int colors_per_light = 3;
  const unsigned int rows = original_image.rows * original_image.cols / div * colors_per_light;
  const unsigned int components_per_light = 2;
  const unsigned int cols = (1 + lights.size() * components_per_light) * colors_per_light; // TODO add for each component RGB
  gsl_matrix *x = gsl_matrix_alloc (rows, cols);
  gsl_matrix *cov = gsl_matrix_alloc(cols, cols);
  gsl_vector *y = gsl_vector_alloc(rows);
  gsl_vector *c = gsl_vector_alloc(cols);

  cv::Mat eye_dir = (cv::Mat_<float>(3,1) << 0, 0, -1);
  cv::Mat used_pixels(image.rows, image.cols, CV_8S, cv::Scalar(0));

  for (unsigned int row = 0; row < rows; row+=colors_per_light) {
    // 1. find a good pixel
    unsigned int _x = 0;
    unsigned int _y = 0;
    while (_x == 0 && _y == 0) {
      unsigned int x = image.cols * drand48();
      unsigned int y = image.rows * drand48();

      // skip if already taken
      if (used_pixels.at<char>(y, x))
        continue;

      // skip if no object
      // assume that a normal is (0,0,0) where no object is
      cv::Vec<float, 3> normal = normals.at<cv::Vec<float, 3> >(y, x);
      if (normal[0] == 0 && normal[1] == 0 && normal[2] == 0)
        continue;

      _x = x;
      _y = y;
      used_pixels.at<char>(y,x) = 1;
    }

    // 2. set matrix parameter for pixel
    // set value of pixel in the image to the vector
    const cv::Vec<float, colors_per_light>& pixel = image.at<cv::Vec<float, colors_per_light> >(_y, _x);
    for (unsigned int i = 0; i < colors_per_light; i++) {
      gsl_vector_set(y, row + i, pixel[i]);
    }
    // set shading parameter for a pixel in the matrix
    // ambient term
    for (unsigned int i = 0; i < colors_per_light; i++)
      for (unsigned int j = 0; j < colors_per_light; j++)
        if (i == j)
          gsl_matrix_set(x, row + i, 0, 1);
        else
          gsl_matrix_set(x, row + i, 0, 0);

    const cv::Mat pos_vec = (cv::Mat_<float>(3,1) << _x, _y, depth.at<float>(_y, _x));
    const cv::Mat normal(normals.at<cv::Vec<float, 3> >(_y, _x), false);
    for (unsigned int col = colors_per_light; col < cols; col+=components_per_light*colors_per_light) {
      typename Light<T>::properties& props = lights.at(col/components_per_light/colors_per_light);
      // TODO need to transform light_pos to image_space
      const std::vector<T>& light_pos_in_world_space_vector = props[get_position_name(col/components_per_light/colors_per_light)];
      const cv::Mat light_pos_in_world_space_mat(light_pos_in_world_space_vector, false);
      const cv::Mat light_pos(modelview_projection_matrix * light_pos_in_world_space_mat, cv::Range(0, 3));

      const cv::Mat L_m = light_pos - pos_vec;
      // should be a scalar
      const cv::Mat L_m_N = L_m.t() * normal;
      assert(L_m_N.dims == 2);
      assert(L_m_N.cols == 1);
      assert(L_m_N.rows == 1);
      const float diffuse = L_m_N.at<float>(0,0);

      const cv::Mat R_m = 2*diffuse * normal - L_m;
      // should be a scalar
      const cv::Mat R_m_V = R_m.t() * eye_dir;
      assert(R_m_V.dims == 2);
      assert(R_m_V.cols == 1);
      assert(R_m_V.rows == 1);
      const float specular = std::pow(R_m_V.at<float>(0,0), alpha);

      //   global  each light
      //   amb     diff   spec
      //   r g b   r g b  r g b
      // r 1 0 0   1 0 0  1 0 0
      // g 0 1 0   0 1 0  0 1 0
      // b 0 0 1   0 0 1  0 0 1
      for (unsigned int i = 0; i < colors_per_light; i++) {
        for (unsigned int j = 0; j < colors_per_light; j++) {
          std::cout << col+j << "/" << cols << std::endl;
          std::cout << col+colors_per_light+j << "/" << cols << std::endl;
          if (i == j) {
            // diffuse
            gsl_matrix_set(x, row+i, col+j, diffuse);
            // specular
            gsl_matrix_set(x, row+i, col+colors_per_light+j, specular);
          } else {
            // diffuse
            gsl_matrix_set(x, row+i, col+j, 0);
            // specular
            std::cout << col+colors_per_light+j << "/" << cols << std::endl;
            gsl_matrix_set(x, row+i, col+colors_per_light+j, 0);
          }
        }
      }
    }
  }

  double chisq;
  gsl_multifit_linear_workspace * problem = gsl_multifit_linear_alloc(rows, cols);
  gsl_multifit_linear (x, y, c, cov, &chisq, problem);
  gsl_multifit_linear_free(problem);

  // TODO handle ambient light here and in shader
  for (unsigned int col = colors_per_light; col < cols; col+=components_per_light*colors_per_light) {
    typename Light<T>::properties& props = lights.at(col/components_per_light/colors_per_light);
    std::vector<T>& diff = props[get_diffuse_name(col/components_per_light/colors_per_light)];
    std::vector<T>& spec = props[get_specular_name(col/components_per_light/colors_per_light)];
    for (unsigned int i = 0; i < colors_per_light; i++) {
      diff.at(i) = gsl_vector_get(c, col + i);
      spec.at(i) = gsl_vector_get(c, col + colors_per_light + i);
    }
  }

  gsl_matrix_free(x);
  gsl_matrix_free(cov);
  gsl_vector_free(y);
  gsl_vector_free(c);
}

#endif /* SOLVER_H_ */
