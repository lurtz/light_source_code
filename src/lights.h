#ifndef LIGHTS_H_
#define LIGHTS_H_

#include "utils.h"

#include <GL/glew.h>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <functional>
#include <utility>
#include <array>
#ifdef OPENCV_OLD_INCLUDES
  #include <cv.h>
#else
  #include <opencv2/core/core.hpp>
#endif

namespace Lights {

using output_operators::operator<<;

template<typename T, int dim>
void set_uniforms(const GLuint programm_id, const std::string &uniform_name, cv::Vec<T, dim> value) {
  static_assert(dim <= 4, "opengl shader supports only up to 4 elements per vector");
  cv::Vec<T, 4> tmp = cv::Vec<T, 4>::all(0);
  for (unsigned int i = 0; i < 4; i++)
    tmp[i] = value[i];
  GLint uniform_light_property = glGetUniformLocation(programm_id, uniform_name.c_str());
  glUniform4f(uniform_light_property, tmp[0], tmp[1], tmp[2], tmp[3]);
//  if (uniform_light_property == -1)
//    std::cout << "uniform handle is -1 with uniform name " << iter_properties.first << std::endl;
}

// position, ambient, diffuse, specular in vec4
// RGB format
template<typename T, std::size_t dim>
struct Simple_Light {
  std::array<T, dim> position;
  std::array<T, dim> diffuse;
  std::array<T, dim> specular;
};

const std::array<Simple_Light<float, 4>, 4> light_properties = {{
  {{{ 4, 4, 2, 1}}, {{0.5, 0.0, 0.0, 0}}, {{1, 0, 0, 0}}}
 ,{{{-60,  1, 0, 1}}, {{0.0, 0.5, 0.0, 0}}, {{0, 1, 0, 0}}}
 ,{{{ 30,  0, 0, 1}}, {{0.0, 0.0, 0.5, 0}}, {{0, 0, 1, 0}}}
 ,{{{ 0,  -10, 0, 1}}, {{0.5, 0.0, 0.5, 0}}, {{1, 0, 1, 0}}}
}};

template<typename T, int dim>
cv::Vec<T, dim> default_ambient_color(T val = 0.1) {
  cv::Vec<T, dim> ret_val;
  for (unsigned int i = 0; i < dim; i++)
    ret_val[i] = val;
  return ret_val;
}

enum Properties {POSITION, DIFFUSE, SPECULAR};

template<typename T, int dim>
struct Light {
  static const std::map<Properties, std::string> binds;
  static const cv::Vec<T, dim> default_light_property;
  // position, diffuse, specuar
  std::map<std::string, cv::Vec<T, dim>> props;
  
  Light() {}

  Light(const cv::Vec<T, dim>& position, const cv::Vec<T, dim>& diffuse = default_light_property, const cv::Vec<T, dim>& specular = default_light_property) {
    get<POSITION>() = position;
    get<DIFFUSE>() = diffuse;
    get<SPECULAR>() = specular;
  }
  
  Light(const Simple_Light<T, dim>& light) {
    get<POSITION>() = create_vector_from_array(light.position);
    get<DIFFUSE>() = create_vector_from_array(light.diffuse);
    get<SPECULAR>() = create_vector_from_array(light.specular);
  }

  template<Properties prop>
  cv::Vec<T, dim>& get() {
    return props[binds.find(prop)->second];
  }

  template<Properties prop>
  const cv::Vec<T, dim>& get() const {
    auto x = props.find(binds.find(prop)->second);
    if (x == props.end())
      throw;
    return x->second;
  }

  static std::string get_shader_name(const unsigned int number, const std::string& property) {
    std::stringstream name;
    name << "lights[" << number << "]." << property;
    return name.str();
  }
  
  void setUniforms(const GLuint programm_id, const unsigned int i) const {
    for (auto iter_properties : props) {
      std::string property_name = iter_properties.first; // position, diffuse, specuar
      auto value = iter_properties.second;
      set_uniforms(programm_id, get_shader_name(i, property_name), value);
    }
  }
};

template<typename T, int dim>
const std::map<Properties, std::string> Light<T, dim>::binds = {{POSITION, "position"}, {DIFFUSE, "diffuse"}, {SPECULAR, "specular"}};

template<typename T, int dim>
const cv::Vec<T, dim> Light<T, dim>::default_light_property(cv::Scalar_<T>(0));

template<typename T, int dim>
std::ostream& operator<<(std::ostream& out, const Lights::Light<T, dim>& light) {
  out << light.props;
  return out;
}

// taken from
// http://www.xsi-blog.com/archives/115
template<typename T>
struct uniform_on_sphere_point_distributor {
  const double inc = M_PI * (3 - std::sqrt(5));
  const double off;
  unsigned int i;
  const float radius;
  const unsigned int num_lights;
  uniform_on_sphere_point_distributor(const float radius, const unsigned int num_lights) : off(2.0/num_lights), i(0), radius(radius), num_lights(num_lights) {
  }
  void seed(unsigned int c = 0) {
    i = c;
  }
  cv::Vec<T, 4> operator()() {
    i %= num_lights;
    const double y = i*off - 1 + off/2;
    const double r = sqrt(1 - y*y);
    const double phi = i * inc;
    cv::Vec<T, 4> position;
    position[0] = std::cos(phi) * r * radius;
    position[1] = y                 * radius;
    position[2] = std::sin(phi) * r * radius;
    position[3] = 1;
    assert(has_length_homogen_coordinates(position, radius, 16*std::numeric_limits<T>::epsilon()));
    i++;
    return position;
  }
  operator bool() {
    return i < num_lights;
  }
};

template<typename T, int D>
std::function<bool(cv::Vec<T, D>)> plane_acceptor(const cv::Vec<T, D>& normal, const cv::Vec<T, D>& point) {
  return [&](const cv::Vec<T, D>& p){return distFromPlane(p, normal, point) > 0;};
}

template<typename T, int dim>
struct Lights {
  cv::Vec<T, dim> ambient;
  std::vector<Light<T, dim>> lights;
  
  Lights() {}

  Lights(const T radius, const unsigned int num_lights = 10,
      const decltype(plane_acceptor<T, dim>(std::declval<const cv::Vec<T, dim>>(), std::declval<const cv::Vec<T, dim>>())) &point_acceptor = [](const cv::Vec<T, dim>& pos){return true;},
      const cv::Vec<T, dim> &ambient = (default_ambient_color<T, dim>())) : ambient(ambient) {
    auto parameter_candidates = find_border_parameter(num_lights, [&](const unsigned int number){return uniform_on_sphere_point_distributor<T>(radius, number);}, point_acceptor);
    std::vector<cv::Vec<T, dim>>& candidate_positions = std::get<1>(parameter_candidates);
    assert(candidate_positions.size() == num_lights);
    lights = std::vector<Light<T, dim>>(std::begin(candidate_positions), std::end(candidate_positions));
  }

  // TODO merge with last constructor, structure is really the same
  template<std::size_t count>
  Lights(const std::array<Simple_Light<T, dim>, count> &light_props, const cv::Vec<T, dim> &ambient = (default_ambient_color<T, dim>())) : ambient(ambient) {
    lights = std::vector<Light<T, dim>>(std::begin(light_props), std::end(light_props));
  }

  template<template <typename> class ForwardIterable>
  Lights(const ForwardIterable<cv::Vec<T, dim>>& positions) : ambient(default_ambient_color<T, dim>()) {
    lights = std::vector<Light<T, dim>>(std::begin(positions), std::end(positions));
  }
  
  void setUniforms(const GLuint programm_id) const {
    set_uniforms(programm_id, "ambient_light", ambient);

    unsigned int i = 0;
    for (auto &iter : lights) {
      iter.setUniforms(programm_id, i++);
    }
  }
};

template<typename T, int dim>
std::ostream& operator<<(std::ostream& out, const Lights<T, dim>& lights) {
  out << "ambient illumination: ";
  out << lights.ambient << std::endl;
  unsigned int i = 0;
  // TODO std::copy + std::ostream_iterator ?
  for (const Light<T, dim> &iter : lights.lights)
    out << "Light\n" << i++ << ": " << iter;
  return out;
}

}

#endif /* LIGHTS_H_ */
