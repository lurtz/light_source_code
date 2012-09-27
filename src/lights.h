#ifndef LIGHTS_H_
#define LIGHTS_H_

#include <GL/glew.h>
#include <GL/glut.h>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#ifdef OPENCV_OLD_INCLUDES
  #include <cv.h>
#else
  #include <opencv2/core/core.hpp>
#endif
#include "utils.h"

// position, ambient, diffuse, specular in vec4
// RGB format
const unsigned int NUM_PROPERTIES = 3;
const float light_properties[][4] = {
    { 4, 4, 2, 1}, {0.5, 0.0, 0.0, 0}, {1, 0, 0, 0}
    ,{-60,  1, 0, 1}, {0.0, 0.5, 0.0, 0}, {0, 1, 0, 0}
    ,{ 30,  0, 0, 1}, {0.0, 0.0, 0.5, 0}, {0, 0, 1, 0}
    ,{ 0,  -10, 0, 1}, {0.5, 0.0, 0.5, 0}, {1, 0, 1, 0}
};

// TODO variadic number of arguments
template<typename T, int dim>
cv::Vec<T, dim> create_ambient_color(T r = 0.1, T g = 0.1, T b = 0.1, T a = 0.0) {
  return cv::Vec<T, dim>(r, b, g, a);
}

template<typename T, int dim>
struct Light {
  static const std::string position_name;
  static const std::string diffuse_name;
  static const std::string specular_name;
  
  typedef std::map<std::string, cv::Vec<T, dim>> properties; // position, diffuse, specuar
  properties props;
  
  Light() {}
  
  Light(const std::vector<T>& position, const std::vector<T>& diffuse, const std::vector<T>& specular) {
    for (unsigned int i = 0; i < dim; i++) {
      props[position_name][i] = position.at(i);
      props[diffuse_name][i] = diffuse.at(i);
      props[specular_name][i] = specular.at(i);
    }
  }

  Light(const cv::Vec<T, dim>& position, const cv::Vec<T, dim>& diffuse, const cv::Vec<T, dim>& specular) {
    props[position_name] = position;
    props[diffuse_name] = diffuse;
    props[specular_name] = specular;
  }
  
  Light(const T (&position)[dim], const T (&diffuse)[dim], const T (&specular)[dim]) {
    props[position_name] = create_vector_from_array(position);
    props[diffuse_name] = create_vector_from_array(diffuse);
    props[specular_name] = create_vector_from_array(specular);
  }
  
  const cv::Vec<T, dim>& get_position() const {
    return props.find(position_name)->second;
  }

  cv::Vec<T, dim>& get_diffuse() {
    return props[diffuse_name];
  }

  const cv::Vec<T, dim>& get_diffuse() const {
    return props.find(diffuse_name)->second;
  }

  cv::Vec<T, dim>& get_specular() {
    return props[specular_name];
  }

  const cv::Vec<T, dim>& get_specular() const {
    return props.find(specular_name)->second;
  }

  static std::string get_shader_name(const unsigned int number, const std::string& property) {
    std::stringstream name;
    name << "lights[" << number << "]." << property;
    return name.str();
  }
  
  void setUniforms(const GLuint programm_id, const unsigned int i) const {
    for (auto iter_properties : props) {
      std::string property_name = iter_properties.first; // position, diffuse, specuar
      GLint uniform_light_property = glGetUniformLocation(programm_id, get_shader_name(i, property_name).c_str());
      auto value = iter_properties.second;
      glUniform4f(uniform_light_property, value[0], value[1], value[2], value[3]);
//      if (uniform_light_property == -1)
//        std::cout << "uniform handle is -1 with uniform name " << iter_properties.first << std::endl;
    }
  }
};

template<typename T, int dim>
const std::string Light<T, dim>::position_name = "position";
template<typename T, int dim>
const std::string Light<T, dim>::diffuse_name = "diffuse";
template<typename T, int dim>
const std::string Light<T, dim>::specular_name = "specular";

template<typename T, int dim>
std::ostream& operator<<(std::ostream& out, const Light<T, dim>& light) {
  out << light.props;
  return out;
}

// taken from
// http://www.xsi-blog.com/archives/115
template<typename T>
struct uniform_on_sphere_point_distributor {
  double inc;
  const double off;
  unsigned int i;
  const float radius;
  const unsigned int num_lights;
  uniform_on_sphere_point_distributor(const float radius, const unsigned int num_lights) : inc(M_PI * (3 - sqrt(5))), off(2.0/num_lights), i(0), radius(radius), num_lights(num_lights) {
  }
  void seed(unsigned int c = 0) {
    i = c;
  }
  cv::Vec<T, 4> operator()() {
    const double y = i*off - 1 + off/2;
    const double r = sqrt(1 - y*y);
    const double phi = i * inc;
    cv::Vec<T, 4> position;
    position[0] = cos(phi)*r * radius;
    position[1] = y          * radius;
    position[2] = sin(phi)*r * radius;
    position[3] = 1;
    assert(abs(cv::norm(cv::Vec<T, 3>(position[0], position[1], position[2])) - radius) < std::numeric_limits<T>::epsilon());
    i++;
    return position;
  }
};

template<typename T>
struct uniform_on_sphere_point_distributor_without_limit {
  const T radius;
  halton_sequence seq1, seq2;
  std::default_random_engine engine;
  std::uniform_real_distribution<T> z_dist, t_dist;
  bool sign;
  uniform_on_sphere_point_distributor_without_limit(const T radius) : radius(radius), seq1(3, 1), z_dist(0.0, 360.0), t_dist(0.0, M_PI/2), sign(false) {
  }
  cv::Vec<T, 4> operator()() {
    T theta = z_dist(engine);
    T phi = t_dist(engine);
//    T theta = 360*seq1();
//    T phi = M_PI/2*seq2();
    cv::Vec<T, 4> tmp;
    tmp[0] = cos(sqrt(phi)) * cos(theta) * radius;
    tmp[1] = cos(sqrt(phi)) * sin(theta) * radius;
    tmp[2] = sin(sqrt(phi))              * radius;
    tmp[3] = 1;
    if (sign)
      tmp.at(2) = - tmp.at(2);
    sign ^= true;
    assert(abs(cv::norm(cv::Vec<T, 3>(tmp[0], tmp[1], tmp[2])) - radius) < std::numeric_limits<T>::epsilon());
    return tmp;
  }
};

template<typename T, int D>
std::function<bool(cv::Vec<T, D>)> plane_acceptor(const cv::Vec<T, D>& normal, const cv::Vec<T, D>& point) {
  return [&](const cv::Vec<T, D>& p){return distFromPlane(p, normal, point) > 0;};
}

template<typename T, int D>
std::tuple<decltype(plane_acceptor<T, D>), double> plane_acceptor_tuple(const cv::Vec<T, D>& normal, const cv::Vec<T, D>& point) {
  return std::make_tuple(plane_acceptor(normal, point), 2.1);
}

template<typename T, int dim>
struct Lights {
  cv::Vec<T, dim> ambient;
  std::vector<Light<T, dim>> lights;
  
  Lights() {}

  Lights(const T radius, const unsigned int num_lights = 10,
      const decltype(plane_acceptor<T, dim>) &point_acceptor = [](const cv::Vec<T, dim>& pos){return true;},
      const cv::Vec<T, dim> &ambient = create_ambient_color<T>()) : ambient(ambient), lights(num_lights) {
    cv::Vec<T, dim> default_light_property(cv::Scalar_<T>(0));
    uniform_on_sphere_point_distributor_without_limit<T> dist(radius);
    for (unsigned int i = 0; i < num_lights;) {
      auto position = dist();
      if (point_acceptor(position)) {
        lights.at(i) = Light<T, dim>(position, default_light_property, default_light_property);
        i++;
      }
    }
  }
  
  Lights(std::string bla, float radius = 10, unsigned int num_lights = 10,
      const std::tuple<decltype(plane_acceptor<T, dim>), double> &point_acceptor = std::make_tuple([](const cv::Vec<T, dim>& pos){return true;}, 1),
      const cv::Vec<T, dim> &ambient = create_ambient_color<T>()) : ambient(ambient), lights(num_lights) {
    cv::Vec<T, dim> default_light_property(cv::Scalar_<T>(0));
    decltype(plane_acceptor<T, dim>()) func;
    double num_discarded_points;
    std::tie(func, num_discarded_points) = point_acceptor;
    // distribute light sources uniformly on the sphere
    uniform_on_sphere_point_distributor<T> dist(radius, num_lights*num_discarded_points);
    for (unsigned int i = 0; i < num_lights;) {
      cv::Vec<T, 4> position = dist();
      if (func(position)) {
        lights.at(i) = Light<T, dim>(position, default_light_property, default_light_property);
        i++;
      }
    }
  }
  
  Lights(const T light_props[][dim], const unsigned int count, const cv::Vec<T, dim> &ambient = create_ambient_color<T, dim>()) : ambient(ambient), lights(count) {
    for (unsigned int i = 0; i < NUM_PROPERTIES * count; i += NUM_PROPERTIES) {
      const unsigned int pos = i / NUM_PROPERTIES;
      lights.at(pos) = Light<T, dim>(light_props[i + 0], light_props[i + 1], light_props[i + 2]);
    }
  }

  Lights(const cv::Mat_<cv::Vec<T, dim>>& positions, const cv::Vec<T, dim> &ambient = create_ambient_color<T, dim>()) : ambient(ambient), lights(positions.rows) {
    std::vector<T> default_light_property(4);
    unsigned int i = 0;
    for (const auto& pos : positions) {
      lights.at(i) = Light<T, dim>(pos, default_light_property, default_light_property);
      i++;
    }
  }
  
  void setUniforms(const GLuint programm_id) const {
    GLint uniform_light_property = glGetUniformLocation(programm_id, "ambient_light");
    glUniform4f(uniform_light_property, ambient[0], ambient[1], ambient[2], ambient[3]);
//    if (uniform_light_property == -1)
//      std::cout << "uniform handle is -1 with uniform name " << "ambient_color" << std::endl;

    unsigned int i = 0;
    for (auto iter : lights) {
      iter.setUniforms(programm_id, i++);
    }
  }
};

template<typename T, int dim>
std::ostream& operator<<(std::ostream& out, const Lights<T, dim>& lights) {
  out << "ambient illumination: ";
  out << lights.ambient << std::endl;
  unsigned int i = 0;
  // std::copy + std::ostream_iterator ?
  for (Light<T, dim> iter : lights.lights)
    out << "Light\n" << i++ << ": " << iter;
  return out;
}

#endif /* LIGHTS_H_ */
