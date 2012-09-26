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

template<typename T>
std::vector<T> create_ambient_color(T r = 0.1, T g = 0.1, T b = 0.1, T a = 0.0) {
  std::vector<T> ret_val;
  ret_val.push_back(r);
  ret_val.push_back(g);
  ret_val.push_back(b);
  ret_val.push_back(a);
  return ret_val;
}
  
template<typename T>
struct Light {
  static const std::string position_name;
  static const std::string diffuse_name;
  static const std::string specular_name;
  
  typedef std::map<std::string, std::vector<T> > properties; // position, diffuse, specuar
  properties props;
  
  Light() {}
  
  Light(const std::vector<T>& position, const std::vector<T>& diffuse, const std::vector<T>& specular) {
    props[position_name] = position;
    props[diffuse_name] = diffuse;
    props[specular_name] = specular;
  }
  
  template<int D>
  Light(const T (&position)[D], const T (&diffuse)[D], const T (&specular)[D]) {
    props[position_name] = create_vector_from_array(position);
    props[diffuse_name] = create_vector_from_array(diffuse);
    props[specular_name] = create_vector_from_array(specular);
  }

  template<int D>
  Light(const cv::Vec<T, D>& pos, const std::vector<T>& diffuse, const std::vector<T>& specular) {
    std::vector<T> tmp_pos(D);
    for (unsigned int i = 0; i < D; i++)
      tmp_pos.at(i) = pos[i];
    props[position_name] = tmp_pos;
    props[diffuse_name] = diffuse;
    props[specular_name] = specular;
  }
  
  const std::vector<T>& get_position() const {
    return props.find(position_name)->second;
  }

  std::vector<T>& get_diffuse() {
    return props[diffuse_name];
  }

  const std::vector<T>& get_diffuse() const {
    return props.find(diffuse_name)->second;
  }

  std::vector<T>& get_specular() {
    return props[specular_name];
  }

  const std::vector<T>& get_specular() const {
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
      glUniform4f(uniform_light_property, value.at(0), value.at(1), value.at(2), value.at(3));
//      if (uniform_light_property == -1)
//        std::cout << "uniform handle is -1 with uniform name " << iter_properties.first << std::endl;
    }
  }
};

template<typename T>
const std::string Light<T>::position_name = "position";
template<typename T>
const std::string Light<T>::diffuse_name = "diffuse";
template<typename T>
const std::string Light<T>::specular_name = "specular";

template<typename T>
std::ostream& operator<<(std::ostream& out, const Light<T>& light) {
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
  std::vector<T> operator()() {
    const double y = i*off - 1 + off/2;
    const double r = sqrt(1 - y*y);
    const double phi = i * inc;
    std::vector<T> position(4);
    position.at(0) = cos(phi)*r * radius;
    position.at(1) = y          * radius;
    position.at(2) = sin(phi)*r * radius;
    position.at(3) = 1;
    assert(abs(cv::norm(cv::Vec<T, 3>(position.at(0), position.at(1), position.at(2))) - radius) < std::numeric_limits<T>::epsilon());
    i++;
    return position;
  }
};

typedef struct halton_sequence {
  const unsigned int m_base;
  float m_number;
  halton_sequence(const unsigned int base = 2, const unsigned int number = 1);
  float operator()();
  void discard(unsigned long long z);
  void seed(const unsigned int i = 1);
  static float min();
  static float max();
} halton_sequence;


template<typename T>
struct uniform_on_sphere_point_distributor_without_limit {
  const T radius;
  halton_sequence seq1, seq2;
  std::default_random_engine engine;
  std::uniform_real_distribution<T> z_dist, t_dist;
  bool sign;
  uniform_on_sphere_point_distributor_without_limit(const T radius) : radius(radius), seq1(3, 1), z_dist(0.0, 360.0), t_dist(0.0, M_PI/2), sign(false) {
  }
  std::vector<T> operator()() {
    T theta = z_dist(engine);
    T phi = t_dist(engine);
//    T theta = 360*seq1();
//    T phi = M_PI/2*seq2();
    std::vector<T> tmp(4);
    tmp.at(0) = cos(sqrt(phi)) * cos(theta) * radius;
    tmp.at(1) = cos(sqrt(phi)) * sin(theta) * radius;
    tmp.at(2) = sin(sqrt(phi))              * radius;
    tmp.at(3) = 1;
    if (sign)
      tmp.at(2) = - tmp.at(2);
    sign ^= true;
    assert(abs(cv::norm(cv::Vec<T, 3>(tmp.at(0), tmp.at(1), tmp.at(2))) - radius) < std::numeric_limits<T>::epsilon());
    return tmp;
  }
};

template<typename T, int D>
std::function<bool(std::vector<T>)> plane_acceptor(const cv::Vec<T, D>& normal, const cv::Vec<T, D>& point) {
  return [&](const std::vector<T>& p){return distFromPlane(p, normal, point) > 0;};
}

template<typename T, int D>
std::tuple<std::function<bool(std::vector<T>)>, double> plane_acceptor_tuple(const cv::Vec<T, D>& normal, const cv::Vec<T, D>& point) {
  return std::make_tuple(plane_acceptor(normal, point), 2.1);
}

template<typename T>
struct Lights {
  std::vector<T> ambient;
  std::vector<Light<T> > lights;
  
  Lights() {}
  
  Lights(const T radius, const unsigned int num_lights = 10,
      const std::function<bool(std::vector<T>)> &point_acceptor = [](const std::vector<T>& pos){return true;},
      const std::vector<T> &ambient = create_ambient_color<T>()) : ambient(ambient), lights(num_lights) {
    std::vector<T> default_light_property(4);
    uniform_on_sphere_point_distributor_without_limit<T> dist(radius);
    for (unsigned int i = 0; i < num_lights;) {
      std::vector<T> position = dist();
      if (point_acceptor(position)) {
        lights.at(i) = Light<T>(position, default_light_property, default_light_property);
        i++;
      }
    }
  }
  
  Lights(std::string bla, float radius = 10, unsigned int num_lights = 10,
      const std::tuple<std::function<bool(std::vector<T>)>, double> &point_acceptor = std::make_tuple([](const std::vector<T>& pos){return true;}, 1),
      std::vector<T> ambient = create_ambient_color<T>()) : ambient(ambient), lights(num_lights) {
    std::vector<T> default_light_property(4);
    std::function<bool(std::vector<T>)> func;
    double num_discarded_points;
    std::tie(func, num_discarded_points) = point_acceptor;
    // distribute light sources uniformly on the sphere
    uniform_on_sphere_point_distributor<T> dist(radius, num_lights*num_discarded_points);
    for (unsigned int i = 0; i < num_lights;) {
      std::vector<T> position = dist();
      if (func(position)) {
        lights.at(i) = Light<T>(position, default_light_property, default_light_property);
        i++;
      }
    }
  }
  
  template<int dim>
  Lights(const T light_props[][dim], const unsigned int count, const std::vector<T> &ambient = create_ambient_color<T>()) : ambient(ambient), lights(count) {
    for (unsigned int i = 0; i < NUM_PROPERTIES * count; i += NUM_PROPERTIES) {
      const unsigned int pos = i / NUM_PROPERTIES;
      lights.at(pos) = Light<T>(light_props[i + 0], light_props[i + 1], light_props[i + 2]);
    }
  }

  template<int dim>
  Lights(const cv::Mat_<cv::Vec<T, dim>>& positions, const std::vector<T> &ambient = create_ambient_color<T>()) : ambient(ambient), lights(positions.rows) {
    std::vector<T> default_light_property(4);
    unsigned int i = 0;
    for (const auto& pos : positions) {
      lights.at(i) = Light<T>(pos, default_light_property, default_light_property);
      i++;
    }
  }
  
  void setUniforms(const GLuint programm_id) const {
    GLint uniform_light_property = glGetUniformLocation(programm_id, "ambient_light");
    glUniform4f(uniform_light_property, ambient.at(0), ambient.at(1), ambient.at(2), ambient.at(3));
//    if (uniform_light_property == -1)
//      std::cout << "uniform handle is -1 with uniform name " << "ambient_color" << std::endl;

    unsigned int i = 0;
    for (auto iter : lights) {
      iter.setUniforms(programm_id, i++);
    }
  }
};

template<typename T>
std::ostream& operator<<(std::ostream& out, const Lights<T>& lights) {
  out << "ambient illumination: ";
  out << lights.ambient << std::endl;
  unsigned int i = 0;
  for (Light<T> iter : lights.lights)
    out << "Light\n" << i++ << ": " << iter;
  return out;
}

#endif /* LIGHTS_H_ */
