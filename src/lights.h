#ifndef LIGHTS_H_
#define LIGHTS_H_

#include <GL/glew.h>
#include <GL/glut.h>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cassert>
#include <iterator>
#ifdef OPENCV_OLD_INCLUDES
  #include <cv.h>
#else
  #include <opencv2/core/core.hpp>
#endif

// position, ambient, diffuse, specular in vec4
// RGB format
const unsigned int NUM_PROPERTIES = 3;
const float light_properties[][4] = {
    { 4, 4, 2, 1}, {0.5, 0.0, 0.0, 0}, {1, 0, 0, 0}
    ,{-60,  1, 0, 1}, {0.0, 0.5, 0.0, 0}, {0, 1, 0, 0}
    ,{ 30,  0, 0, 1}, {0.0, 0.0, 0.5, 0}, {0, 0, 1, 0}
    ,{ 0,  -10, 0, 1}, {0.5, 0.0, 0.5, 0}, {1, 0, 1, 0}
};

template<typename T, int dim>
std::vector<T> create_vector_from_array(const T (&array)[dim]) {
  std::vector<T> tmp(array, array + dim);
  return tmp;
}

template<typename T>
void print(T t) {
  std::cout << t << std::endl;
}

template<typename T>
std::vector<T> create_ambient_color(T r = 0.1, T g = 0.1, T b = 0.1, T a = 0.0) {
  std::vector<T> ret_val;
  ret_val.push_back(r);
  ret_val.push_back(g);
  ret_val.push_back(b);
  ret_val.push_back(a);
  return ret_val;
}

std::string get_name(const unsigned int number, const std::string& item);
std::string get_position_name(const unsigned int number);
std::string get_diffuse_name(const unsigned int number);
std::string get_specular_name(const unsigned int number);

template<typename T>
struct Light {
  typedef std::map<std::string, std::vector<T> > properties;
  properties props;
  unsigned int number;
  
  Light() {}
  
  Light(const unsigned int number, const std::vector<T>& position, const std::vector<T>& diffuse, const std::vector<T>& specular)
  : number(number) {
    props[get_position_name(number)] = position;
    props[get_diffuse_name(number)] = diffuse;
    props[get_specular_name(number)] = specular;
  }
  
  template<int D>
  Light(const unsigned int number, const T (&position)[D], const T (&diffuse)[D], const T (&specular)[D]) : number(number) {
    props[get_position_name(number)] = create_vector_from_array(position);
    props[get_diffuse_name(number)] = create_vector_from_array(diffuse);
    props[get_specular_name(number)] = create_vector_from_array(specular);
  }
  
  const std::vector<T>& get_position(const unsigned int number) const {
    return props.find(get_position_name(number))->second;
  }
  
  std::vector<T>& get_diffuse(const unsigned int number) {
    return props[get_diffuse_name(number)];
  }
  
  std::vector<T>& get_specular(const unsigned int number) {
    return props[get_specular_name(number)];
  }
  
  void setUniforms(const GLuint programm_id) const {
    for (auto iter_properties : props) {
      GLint uniform_light_property = glGetUniformLocation(programm_id, iter_properties.first.c_str());
      auto value = iter_properties.second;
      glUniform4f(uniform_light_property, value.at(0), value.at(1), value.at(2), value.at(3));
      if (uniform_light_property == -1)
        std::cout << "uniform handle is -1 with uniform name " << iter_properties.first << std::endl;
    }
  }
};

template<typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec) {
  stream << "(";
  std::copy(std::begin(vec), std::end(vec)-1, std::ostream_iterator<T>(stream, ", "));
  stream << vec.back();
  stream << ")";
  return stream;
}

template<typename K, typename V>
std::ostream& operator<<(std::ostream& out, const std::pair<K,V>& pair) {
  out << pair.first << ", " << pair.second;
  return out;
}

template<typename K, typename V>
std::ostream& operator<<(std::ostream& out, const std::map<K,V>& map) {
  for (const auto& iter : map)
    out << iter << std::endl;
  return out;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Light<T>& light) {
  out << light.props;
  return out;
}

template<typename T, int D>
T distFromPlane(cv::Vec<T, D> x, cv::Vec<T, D> normal, cv::Vec<T, D> point) {
  return normal.dot(x-point);
}

template<typename T>
struct Lights {
  std::vector<T> ambient;
  std::vector<Light<T> > lights;
  
  // taken from
  // http://www.xsi-blog.com/archives/115
  Lights(float radius = 10, unsigned int num_lights = 10, std::vector<T> ambient = create_ambient_color<T>()) : ambient(ambient), lights(num_lights) {
    std::vector<T> default_light_property(4);
    // distribute light sources uniformly on the sphere
    const double inc = M_PI * (3 - sqrt(5));
    const double off = 2.0/num_lights;
    for (unsigned int i = 0; i < num_lights; i++) {
      const double y = i*off - 1 + off/2;
      const double r = sqrt(1 - y*y);
      const double phi = i * inc;
      std::vector<T> position(4);
      position.at(0) = cos(phi)*r * radius;
      position.at(1) = y          * radius;
      position.at(2) = sin(phi)*r * radius;
      position.at(3) = 1;
      assert(abs(cv::norm(cv::Vec<T, 3>(position.at(0), position.at(1), position.at(2))) - radius) < std::numeric_limits<T>::epsilon());
      lights.at(i) = Light<T>(i, position, default_light_property, default_light_property);
    }
  }
  
  template<int dim>
  Lights(const T light_props[][dim], const unsigned int count, const std::vector<T> &ambient = create_ambient_color<T>()) : ambient(ambient), lights(count) {
    for (unsigned int i = 0; i < NUM_PROPERTIES * count; i += NUM_PROPERTIES) {
      const unsigned int pos = i / NUM_PROPERTIES;
      lights.at(pos) = Light<T>(pos, light_props[i + 0], light_props[i + 1], light_props[i + 2]);
    }
  }
  
  void setUniforms(const GLuint programm_id) const {
    GLint uniform_light_property = glGetUniformLocation(programm_id, "ambient_color");
    glUniform4f(uniform_light_property, ambient.at(0), ambient.at(1), ambient.at(2), ambient.at(3));
    if (uniform_light_property == -1)
      std::cout << "uniform handle is -1 with uniform name " << "ambient_color" << std::endl;
    
    for (auto iter : lights) {
      iter.setUniforms(programm_id);
    }
  }
  
  void cropByPlane(cv::Vec3f normal, cv::Vec3f point) {
    // E : x = point + d1 * y + d2 * z
    // E : normal * x = normal * point
    // E : normal * (x-point) = 0
    
    // normal * (x + \lambda * normal) = normal * point
    // normal * x + \lambda = normal * point
    // \lambda = normal * (point-x)
    decltype(lights) lights_to_delete;
    for (const auto& light : lights)
      if (distFromPlane( light.get_position(light.number), normal, point) < 0) // TODP verify
        lights_to_delete.push_back(light);
//    lights. TODO
  }
};

template<typename T>
std::ostream& operator<<(std::ostream& out, const Lights<T>& lights) {
  out << "ambient illumination: ";
  out << lights.ambient << std::endl;
  for (Light<T> iter : lights.lights)
    out << iter;
  return out;
}

#endif /* LIGHTS_H_ */
