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

template<typename T>
std::vector<T> create_ambient_color(T r = 0.1, T g = 0.1, T b = 0.1, T a = 0.0) {
  std::vector<T> ret_val;
  ret_val.push_back(r);
  ret_val.push_back(g);
  ret_val.push_back(b);
  ret_val.push_back(a);
  return ret_val;
}

std::string get_name(const unsigned int number, std::string item);
std::string get_position_name(const unsigned int number);
std::string get_diffuse_name(const unsigned int number);
std::string get_specular_name(const unsigned int number);

template<typename T>
struct Light {
  typedef std::map<std::string, std::vector<T> > properties;
  properties props;
  Light(const unsigned int number, const std::vector<T>& position,
    const std::vector<T>& diffuse, const std::vector<T>& specular) {
    props[get_position_name(number)] = position;
    props[get_diffuse_name(number)] = diffuse;
    props[get_specular_name(number)] = specular;
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
std::ostream& operator<<(std::ostream& out, const std::map<K,V>& map) {
  for (auto iter : map) {
    out << iter->first << ", " << iter->second << std::endl;
  }
  return out;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Light<T>& light) {
  out << light.props;
  return out;
}

template<typename T>
struct Lights {
  std::vector<T> ambient;
  std::vector<Light<T> > lights;
  
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
      lights.at(i) = Light<T>(i, std::move(position), default_light_property, default_light_property);
    }
  }
  
  template<int dim>
  Lights(const T light_props[][dim], const unsigned int count, std::vector<T> ambient = create_ambient_color<T>()) : ambient(ambient), lights(count) {
    for (unsigned int i = 0; i < NUM_PROPERTIES * count; i += NUM_PROPERTIES) {
      std::vector<T> position = create_vector_from_array(light_props[i + 0]);
      std::vector<T> diffuse = create_vector_from_array(light_props[i + 1]);
      std::vector<T> specular = create_vector_from_array(light_props[i + 2]);
      lights.at(i/NUM_PROPERTIES) = Light<T>(i / NUM_PROPERTIES, std::move(position), std::move(diffuse), std::move(specular));
    }
  }
  
  void print() {
    std::cout << *this << std::endl;
  }
};

template<typename T>
std::ostream& operator<<(std::ostream& out, Lights<T> lights) {
  out << "ambient illumination: ";
  out << lights.ambient << std::endl;
  for (auto iter : lights.lights)
    out << *iter;
  return out;
}

template<typename T>
typename Light<T>::properties create_light(const unsigned int number, const std::vector<T>& position,
    const std::vector<T>& diffuse, const std::vector<T>& specular) {
  typename Light<T>::properties tmp;
  tmp[get_position_name(number)] = position;
  tmp[get_diffuse_name(number)] = diffuse;
  tmp[get_specular_name(number)] = specular;
  return tmp;
}

/*
 * http://www.xsi-blog.com/archives/115
def pointsOnSphere(N):
    N = float(N) # in case we got an int which we surely got
    pts = []

    inc = math.pi * (3 - math.sqrt(5))
    off = 2 / N
    for k in range(0, N):
        y = k * off - 1 + (off / 2)
        r = math.sqrt(1 - y*y)
        phi = k * inc
        pts.append([math.cos(phi)*r, y, math.sin(phi)*r])

    return pts
*/

template<typename T>
std::vector<typename Light<T>::properties> create_light_sphere(float radius = 10, unsigned int num_lights = 10) {
  std::vector<typename Light<T>::properties> tmp(num_lights);
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
    tmp.at(i) = create_light(i, std::move(position), default_light_property, default_light_property);
  }
  return tmp;
}

template<typename T, int dim>
std::vector<T> create_vector_from_array(const T (&array)[dim]) {
  std::vector<T> tmp(array, array + dim);
  return tmp;
}

template<typename T, int dim>
std::vector<typename Light<T>::properties> create_lights_from_array(const T light_props[][dim], const unsigned int count) {
  std::vector<typename Light<T>::properties> tmp;
  for (unsigned int i = 0; i < NUM_PROPERTIES * count; i += NUM_PROPERTIES) {
    std::vector<T> position = create_vector_from_array(light_props[i + 0]);
    std::vector<T> diffuse = create_vector_from_array(light_props[i + 1]);
    std::vector<T> specular = create_vector_from_array(light_props[i + 2]);

    typename Light<T>::properties props = create_light(i / NUM_PROPERTIES, std::move(position), std::move(diffuse), std::move(specular));
    tmp.push_back(std::move(props));
  }
  return tmp;
}

template<typename T>
void print_lights(const std::vector<typename Light<T>::properties> &lights) {
  unsigned int size = lights.size();
  std::cout << "number of lights: " << size << std::endl;
  unsigned int i = 0;
  for (auto iter = lights.begin(); iter != lights.end(); iter++, i++) {
    typename Light<T>::properties prop = *iter;
    for (auto iter_prop = prop.begin(); iter_prop != prop.end(); iter_prop++) {
      std::string name = iter_prop->first;
      std::vector<T> value = iter_prop->second;
      std::cout << "light " << i << ", property: " << name << ", value: ";
      for (auto iter_val = value.begin(); iter_val != value.end(); iter_val++)
        std::cout << *iter_val << ", ";
      std::cout << std::endl;
    }
  }
}

template<typename T>
void print_lights(const std::vector<typename Light<T>::properties> &lights, const std::vector<T> &ambient) {
  std::cout << "ambient illumination: ";
  for (auto iter = ambient.begin(); iter != ambient.end(); iter++)
    std::cout << *iter << ", ";
  std::cout << std::endl;
  print_lights<T>(lights);
}

template<typename T>
void setUniforms(GLuint programm_id, const std::vector<T> &ambient_color, const std::vector<typename Light<T>::properties> &lights) {
  GLint uniform_light_property = glGetUniformLocation(programm_id, "ambient_color");
  glUniform4f(uniform_light_property, ambient_color.at(0), ambient_color.at(1), ambient_color.at(2), ambient_color.at(3));
  if (uniform_light_property == -1)
    std::cout << "uniform handle is -1 with uniform name " << "ambient_color" << std::endl;

  for (auto iter_lights = lights.begin(); iter_lights != lights.end(); iter_lights++) {
    for (auto iter_properties = iter_lights->begin(); iter_properties != iter_lights->end(); iter_properties++) {
      GLint uniform_light_property = glGetUniformLocation(programm_id, iter_properties->first.c_str());
      auto value = iter_properties->second;
      glUniform4f(uniform_light_property, value.at(0), value.at(1), value.at(2), value.at(3));

      if (uniform_light_property == -1)
        std::cout << "uniform handle is -1 with uniform name " << iter_properties->first << std::endl;
    }
  }
}

template<typename T>
void setUniforms(GLuint programm_id, const Lights<T> &lights) {
  GLint uniform_light_property = glGetUniformLocation(programm_id, "ambient_color");
  const std::vector<T>& ambient_color = lights.ambient;
  glUniform4f(uniform_light_property, ambient_color.at(0), ambient_color.at(1), ambient_color.at(2), ambient_color.at(3));
  if (uniform_light_property == -1)
    std::cout << "uniform handle is -1 with uniform name " << "ambient_color" << std::endl;

  for (auto iter_lights : lights.lights) {
    for (auto iter_properties : iter_lights->props) {
      GLint uniform_light_property = glGetUniformLocation(programm_id, iter_properties->first.c_str());
      auto value = iter_properties->second;
      glUniform4f(uniform_light_property, value.at(0), value.at(1), value.at(2), value.at(3));

      if (uniform_light_property == -1)
        std::cout << "uniform handle is -1 with uniform name " << iter_properties->first << std::endl;
    }
  }
}

#endif /* LIGHTS_H_ */
