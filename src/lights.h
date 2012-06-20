#ifndef LIGHTS_H_
#define LIGHTS_H_

#include <GL/glew.h>
#include <GL/glut.h>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>

// position, ambient, diffuse, specular in vec4
const unsigned int NUM_PROPERTIES = 3;
const float light_properties[][4] = {
    { 0, -5, 0, 0}, {0.5, 0.5, 0.5, 0}, {1, 0, 0, 0},
    {-6,  1, 0, 0}, {0.5, 0.5, 0.5, 0}, {0, 1, 0, 0},
    { 3,  0, 0, 0}, {0.5, 0.5, 0.5, 0}, {0, 0, 1, 0}
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
};

template<typename T>
typename Light<T>::properties create_light(const unsigned int number, const std::vector<T>& position,
    const std::vector<T>& diffuse, const std::vector<T>& specular) {
  typename Light<T>::properties tmp;
  std::stringstream name;
  tmp[get_position_name(number)] = position;
  tmp[get_diffuse_name(number)] = diffuse;
  tmp[get_specular_name(number)] = specular;
  return tmp;
}

template<typename T, int dim>
std::vector<T> create_vector_from_array(const T (&array)[dim]) {
  std::vector<T> tmp(array, array + dim);
  return tmp;
}

template<typename T, int dim>
std::vector<typename Light<T>::properties> create_lights(const T light_props[][dim], const unsigned int count) {
  std::vector<typename Light<T>::properties> tmp;
  for (unsigned int i = 0; i < NUM_PROPERTIES * count; i += NUM_PROPERTIES) {
    std::vector<T> position = create_vector_from_array(light_props[i + 0]);
    std::vector<T> diffuse = create_vector_from_array(light_props[i + 1]);
    std::vector<T> specular = create_vector_from_array(light_props[i + 2]);

    typename Light<T>::properties props = create_light(i / NUM_PROPERTIES, position, diffuse, specular);
    tmp.push_back(props);
  }
  return tmp;
}

template<typename T>
void prints_lights(const std::vector<typename Light<T>::properties> &lights) {
  unsigned int size = lights.size();
  std::cout << size << std::endl;
  for (auto iter = lights.begin(); iter != lights.end(); iter++) {
    typename Light<T>::properties prop = *iter;
    for (auto iter_prop = prop.begin(); iter_prop != prop.end(); iter_prop++) {
      std::string name = iter_prop->first;
      std::vector<T> value = iter_prop->second;
      std::cout << "light property name: " << name << ", value: ";
      for (auto iter_val = value.begin(); iter_val != value.end(); iter_val++)
        std::cout << *iter_val << ", ";
      std::cout << std::endl;
    }
  }
}

template<typename T>
void setUniforms(GLuint programm_id, const std::vector<T> ambient_color, const std::vector<typename Light<T>::properties> &lights) {
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

#endif /* LIGHTS_H_ */
