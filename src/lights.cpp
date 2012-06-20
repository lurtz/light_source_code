#include "lights.h"

std::string get_name(const unsigned int number, std::string item) {
  std::stringstream name;
  name << "lights[" << number << "]." << item;
  return name.str();
}

std::string get_position_name(const unsigned int number) {
  return get_name(number, "position");
}

std::string get_diffuse_name(const unsigned int number) {
  return get_name(number, "diffuse");
}

std::string get_specular_name(const unsigned int number) {
  return get_name(number, "specular");
}
