#ifndef __arguments_h__
#define __arguments_h__

#include <array>

typedef struct arguments {
  std::string mesh_filename;
  std::string texture_filename;
  std::string image_filename;
  float scale;
  std::array<float, 3> rotation;
  std::array<float, 3> translation;
  bool optimize;
  bool single_pass;
  std::array<float, 3> camera_position;
  arguments() : mesh_filename{""}, texture_filename{""}, image_filename{""}, scale{1}, rotation{{0}}, translation{{0}}, optimize{true}, single_pass{false}, camera_position{{0}} {}
} arguments;

#endif /* __arguments_h__ */