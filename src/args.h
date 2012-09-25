#ifndef __arguments_h__
#define __arguments_h__

typedef struct arguments {
  std::string mesh_filename;
  std::string texture_filename;
  std::string image_filename;
  float scale;
  float rotation[3];
  float translation[3];
  bool optimize;
  bool single_pass;
  arguments() : mesh_filename(""), texture_filename(""), image_filename(""), scale(1), rotation{0}, translation{0}, optimize(true), single_pass(false) {}
} arguments;

#endif /* __arguments_h__ */