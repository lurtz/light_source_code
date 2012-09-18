#ifndef __OPENGL_H__
#define __OPENGL_H__

#include "MeshObj.h"
#include <opencv2/core/core.hpp>

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

void setupOpenGL(int * argc, char ** argv, const arguments &args);
// starts the glutMainLoop
void run(MeshObj * const meshobj);

#endif // __OPENGL_H__
