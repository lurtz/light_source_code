#ifndef __OPENGL_H__
#define __OPENGL_H__

#include <GL/gl.h>
#include "MeshObj.h"

void setupOpenGL(int * argc, char ** argv, const unsigned int width = 800, const unsigned int height = 600);
// starts the glutMainLoop
void run(const cv::Mat& original_image, MeshObj * const meshobj);

#endif // __OPENGL_H__
