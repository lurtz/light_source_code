#ifndef __OPENGL_H__
#define __OPENGL_H__

#include <GL/gl.h>
#include "MeshObj.h"

void setupOpenGL(int * argc, char ** argv, MeshObj const * const meshobj);
// starts the glutMainLoop
void run();

#endif // __OPENGL_H__
