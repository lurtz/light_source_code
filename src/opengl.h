#ifndef __OPENGL_H__
#define __OPENGL_H__

#include "MeshObj.h"
#include "args.h"

void setupOpenGL(int * argc, char ** argv, const arguments &args);
// starts the glutMainLoop
void run(MeshObj * const meshobj);

#endif // __OPENGL_H__
