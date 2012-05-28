#ifndef __SHADER__
#define __SHADER__

#include <GL/glew.h>
#include <GL/glut.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>

class Shader {
  public:
    Shader(const char* vertShaderFile, const char* fragShaderFile);
    ~Shader();
    
    bool loadVertShader(const char* shaderFile);
    bool loadFragShader(const char* shaderFile);
    
    GLuint loadShaderCode(const char* fileName, GLenum shaderType);
    
    void link();
    bool ready();
    bool enabled();
    
    void enable();
    void disable();
    
    GLuint getProgramID();
    GLint getUniformLocation(const char *uniformName);

  private:
    char* loadShaderSource(const char* fileName);
    bool mEnabled;
    
    GLuint mVertShaderID;
    GLuint mFragShaderID;
    GLuint mShaderProgramID;
};

#endif
