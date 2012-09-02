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
    Shader(const std::string &vertShaderFile, const std::string &fragShaderFile);
    ~Shader();
    
    bool loadVertShader(const std::string &shaderFile);
    bool loadFragShader(const std::string &shaderFile);
    
    GLuint loadShaderCode(const std::string &fileName, GLenum shaderType);
    
    void link();
    bool ready();
    bool enabled();
    
    void enable();
    void disable();
    
    GLuint getProgramID();
    GLint getUniformLocation(const std::string &uniformName);

  private:
    char* loadShaderSource(const std::string &fileName);
    bool mEnabled;
    
    GLuint mVertShaderID;
    GLuint mFragShaderID;
    GLuint mShaderProgramID;
};

#endif
