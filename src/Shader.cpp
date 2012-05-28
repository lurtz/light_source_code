#include "Shader.h"

Shader::Shader(const char* vertShaderFile, const char* fragShaderFile) {
  mVertShaderID = 0;
  mFragShaderID = 0;
  mShaderProgramID = 0;
  mEnabled = false;
  
  if (!loadVertShader(vertShaderFile)) {
    return;
  }
  if (!loadFragShader(fragShaderFile)) {
    return;
  }
  
  link();
}

Shader::~Shader() {
  glDeleteProgram(mShaderProgramID);
}

bool Shader::loadVertShader(const char* shaderFile) {
  mVertShaderID = loadShaderCode(shaderFile, GL_VERTEX_SHADER);
  if (mVertShaderID == 0) {
    std::cout << "(initShader) - Could not create vertex shader." << std::endl;
  }
  return mVertShaderID != 0;
}

bool Shader::loadFragShader(const char* shaderFile) {
  mFragShaderID = loadShaderCode(shaderFile, GL_FRAGMENT_SHADER);
  if (mFragShaderID == 0) {
    std::cout << "(initShader) - Could not create vertex shader." << std::endl;
  }
  return mFragShaderID != 0;
}

GLuint Shader::loadShaderCode(const char* fileName, GLenum shaderType) {
  GLuint shader = glCreateShader(shaderType);
  if (shader == 0) {
    return 0;
  }
  
  const char* shaderSrc = loadShaderSource(fileName);
  if (shaderSrc == NULL) return 0;
  glShaderSource(shader, 1, (const char**)&shaderSrc, NULL);
  
  glCompileShader(shader);
  
  int logMaxLength;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logMaxLength);
  char log[logMaxLength];
  int logLength = 0;
  glGetShaderInfoLog(shader, logMaxLength, &logLength, log);
  if (logLength > 0) {
    std::cout << "(Shader::loadShaderCode) - Compiler log:\n------------------\n" << log << "\n------------------" << std::endl;
  }
  
  return shader;
}

char* Shader::loadShaderSource(const char* fileName) {
  std::ifstream file(fileName, std::ios::in);
  if (file.good()) {
    unsigned long srcLength = 0;
    file.tellg();
    file.seekg(0, std::ios::end);
    srcLength = file.tellg();
    file.seekg(0, std::ios::beg);
    
    char *srcData = new char[srcLength + 1];
    srcData[srcLength] = 0;
    
    unsigned long i = 0;
    while (file.good()) {
      srcData[i] = file.get();
      if (!file.eof()) {
        ++i;
      }
    }
    file.close();
    srcData[i] = 0;
    
    // return shader source //
    return srcData;
  } else {
    std::cout << "(Shader::loadShaderSource) - Could not open file \"" << fileName << "\"." << std::endl;
    return NULL;
  }
}

void Shader::link() {
  if (mVertShaderID > 0 && mFragShaderID > 0) {
    mShaderProgramID = glCreateProgram();
    
    // loaded shaders seem to work -> attach them to program //
    glAttachShader(mShaderProgramID, mVertShaderID);
    glAttachShader(mShaderProgramID, mFragShaderID);
    
    // link final program //
    glLinkProgram(mShaderProgramID);
    
    // get log //
    int logMaxLength;
    glGetProgramiv(mShaderProgramID, GL_INFO_LOG_LENGTH, &logMaxLength);
    char log[logMaxLength];
    int logLength = 0;
    glGetShaderInfoLog(mShaderProgramID, logMaxLength, &logLength, log);
    if (logLength > 0) {
      std::cout << "(Shader::link) - Linker log:\n------------------\n" << log << "\n------------------" << std::endl;
    }
  } else {
    std::cerr << "(Shader::link) - Cannot link shaderprogram, since shaders aren't compiled properly." << std::endl;
  }
}

bool Shader::ready() {
  return mShaderProgramID > 0; 
}

bool Shader::enabled() {
  return mEnabled;
}

void Shader::enable() {
  if (ready()) {
    if (!enabled()) {
      glUseProgram(mShaderProgramID);
      mEnabled = true;
    }
  } else {
    std::cerr << "(Shader::enableShader) - Shaderprogram not ready!" << std::endl;
    mEnabled = false;
  }
}

void Shader::disable() {
  glUseProgram(0);
  mEnabled = false;
}

GLuint Shader::getProgramID() {
  return mShaderProgramID;
}

GLint Shader::getUniformLocation(const char *uniformName) {
  //std::cerr << "(Shader::getUniformLocation) - UniformLocation for \"" << uniformName << "\": " << glGetUniformLocation(mShaderProgramID, uniformName) << std::endl;
  return glGetUniformLocation(mShaderProgramID, uniformName);
}