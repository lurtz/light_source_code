#ifndef __MATERIAL__
#define __MATERIAL__

#include <string>

#include "Shader.h"

struct Texture {
  Texture() {
    data = nullptr;
    width = 0;
    height = 0;
    glTextureLocation = 0;
    
    uniformLocationName = "";
    uniformLocation = -1;
    
    isEnabled = false;
    uniformIsEnabledName = "";
    uniformIsEnabled = -1;
  };
  
  ~Texture() {
    if (data != nullptr) {
      delete [] data;
    }
  };
  
  unsigned char *data;
  unsigned int width, height;
  GLuint glTextureLocation;
  
  std::string uniformLocationName;
  GLint uniformLocation;
  
  bool isEnabled;
  std::string uniformIsEnabledName;
  GLint uniformIsEnabled;
};

struct MaterialComponent {
  MaterialComponent() {
    for (unsigned int i = 0; i < 4; ++i) {
      value[i] = 0;
    }
    
    uniformLocationName = "";
    uniformLocation = -1;
    
    isEnabled = false;
    uniformIsEnabledName = "";
    uniformIsEnabled = -1;
  };
  
  GLfloat value[4];
  
  std::string uniformLocationName;
  GLint uniformLocation;
  
  bool isEnabled;
  std::string uniformIsEnabledName;
  GLint uniformIsEnabled;
};

class Material {
  public:
    Material();
    ~Material();
    
    void disableMaterialComponent(unsigned int comp);
    void setMaterialComponent(unsigned int comp, GLfloat r = -1, GLfloat g = 0, GLfloat b = 0, GLfloat a = 1);
    void setMaterialComponentUniformNames(unsigned int comp, const std::string &uniformLocationName, const std::string &uniformIsEnabledName = "");
    
    void setMaterialTexture(unsigned int layer, const std::string &textureFile);
    void setMaterialTextureUniformNames(unsigned int layer, const std::string &uniformLocationName, const std::string &uniformIsEnabledName = "");
    
    void setAmbientColor(GLfloat r = -1, GLfloat g = 0, GLfloat b = 0);
    void setDiffuseColor(GLfloat r = -1, GLfloat g = 0, GLfloat b = 0);
    void setEmissiveColor(GLfloat r = -1, GLfloat g = 0, GLfloat b = 0);
    void setSpecularColor(GLfloat r = -1, GLfloat g = 0, GLfloat b = 0, GLfloat a = 0);
    
    void setDiffuseTexture(const std::string &filename);
    void setSpecularTexture(const std::string &filename);
    void setEmissiveTexture(const std::string &filename);
    void setNormalTexture(const std::string &filename);
        
    void setShaderProgram(Shader *shader);
    Shader *getShaderProgram();
    
    void enable();
    void disable();
    
  private:
    bool loadTextureData(const std::string &textureFile, Texture &texture);
    void initUniforms();
    bool mUniformsInitialized;
    
    enum MaterialComponents {AMBIENT_COMP = 0, DIFFUSE_COMP, SPECULAR_COMP, EMISSIVE_COMP, MATERIAL_COMPONENT_COUNT};
    bool mMaterialComponentEnabled[MATERIAL_COMPONENT_COUNT];
    MaterialComponent mMaterialComponent[MATERIAL_COMPONENT_COUNT];

    enum TextureLayer {DIFFUSE_TEX = 0, SPECULAR_TEX, EMISSIVE_TEX, NORMAL_TEX, TEX_LAYER_COUNT};
    Texture mTexture[TEX_LAYER_COUNT];
    
    Shader *mShaderProgram;
};

#endif
