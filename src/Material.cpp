#include "Material.h"

Material::Material() {
  mShaderProgram = NULL;
  setAmbientColor();
  setDiffuseColor();
  setEmissiveColor();
  setSpecularColor();
  mUniformsInitialized = false;
  for (unsigned int i = 0; i < MATERIAL_COMPONENT_COUNT; ++i) {
    mMaterialComponentEnabled[i] = false;
  }
}

Material::~Material() {
  // do not delete mShaderProgram, since it may be used by other materials //
}

void Material::setMaterialComponent(unsigned int comp, GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
  if (comp < MATERIAL_COMPONENT_COUNT) {
    MaterialComponent &mat = mMaterialComponent[comp];
    mMaterialComponentEnabled[comp] = true;
    mat.isEnabled = true;
    mat.value[0] = r;
    mat.value[1] = g;
    mat.value[2] = b;
    mat.value[3] = a;
    mUniformsInitialized = false;
  }
}

void Material::setMaterialComponentUniformNames(unsigned int comp, const char *uniformLocationName, const char *uniformIsEnabledName) {
  if (comp < MATERIAL_COMPONENT_COUNT) {
    mMaterialComponent[comp].uniformLocationName = std::string(uniformLocationName);
    mMaterialComponent[comp].uniformIsEnabledName = std::string(uniformIsEnabledName);
    if (mMaterialComponent[comp].uniformIsEnabledName.size() == 0) {
      std::stringstream sstr;
      sstr << mMaterialComponent[comp].uniformLocationName << "Enabled";
      mMaterialComponent[comp].uniformIsEnabledName = sstr.str();
    }
    mUniformsInitialized = false;
  }
}
    
void Material::setMaterialTexture(unsigned int layer, const char *textureFile) {
  if (layer < TEX_LAYER_COUNT) {
    if (loadTextureData(textureFile, mTexture[layer])) {
      if (layer < NORMAL_TEX) {
        mMaterialComponentEnabled[layer + 1] = true;
      }
      mUniformsInitialized = false;
    }
  }
}

void Material::setMaterialTextureUniformNames(unsigned int layer, const char *uniformLocationName, const char *uniformIsEnabledName) {
  if (layer < TEX_LAYER_COUNT) {
    mTexture[layer].uniformLocationName = std::string(uniformLocationName);
    mTexture[layer].uniformIsEnabledName = std::string(uniformIsEnabledName);
    if (mTexture[layer].uniformIsEnabledName.size() == 0) {
      std::stringstream sstr;
      sstr << mTexture[layer].uniformLocationName << "Enabled";
      mTexture[layer].uniformIsEnabledName = sstr.str();
    }
    mUniformsInitialized = false;
  }
}

void Material::setAmbientColor(GLfloat r, GLfloat g, GLfloat b) {
  setMaterialComponent(AMBIENT_COMP, r, g, b);
}

void Material::setDiffuseColor(GLfloat r, GLfloat g, GLfloat b) {
  setMaterialComponent(DIFFUSE_COMP, r, g, b);
}

void Material::setEmissiveColor(GLfloat r, GLfloat g, GLfloat b) {
  setMaterialComponent(EMISSIVE_COMP, r, g, b);
}

void Material::setSpecularColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
  setMaterialComponent(SPECULAR_COMP, r, g, b, a);
}

void Material::setDiffuseTexture(const char *filename) {
  setMaterialTexture(DIFFUSE_TEX, filename);
}

void Material::setSpecularTexture(const char *filename) {
  setMaterialTexture(SPECULAR_TEX, filename);
}

void Material::setEmissiveTexture(const char *filename) {
  setMaterialTexture(EMISSIVE_TEX, filename);
}

void Material::setNormalTexture(const char *filename) {
  setMaterialTexture(NORMAL_TEX, filename);
}


void Material::setShaderProgram(Shader *shaderProgram) {
  mShaderProgram = shaderProgram;
  std::cerr << "(Material::setShaderProgram) - Set new shaderprogram (ID: " << mShaderProgram->getProgramID() << ")" << std::endl;
  
  // init default uniform locations //
  setMaterialComponentUniformNames(AMBIENT_COMP, "ambientColor");
  setMaterialComponentUniformNames(DIFFUSE_COMP, "diffuseColor");
  setMaterialComponentUniformNames(SPECULAR_COMP, "specularColor");
  setMaterialComponentUniformNames(EMISSIVE_COMP, "emissiveColor");
  setMaterialTextureUniformNames(DIFFUSE_TEX, "diffuseTex");
  setMaterialTextureUniformNames(SPECULAR_TEX, "specularTex");
  setMaterialTextureUniformNames(EMISSIVE_TEX, "emissiveTex");
  setMaterialTextureUniformNames(NORMAL_TEX, "normalTex");
}

Shader *Material::getShaderProgram() {
  return mShaderProgram;
}

void Material::initUniforms() {
  if (mShaderProgram) {
    // init active components //
    for (unsigned int comp = 0; comp < MATERIAL_COMPONENT_COUNT; ++comp) {
      // init uniform locations for material component //
      if (mMaterialComponent[comp].uniformLocationName.size() > 0) {
        // if name for uniform is defined -> try to get uniform location //
        mMaterialComponent[comp].uniformLocation = mShaderProgram->getUniformLocation(mMaterialComponent[comp].uniformLocationName.c_str());
        if (mMaterialComponent[comp].uniformLocation < 0) {
          std::cerr << "(Material::initUniforms) - Could not access uniform location for material component \"" << mMaterialComponent[comp].uniformLocationName << "\"" <<  std::endl;
        }
        mMaterialComponent[comp].uniformIsEnabled = mShaderProgram->getUniformLocation(mMaterialComponent[comp].uniformIsEnabledName.c_str());
        if (mMaterialComponent[comp].uniformIsEnabled < 0) {
          std::cerr << "(Material::initUniforms) - Could not access uniform location for material component enable \"" << mMaterialComponent[comp].uniformIsEnabledName << "\"" <<  std::endl;
        }
      }
    }
    
    for (unsigned int texLayer = 0; texLayer < TEX_LAYER_COUNT; ++texLayer) {
      if (mTexture[texLayer].uniformLocationName.size() > 0) {
        mTexture[texLayer].uniformLocation = mShaderProgram->getUniformLocation(mTexture[texLayer].uniformLocationName.c_str());
        if (mTexture[texLayer].uniformLocation < 0) {
          std::cerr << "(Material::initUniforms) - Could not access uniform location for texture \"" << mTexture[texLayer].uniformLocationName << "\"" <<  std::endl;
        }
        mTexture[texLayer].uniformIsEnabled = mShaderProgram->getUniformLocation(mTexture[texLayer].uniformIsEnabledName.c_str());
        if (mTexture[texLayer].uniformIsEnabled < 0) {
          std::cerr << "(Material::initUniforms) - Could not access uniform location for texture enable \"" << mTexture[texLayer].uniformIsEnabledName << "\"" <<  std::endl;
        }
      }
    }

    mUniformsInitialized = true;
  }
}

void Material::enable() {
  if (mShaderProgram != NULL && mShaderProgram->ready()) {
    mShaderProgram->enable();
    if (!mUniformsInitialized) {
      initUniforms();
    }
  } else {
    return;
  }
  
  for (unsigned int comp = 0; comp < MATERIAL_COMPONENT_COUNT; ++comp) {
    MaterialComponent &mat = mMaterialComponent[comp];
    if (mMaterialComponentEnabled[comp]) {
      // enable material component, if trigger uniform is defined //
      if (mat.uniformIsEnabled >= 0) {
        glUniform1i(mat.uniformIsEnabled, 1);
      }
      // link material value //
      glUniform4f(mat.uniformLocation, mat.value[0], mat.value[1], mat.value[2], mat.value[3]);
    } else {
      // disable material component, if trigger uniform is defined //
      if (mat.uniformIsEnabled >= 0) {
        glUniform1i(mat.uniformIsEnabled, 0);
      }
    }
  }
  
  for (unsigned int layer = 0; layer < TEX_LAYER_COUNT; ++layer) {
    if (mTexture[layer].isEnabled) {
      // enable texture, if trigger uniform defined //
      if (mTexture[layer].uniformIsEnabled >= 0) {
        glUniform1i(mTexture[layer].uniformIsEnabled, 1);
      }
      // link texture //
      glActiveTexture(GL_TEXTURE0 + layer);
      glBindTexture(GL_TEXTURE_2D, mTexture[layer].glTextureLocation);
      glUniform1i(mTexture[layer].uniformLocation, layer);
    } else {
      // disable texture, if trigger uniform defined  //
      if (mTexture[layer].uniformIsEnabled >= 0) {
        glUniform1i(mTexture[layer].uniformIsEnabled, 0);
      }
    }
  }
}

void Material::disable() {
  if (mShaderProgram != NULL) {
    mShaderProgram->disable();
  }
  for (unsigned int layer = 0; layer < TEX_LAYER_COUNT; ++layer) {
    glActiveTexture(GL_TEXTURE0 + layer);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
}

bool Material::loadTextureData(const char *textureFile, Texture &texture) {
  IplImage *image = cvLoadImage(textureFile, CV_LOAD_IMAGE_COLOR);
  if (image != NULL) {
    texture.width = image->width;
    texture.height = image->height;
    texture.data = new unsigned char[image->imageSize];
    if (image->origin == 0) {
      // flip rows of the image from top to bottom //
      cvFlip(image);
    }
    memcpy(texture.data, image->imageData, image->imageSize);
    std::cout << "(Material::loadTextureData) - imported \"" << textureFile << "\" (" << texture.width << ", " << texture.height << ")" << std::endl;
    
    glEnable(GL_TEXTURE_2D);
    
    // init openGL texture //
    if (texture.glTextureLocation <= 0) {
      // texture has not been generated yet //
      glGenTextures(1, &texture.glTextureLocation);
    }
    glBindTexture(GL_TEXTURE_2D, texture.glTextureLocation);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, texture.width, texture.height, GL_BGR, GL_UNSIGNED_BYTE, texture.data);
    
    glDisable(GL_TEXTURE_2D);
    
    texture.isEnabled = true;
    
    mUniformsInitialized = false;
  }
  cvReleaseImage(&image);
  return texture.isEnabled;
}
