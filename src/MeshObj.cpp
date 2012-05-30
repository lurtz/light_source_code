#include "MeshObj.h"
#include <iostream>
#include <limits>
#include <cmath>

#define OFFSET(i) ((char*)NULL + (i))

MeshObj::MeshObj(float const * const rotation, float const * const translation, const float scale)
  : mMaterial(0), mVBO(0), mIBO(0), mIndexCount(0), mShadowVBO(0), mShadowIBO(0), mShadowIndexCount(0), _scale(scale) {
  for (int i = 0; i < 3; ++i) {
    mMinBounds[i] = std::numeric_limits<float>::max();
    mMaxBounds[i] = std::numeric_limits<float>::min();
    if (rotation != 0)
      _rotation[i] = rotation[i];
    else
      _rotation[i] = 0;
    if (translation != 0)
      _translation[i] = translation[i];
    else
      _translation[i] = 0;
  }
}

MeshObj::~MeshObj() {
}

void MeshObj::setData(const std::vector<Vertex> &vertexData, const std::vector<unsigned int> &indexData) {
  // compute bounds //
  for (int i = 0; i < 3; ++i) {
    mMinBounds[i] = std::numeric_limits<float>::max();
    mMaxBounds[i] = std::numeric_limits<float>::min();
  }
  for (size_t i = 0; i < vertexData.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      if (vertexData[i].position[j] < mMinBounds[j]) mMinBounds[j] = vertexData[i].position[j];
      if (vertexData[i].position[j] > mMaxBounds[j]) mMaxBounds[j] = vertexData[i].position[j];
    }
  }
  
  // save local copy of vertex data (needed for shadow volume computation) //
  mVertexData.assign(vertexData.begin(), vertexData.end());
  mIndexData.assign(indexData.begin(), indexData.end());
  mIndexCount = indexData.size();

  // init and bind a VBO (vertex buffer object) //
  if (mVBO == 0) {
    glGenBuffers(1, &mVBO);
  }
  glBindBuffer(GL_ARRAY_BUFFER, mVBO);
  // copy data into the VBO //
  glBufferData(GL_ARRAY_BUFFER, mVertexData.size() * sizeof(Vertex), &mVertexData[0], GL_STATIC_DRAW);

  // init and bind a IBO (index buffer object) //
  if (mIBO == 0) {
    glGenBuffers(1, &mIBO);
  }
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
  // copy data into the IBO //
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, mIndexCount * sizeof(GLint), &mIndexData[0], GL_STATIC_DRAW);

  // unbind buffers //
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void MeshObj::setMaterial(Material *material) {
  mMaterial = material;
}

void MeshObj::setUniforms(GLuint programm_id) {
//  mMaterial->enable();

  float innerAngle = 15.0f;
  float outerAngle = 20.0f;
  GLint uniform_innerSpotAngle = glGetUniformLocation(programm_id, "uni_innerSpotAngle");
  GLint uniform_outerSpotAngle = glGetUniformLocation(programm_id, "uni_outerSpotAngle");
  glUniform1f(uniform_innerSpotAngle, innerAngle);
  glUniform1f(uniform_outerSpotAngle, outerAngle);

  // position, ambient, diffuse, specular in vec4
  float light_properties[] = {
		    1, 1, 0, 0, 1, 1, 1, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0,
		    3, 1, 0, 0, 1, 1, 1, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0,
		    3, 3, 0, 0, 1, 1, 1, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0
  };

  GLint uniform_lights = glGetUniformLocation(programm_id, "lights");
  glUniform4fv(uniform_lights, 12, light_properties);

//  mMaterial->disable();
}

void MeshObj::render(void) {
  if (mMaterial != NULL) {
    mMaterial->enable();
  }

  if (mVBO != 0) {
    // init vertex attribute arrays //
    glBindBuffer(GL_ARRAY_BUFFER, mVBO);

    GLuint programm_id = mMaterial->getShaderProgram()->getProgramID();

    setUniforms(programm_id);

    GLint vertexLoc = glGetAttribLocation(programm_id, "vertex_OS");
    GLint normalLoc = glGetAttribLocation(programm_id, "normal_OS");

    glEnableVertexAttribArray(vertexLoc);
    glVertexAttribPointer(vertexLoc, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), OFFSET(0));
    glEnableVertexAttribArray(normalLoc);
    glVertexAttribPointer(normalLoc, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), OFFSET(12));

    // bind the index buffer object mIBO here //
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);

    // render VBO as triangles //
    glDrawElements(GL_TRIANGLES, mIndexCount, GL_UNSIGNED_INT, (void*)0);

    // unbind the buffers //
    glDisableVertexAttribArray(vertexLoc);
    glDisableVertexAttribArray(normalLoc);

    // unbind the element render buffer //
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    // unbind the vertex array buffer //
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  if (mMaterial != NULL) {
    mMaterial->disable();
  }
}

float MeshObj::getWidth(void) {
  return mMaxBounds[0] - mMinBounds[0];
}

float MeshObj::getHeight(void) {
  return mMaxBounds[1] - mMinBounds[1]; 
}

float MeshObj::getDepth(void) {
  return mMaxBounds[2] - mMinBounds[2]; 
}

void MeshObj::rotate(float rotation[3]) {
  for (unsigned int i = 0; i < 3; i++)
    _rotation[i] = rotation[i];
}

void MeshObj::translate(float translation[3]) {
  for (unsigned int i = 0; i < 3; i++)
    _translation[i] = translation[i];
}

void MeshObj::scale(float scale) {
  _scale = scale;
}
