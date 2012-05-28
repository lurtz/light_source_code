#include "MeshObj.h"
#include <iostream>
#include <limits>
#include <cmath>

MeshObj::MeshObj(float const * const rotation, float const * const translation, const float scale)
  : _scale(scale) {
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
