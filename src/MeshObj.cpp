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

template<typename T>
struct Light {
  typedef std::map<std::string, std::vector<T> > properties;
};

template<typename T>
typename Light<T>::properties create_light(const unsigned int number, const std::vector<T>& position, const std::vector<T>& ambient, const std::vector<T>& diffuse, const std::vector<T>& specular) {
  typename Light<T>::properties tmp;
  std::stringstream name;
  name << "lights[" << number << "]." << "position";
  tmp[name.str()] = position;
  name.str("");
  name << "lights[" << number << "]." << "ambient";
  tmp[name.str()] =  ambient;
  name.str("");
  name << "lights[" << number << "]." << "diffuse";
  tmp[name.str()] = diffuse;
  name.str("");
  name << "lights[" << number << "]." << "specular";
  tmp[name.str()] = specular;
  return tmp;
}

template<typename T, int dim>
std::vector<T> create_vector_from_array(const T (&array)[dim]) {
  std::vector<T> tmp(array, array+dim);
  return tmp;
}

template<typename T, int dim>
std::vector<typename Light<T>::properties> create_lights(const T light_props[][dim], const unsigned int count) {
  std::vector<typename Light<T>::properties> tmp;
  for (unsigned int i = 0; i < 4*count; i+=4) {
	  std::vector<T> position = create_vector_from_array(light_props[i+0]);
	  std::vector<T> ambient = create_vector_from_array(light_props[i+1]);
	  std::vector<T> diffuse = create_vector_from_array(light_props[i+2]);
	  std::vector<T> specular = create_vector_from_array(light_props[i+3]);

	  typename Light<T>::properties props = create_light(i/4, position, ambient, diffuse, specular);
	  tmp.push_back(props);
  }
  return tmp;
}

template<typename T>
void prints_lights(const std::vector<typename Light<T>::properties> &lights) {
	unsigned int size = lights.size();
	  std::cout << size << std::endl;
	  for (auto iter = lights.begin(); iter != lights.end(); iter++) {
		  typename Light<T>::properties prop = *iter;
		  for (auto iter_prop = prop.begin(); iter_prop != prop.end(); iter_prop++) {
			  std::string name = iter_prop->first;
			  std::vector<T> value = iter_prop->second;
			  std::cout
			    << "light property name: "
			    << name
			    << ", value: ";
			  for (auto iter_val = value.begin(); iter_val != value.end(); iter_val++)
			    std::cout << *iter_val << ", ";
			  std::cout << std::endl;
		  }
	  }
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
  float light_properties[][4] = {
		    {1, 1, 0, 0}, {1, 1, 1, 0}, {0.5, 0.5, 0.5, 0}, {0, 0, 0, 0},
		    {3, 1, 0, 0}, {1, 1, 1, 0}, {0.5, 0.5, 0.5, 0}, {0, 0, 0, 0},
		    {3, 3, 0, 0}, {1, 1, 1, 0}, {0.5, 0.5, 0.5, 0}, {0, 0, 0, 0}
  };

  std::vector<Light<float>::properties> lights = create_lights(light_properties, 3);
//  prints_lights<float>(lights);

  for (auto iter_lights = lights.begin(); iter_lights != lights.end(); iter_lights++) {
    for (auto iter_properties = iter_lights->begin(); iter_properties != iter_lights->end(); iter_properties++) {
      std::stringstream name;
      GLint uniform_light_property = glGetUniformLocation(programm_id, iter_properties->first.c_str());
      auto value = iter_properties->second;
      glUniform4f(uniform_light_property, value[0], value[1], value[2], value[3]);

      if (uniform_light_property == -1)
    	  std::cout << "uniform handle is -1 with uniform name " << name.str() << std::endl;
    }
  }

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
