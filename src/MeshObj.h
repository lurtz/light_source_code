#ifndef __MESH_OBJ__
#define __MESH_OBJ__

#include <GL/glew.h>
#include <GL/glut.h>

#include <vector>
#include "Material.h"
#include "lights.h"

struct Vertex {
  Vertex(float x = 0, float y = 0, float z = 0, float nx = 1, float ny = 0, float nz = 0, float tx = 0, float ty = 0) {
    position[0] = x;
    position[1] = y;
    position[2] = z;
    normal[0] = nx;
    normal[1] = ny;
    normal[2] = nz;
    texcoord[0] = tx;
    texcoord[1] = ty;
    tangent[0] = 0;
    tangent[1] = 0;
    tangent[2] = 0;
    bitangent[0] = 0;
    bitangent[1] = 0;
    bitangent[2] = 0;
  }
  float position[3];
  float normal[3];
  float texcoord[2];
  float tangent[3];
  float bitangent[3];
};

class MeshObj {
  public:
    MeshObj(Lights<float> const * const lights = nullptr, float const * const rotation = nullptr, float const * const translation = nullptr, const float scale = 1);
    ~MeshObj();
    
    void setData(const std::vector<Vertex> &vertexData, const std::vector<unsigned int> &indexData);
    void setMaterial(Material *material);
    
    void render(void);

    float getWidth(void);
    float getHeight(void);
    float getDepth(void);

    void rotate(float rotation[3]);
    void translate(float translation[3]);
    void scale(float scale);
    void setLight(const Lights<float>& lights);
  private:
    std::vector<Vertex> mVertexData;
    std::vector<unsigned int> mIndexData;
    Material *mMaterial;
    
    GLuint mVBO;
    GLuint mIBO;
    GLuint mIndexCount;

    GLuint mShadowVBO;
    GLuint mShadowIBO;
    GLuint mShadowIndexCount;

    float mMinBounds[3];
    float mMaxBounds[3];

    float _rotation[3];
    float _translation[3];
    float _scale;

    Lights<float> const *_lights;
};

#endif
