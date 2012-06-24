#version 120

varying vec3 vert_norm_dir;
varying vec3 eyeVec;

// atributes from vertex array //
attribute vec3 vertex_OS;
attribute vec3 normal_OS;

struct Light_properties {
  vec4 position;
  vec4 diffuse;
  vec4 specular;
};
#define MAX_LIGHTS 3
uniform Light_properties lights[MAX_LIGHTS];
varying vec3 vertex_to_light[MAX_LIGHTS];

void main() {
    vec4 pos = vec4(vertex_OS, 1);
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * pos;

    vert_norm_dir = gl_NormalMatrix * normal_OS;

    vec3 vVertex = vec3(gl_ModelViewMatrix * pos);
    eyeVec = -vVertex;
    for (int i = 0; i < MAX_LIGHTS; i++) {
      vertex_to_light[i] = vec3((gl_ModelViewMatrix * lights[i].position).xyz - vVertex);
    }


}
