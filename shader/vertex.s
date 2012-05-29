#version 120

varying vec3 vert_light_dir;
varying vec3 vert_norm_dir;

// atributes from vertex array //
attribute vec3 vertex_OS;
attribute vec3 normal_OS;

void main() {
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vec4(vertex_OS, 1);

    vert_light_dir = (gl_ModelViewMatrix * gl_Vertex).xyz;
    vert_norm_dir = gl_NormalMatrix * normal_OS;
}
