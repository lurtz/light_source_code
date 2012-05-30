#version 120

varying vec3 vert_light_dir;
varying vec3 vert_norm_dir;

// atributes from vertex array //
attribute vec3 vertex_OS;
attribute vec3 normal_OS;

void main() {
    vec4 pos = vec4(vertex_OS, 1);
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * pos;

    vert_light_dir = (gl_ModelViewMatrix * pos).xyz;
    vert_norm_dir = gl_NormalMatrix * normal_OS;
}
