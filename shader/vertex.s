#version 120

varying vec3 vert_norm_dir;
varying vec3 eyeVec;

// atributes from vertex array //
attribute vec3 vertex_OS;
attribute vec3 normal_OS;
attribute vec2 texCoord_OS;

void main() {
    gl_TexCoord[0] = gl_TextureMatrix[0] * vec4(texCoord_OS, 0, 1);
    vec4 pos = vec4(vertex_OS, 1);
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * pos;

    vert_norm_dir = gl_NormalMatrix * normal_OS;

    eyeVec = -vec3(gl_ModelViewMatrix * pos);
}
