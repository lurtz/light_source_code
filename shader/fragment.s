#version 120

// normale des pixels
varying vec3 vert_norm_dir;

uniform float uni_outerSpotAngle;
uniform float uni_innerSpotAngle;

struct Light_properties {
  vec4 position;
  vec4 ambient;
  vec4 diffuse;
  vec4 specular;
};
#define MAX_LIGHTS 3
uniform Light_properties lights[MAX_LIGHTS];
varying vec4 light_positions[MAX_LIGHTS];

void main () {
    // normalize everything necessary //
    vec3 N = normalize(vert_norm_dir);

    // diffuse component //
    float NdotL = 0;
    for (int i = 0; i < MAX_LIGHTS; i++) {
      vec3 L = normalize(light_positions[i].xyz);
      NdotL += min(max(0.0, dot(-N, L)), 1.0);
    }
    gl_FragColor = vec4(1,1,1,0);
    gl_FragColor *= min(NdotL/MAX_LIGHTS, 1);

    float amb_val = 0.1;
    gl_FragColor += vec4(amb_val, amb_val, amb_val, 0.0);
}
