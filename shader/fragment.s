#version 120

// normale des pixels
varying vec3 vert_norm_dir;

uniform float uni_outerSpotAngle;
uniform float uni_innerSpotAngle;

float alpha = 50;
uniform vec4 ambient_color;
struct Light_properties {
  vec4 position;
  vec4 diffuse;
  vec4 specular;
};
#define MAX_LIGHTS 3
uniform Light_properties lights[MAX_LIGHTS];
varying vec4 light_positions[MAX_LIGHTS];

void main () {
    // normalize everything necessary //
    vec3 N = normalize(vert_norm_dir);
    vec3 V = vec3(0, 0, -1);

    // ambient component
    vec4 color = ambient_color;

    for (int i = 0; i < MAX_LIGHTS; i++) {
      // diffuse component
      vec3 L = normalize(light_positions[i].xyz);
      float NdotL = min(max(0.0, dot(-N, L)), 1.0);
      color += lights[i].diffuse * NdotL;

      // specular component
      vec3 R = normalize(-2*NdotL*N - L);
      float RdotV = min(max(0.0, dot(R, V)), 1.0);
      color += lights[i].specular * pow(RdotV, alpha);
    }

    // clamp components
    if (color.x > 1.0)
      color.x = 1.0;
    if (color.y > 1.0)
      color.y = 1.0;
    if (color.z > 1.0)
      color.z = 1.0;


    gl_FragData[0] = color;
    gl_FragData[1] = vec4(N, 0.0);
}
