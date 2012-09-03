#version 120

// normale des pixels
varying vec3 vert_norm_dir;
varying vec3 eyeVec;

float alpha = 50;
uniform vec4 ambient_color;
struct Light_properties {
  vec4 position;
  vec4 diffuse;
  vec4 specular;
};

// without solving equations I can get up to 140 lights with a Radeon HD6850 and Mesa
// my thinkpad seems not to like huge values, 30 is max
#define MAX_LIGHTS 30
uniform Light_properties lights[MAX_LIGHTS];

uniform bool diffuseTexEnabled;
uniform sampler2D diffuseTex;

void main () {
    // normalize everything necessary //
    vec3 N = normalize(vert_norm_dir);
    vec3 E = normalize(eyeVec);
    
    bool bla = diffuseTexEnabled;

    // ambient component
    vec4 color = ambient_color;

    for (int i = 0; i < MAX_LIGHTS; i++) {
      // diffuse component
      vec3 L = normalize((gl_ModelViewMatrix * lights[i].position).xyz + eyeVec);
      float NdotL = max(0.0, dot(N, L));
      color += lights[i].diffuse * NdotL;

      // specular component
      if (NdotL > 0.0) {
        vec3 R = reflect(-L, N);
        float RdotE = max(0.0, dot(R, E));
        color += lights[i].specular * pow(RdotE, alpha);
      }
      
      if (NdotL > 3 )
        color += vec4(bla);
    }

    // clamp components
    if (color.x > 1.0)
      color.x = 1.0;
    if (color.y > 1.0)
      color.y = 1.0;
    if (color.z > 1.0)
      color.z = 1.0;
    if (color.x < 0.0)
      color.x = 0.0;
    if (color.y < 0.0)
      color.y = 0.0;
    if (color.z < 0.0)
      color.z = 0.0;

    gl_FragData[0] = color;
    gl_FragData[1] = vec4(N, 0.0);
    gl_FragData[2] = vec4(-eyeVec, 1.0);
}
