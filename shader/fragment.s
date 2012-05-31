#version 120

// normale des pixels
varying vec3 vert_norm_dir;
// vektor von/zu(?) licht
varying vec3 vert_light_dir;

uniform float uni_outerSpotAngle;
uniform float uni_innerSpotAngle;

struct Light_properties {
  vec4 position;
  vec4 ambient;
  vec4 diffuse;
  vec4 specular;
};
uniform Light_properties lights[3];

void main () {
    // normalize everything necessary //
    vec3 N = normalize(vert_norm_dir);
    vec3 L = normalize(vert_light_dir);
    vec3 E = vec3(0,0,-1);

    float LdotE = max(0.0, dot(L, E));
    // TODO PI suchen
    float angle = acos(LdotE)*180/3.14;
    // licht weiß gemacht
    if(angle < uni_innerSpotAngle)
        gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0) + lights[0].specular;
    else if(angle < uni_outerSpotAngle)
        gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0) * (uni_outerSpotAngle - angle)/(uni_outerSpotAngle - uni_innerSpotAngle);
    else
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);

    // diffuse component //
    float NdotL = min(max(0.0, dot(-N, L)), 1.0);
    gl_FragColor *= NdotL;

    float amb_val = 0.1;
    gl_FragColor += vec4(amb_val, amb_val, amb_val, 0.0);
}
