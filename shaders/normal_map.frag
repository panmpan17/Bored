#version 120

#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D u_tex0;
uniform vec2 u_tex0Resolution;

uniform sampler2D u_normalMap;
uniform vec2 u_normalMapResolution;

uniform float u_timeScale = 0.1;
uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;


void main()
{
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;

    vec4 textColor = texture2D(u_tex0, uv);
    vec4 normalColor = texture2D(u_normalMap, uv);

    vec3 normal = normalize(normalColor.rgb * 2.0 - 1.0);

    vec2 mousePoseRelative = u_mouse.xy / u_resolution.xy;
    vec3 lightDir = normalize(vec3(mousePoseRelative, 1.0));
    
    float diffuse = max(dot(normal, lightDir), 0.0);

    gl_FragColor = vec4(textColor.rgb * diffuse, textColor.a);
}

// glslViewer shaders/normal_map.frag test.png -normalMap test_n.png