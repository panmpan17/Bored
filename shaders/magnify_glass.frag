#version 130

#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D u_tex0;
uniform vec2 u_tex0Resolution;

uniform vec2 u_resolution;
uniform vec2 u_mouse;

uniform float u_timeScale = 10;
uniform float u_time;

uniform float u_areaScale = 3;
uniform float u_magnifyScale = 2;

void main()
{
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;

    vec2 centerRaw = u_mouse;
    // centerRaw = vec2(u_resolution.x / 2); // Unmark this line to center the effect
    vec2 center = centerRaw / u_resolution.xy;

    vec2 diffToMouse = (gl_FragCoord.xy - centerRaw.xy) / u_resolution.xy;
    float disSqrt = sqrt(pow(diffToMouse.x, 2) + pow(diffToMouse.y, 2));

    float t = clamp(1 - (disSqrt * u_areaScale), 0, 1);
    t = pow(t, 0.5);
    // t = step(disSqrt * u_areaScale, 0.2);

    vec2  magnifiedUv = (uv + center) / u_magnifyScale;

    vec2 newUV = uv * (1 - t) + magnifiedUv * (t);

    vec4 textColor = texture2D(u_tex0, newUV);
    // vec4 textColor = vec4(t, 0, 0, 1);
    gl_FragColor = textColor;
}
