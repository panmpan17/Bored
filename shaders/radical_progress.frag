#version 120

#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D u_tex0;
uniform vec2 u_tex0Resolution;

uniform float u_timeScale = 0.1;
uniform float u_time;
uniform vec2 u_resolution;

const float FULL_PI = 3.14159265 * 2.0;

float mod(float a, float b)
{
    return a - b * floor(a / b);
}

float radical_progress(vec2 pos, float angleOffset, vec2 center)
{
    vec2 centeredUV = pos - center;
    float angle = mod(atan(centeredUV.y, centeredUV.x) + angleOffset, FULL_PI);
    return angle / FULL_PI;
}

void main()
{
    vec2 pos = gl_FragCoord.xy / u_resolution.xy;
    vec4 textColor = texture2D(u_tex0, pos);
    textColor.w *= step(mod(u_timeScale * u_time, 1), 1 - radical_progress(pos, -1.570796325, vec2(0.5)));
    gl_FragColor = textColor;
}

