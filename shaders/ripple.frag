#version 120

#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D u_tex0;
uniform vec2 u_tex0Resolution;

uniform vec2 u_resolution;
uniform vec2 u_mouse;

uniform float u_timeScale = 10;
uniform float u_time;

const float PI = 3.14159265;

float mod(float a, float b)
{
    return a - b * floor(a / b);
}

vec2 center_circle(vec2 center)
{
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    float xDiff = (st.x - center.x);
    float yDiff = (st.y - center.y);
    float a = xDiff * xDiff + yDiff * yDiff;
    return vec2(a, a);
}


void main()
{
    vec2 st = gl_FragCoord.xy / u_resolution.xy;

    vec2 ripperCenter = u_mouse;
    // vec2 ripperCenter = vec2(u_resolution.x / 2);

    vec2 diffToMouse = (gl_FragCoord.xy - ripperCenter.xy) / u_resolution.xy;
    float disSqrt = sqrt(pow(diffToMouse.x, 2) + pow(diffToMouse.y, 2));
    float invertDisSqrt = max(1 - (disSqrt * 2), 0);

    float c = sin((invertDisSqrt / 0.05f - 0.25) * 2 + (u_time * u_timeScale)) * invertDisSqrt;

    vec2 uv = st + diffToMouse * c;
    vec4 textColor = texture2D(u_tex0, uv);
    gl_FragColor = textColor;
}
