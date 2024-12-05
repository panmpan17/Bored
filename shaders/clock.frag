#version 120

#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D u_tex0;
uniform vec2 u_tex0Resolution;

uniform float u_timeScale = 0.1;
uniform float u_time;
uniform vec2 u_resolution;

uniform vec2 u_mouse;

float mod(float a, float b)
{
    return a - b * floor(a / b);
}

float length_between_point(vec2 point1, vec2 point2)
{
    float xDiff = (point1.x - point2.x);
    float yDiff = (point1.y - point2.y);
    float length = sqrt(xDiff * xDiff + yDiff * yDiff);
    return length;
}

float radical_progress(float angleOffset, vec2 delta)
{
    // vec2 st = gl_FragCoord.xy / u_resolution.xy;

    // vec2 centeredUV = st - center;

    float fullPi = 2.0 * 3.14159265;
    float angle = mod(atan(delta.y, delta.x) + angleOffset, fullPi);

    return angle / fullPi;
}

float customRange(float edge0, float edge1, float x) {
    // Smooth transition between the min and max range
    return smoothstep(edge0, edge1, x) * (1.0 - smoothstep(edge1, edge1 + (edge1 - edge0), x));
}

void main()
{
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    vec2 m = u_mouse.xy / u_resolution.xy;

    float length = length_between_point(st, vec2(0.5));

    float angle = radical_progress(0, st - vec2(0.5));
    float angle2 = radical_progress(0, m - vec2(0.5));
    float angleDiff = abs(1 - (angle - angle2));

    angle = mod(angle, 0.05);
    angle = step(0.025, angle);

    float result = customRange(0.2, 0.3, length);
    result *= angle2;

    gl_FragColor = vec4(result, 0, 0, 1);
    // gl_FragColor = vec4(radical_progress(0, vec2(0.5)), 0, 0, 1);
}
