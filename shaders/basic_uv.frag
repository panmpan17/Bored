#version 120

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;

vec4 basic_uv()
{
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    return vec4(st.x, st.y, 0, 1);
}

vec4 center_circle()
{
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    float xDiff = (st.x - 0.5);
    float yDiff = (st.y - 0.5);
    float a = xDiff * xDiff + yDiff * yDiff;
    return vec4(a, a, a, 1);
}

void main()
{
    // gl_FragColor = basic_uv();
    // gl_FragColor = center_circle();

    vec4 color = center_circle();
    color *= basic_uv();
    color.x += 0.1f;
    color.y += 0.1f;

    // float a = (color.x * color.x + color.y * color.y) + 0.2f;
    // color.x = a;
    // color.y = a;
    // color.z = a;

    gl_FragColor = color;
}