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

void main()
{
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;

    float color = uv.x;
    if (uv.y < 0.2) {
        color = smoothstep(0.0, 1.0, uv.x);
    }
    else if (uv.y < 0.4)
    {
        float t = clamp(uv.x, 0.0, 1.0);
        float curved = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        color = curved;
    }
    else if (uv.y < 0.6)
    {
        float t = clamp(uv.x, 0.0, 1.0);
        float curved = pow(t, 0.5);
        color = curved;
    }

    gl_FragColor = vec4(color, color, color, 1);
}
