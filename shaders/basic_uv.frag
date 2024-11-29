#version 120

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform float u_angle;
uniform float u_time;

uniform sampler2D u_tex0;
uniform vec2 u_tex0Resolution;

// glslViewer shaders/basic_uv.frag test.png -e u_angle,0.5

vec2 basic_uv()
{
    return gl_FragCoord.xy / u_resolution.xy;
}

vec2 center_circle()
{
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    float xDiff = (st.x - 0.5);
    float yDiff = (st.y - 0.5);
    float a = xDiff * xDiff + yDiff * yDiff;
    return vec2(a, a);
}

vec2 radical_uv(float angleOffset)
{
    // Normalize screen coordinates to range [0, 1]
    vec2 st = gl_FragCoord.xy / u_resolution.xy;

    // Shift to make the center of the screen the origin (0.5, 0.5)
    vec2 centeredUV = st - vec2(0.5);

    // Scale by aspect ratio to make it circular (instead of elliptical)
    centeredUV.x *= u_resolution.x / u_resolution.y;

    // Compute radial UV components
    // float radius = length(centeredUV);  // Distance from the center
    float fullPi = 2.0 * 3.14159265;
    float angle = mod(atan(centeredUV.y, centeredUV.x) + angleOffset, fullPi); // Angle in radians

    // Output the radial UV
    float green = (angle) / fullPi;
    return vec2(0, green);
}

vec2 rotate_uv(vec2 uv, vec2 rotatePoint, float angle)
{
    uv -= rotatePoint;
    float cosTheta = cos(angle);
    float sinTheta = sin(angle);
    vec2 rotatedUV = vec2(
        cosTheta * uv.x - sinTheta * uv.y,
        sinTheta * uv.x + cosTheta * uv.y
    );
    rotatedUV += rotatePoint;

    return rotatedUV;
}

float mod(float a, float b)
{
    return a - b * floor(a / b);
}

void main()
{
    // gl_FragColor = basic_uv();
    // gl_FragColor = center_circle();
    // gl_FragColor = vec4(radical_uv(1 * u_time), 0, 1);

    // gl_FragColor = vec4(mod(u_time, 1.0), 0, 0, 1);

    // gl_FragColor = vec4(rotate_uv(radical_uv(1 * u_time), vec2(0.5, 0.5), 0.5), 0, 1);

    // Normalize screen coordinates to range [0, 1]
    // vec2 st = gl_FragCoord.xy / u_resolution.xy;

    // // Shift to make (0.5, 0.5) the center
    // vec2 centeredUV = st - vec2(0.5);

    // // Rotation matrix
    // float u_angle = 1 * u_time;
    // float cosTheta = cos(u_angle);
    // float sinTheta = sin(u_angle);
    // vec2 rotatedUV = vec2(
    //     cosTheta * centeredUV.x - sinTheta * centeredUV.y,
    //     sinTheta * centeredUV.x + cosTheta * centeredUV.y
    // );

    // Output rotated UV as color
    // gl_FragColor = vec4(rotatedUV, 0.0, 1.0);

    // vec2 uv = rotate_uv(basic_uv(), vec2(0.5, 0.5), u_angle * u_time);
    // vec2 uv = radical_uv();
    vec2 uv = basic_uv();
    // vec2 uv = rotate_uv(radical_uv(), vec2(0.5, 0.5), u_angle * u_time);
    // gl_FragColor = vec4(uv, 0.0, 1.0);
    // gl_FragColor = vec4(center_circle(), 0.0, 1.0);

    vec4 texColor = texture2D(u_tex0, uv);
    vec2 radicalUV = radical_uv(-3.1415926 / 2);
    texColor.w = step(mod(u_time * 0.5, 1.0), radicalUV.y);

    gl_FragColor = texColor;
}