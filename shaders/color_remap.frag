#version 120

#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D u_tex0;
uniform vec2 u_tex0Resolution;

uniform sampler2D u_lut;
uniform vec2 u_lutResolution;

uniform float u_timeScale = 0.1;
uniform float u_time;
uniform vec2 u_resolution;

uniform vec2 u_mouse;

vec3 rgb_to_hsv(vec3 rgb) // rgb is also 0.0 to 1.0
{
    float _max = max(max(rgb.r, rgb.g), rgb.b);
    float _min = min(min(rgb.r, rgb.g), rgb.b);
    float diff = _max - _min;

    float S = 0;
    if (_max != 0)
    {
        S = (_max - _min) / _max;
    }

    float V = _max;

    float H = 0;
    if (diff != 0)
    {
        float r_diff = _max - (rgb.r / 6 / diff) + 0.5f;
        float g_diff = _max - (rgb.g / 6 / diff) + 0.5f;
        float b_diff = _max - (rgb.b / 6 / diff) + 0.5f;

        if (_max == rgb.r)
        {
            H = b_diff - g_diff;
        }
        else if (_max == rgb.g)
        {
            H = 1.0 / 3.0 + r_diff - b_diff;
        }
        else if (_max == rgb.b)
        {
            H = 2.0 / 3.0 + g_diff - r_diff;
        }

        if (H < 0)
        {
            H += 1.0;
        }
        else if (H > 1)
        {
            H -= 1.0;
        }

        H *= 360.0; // Convert to degrees
    }

    return vec3(H, S, V);
}

vec3 hsv_to_rgb(vec3 hsv)
{
    if (hsv.y == 0.0)
    {
        return vec3(hsv.z, hsv.z, hsv.z); // Achromatic (grey)
    }

    float h = hsv.x / 60.0; // Convert to 0-6 range
    int i = int(floor(h));
    float f = h - float(i);

    float p = hsv.z * (1.0 - hsv.y);
    float q = hsv.z * (1.0 - hsv.y * f);
    float t = hsv.z * (1.0 - hsv.y * (1.0 - f));

    if (i == 0)
    {
        return vec3(hsv.z, t, p);
    }
    else if (i == 1)
    {
        return vec3(q, hsv.z, p);
    }
    else if (i == 2)
    {
        return vec3(p, hsv.z, t);
    }
    else if (i == 3)
    {
        return vec3(p, q, hsv.z);
    }
    else if (i == 4)
    {
        return vec3(t, p, hsv.z);
    }
    else // i == 5
    {
        return vec3(hsv.z, p, q);
    }
}

vec4 lookup_lut(vec4 color, float size)
{
     float r = color.r * (size - 1.0);
    float g = color.g * (size - 1.0);
    float b = color.b * (size - 1.0);

    float z = floor(b);
    float x = mod(z, size);
    float y = floor(z / size);

    vec2 lutUV = vec2(
        (r + x * size + 0.5) / (size * size),
        (g + y * size + 0.5) / (size)
    );

    return vec4(texture2D(u_lut, lutUV).xyz, color.a);
}

void main()
{
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    uv *= 2;

    vec4 textColor = texture2D(u_tex0, uv);

    // Hue shifting example
    // vec3 hsv = rgb_to_hsv(vec3(textColor));
    // hsv.r = 100; // This is hue in degrees 0-360
    // vec3 newRGB = hsv_to_rgb(hsv);
    // gl_FragColor = vec4(newRGB, textColor.a);

    if (uv.x > 1)
    {
        // My version but failed
        // float dimension = 16.0;
        // float xCoord = textColor.r * dimension + (floor(max(0, textColor.b * dimension - .5f)) * dimension);
        // float yCoord = textColor.g * dimension;
        // vec2 lutUv = vec2(xCoord / (dimension * dimension), yCoord / dimension);
        // textColor = texture2D(u_lut, lutUv);

        textColor = lookup_lut(textColor, 16);
    }

    gl_FragColor = vec4(textColor);
}
