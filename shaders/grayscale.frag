#version 120

#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D u_tex0;
uniform vec2 u_tex0Resolution;

uniform vec2 u_resolution;


float choose_median(float a, float b, float c)
{
    if ((a >= b && a <= c) || (a <= b && a >= c))
    {
        return a;
    }
    else if ((b >= a && b <= c) || (b <= a && b >= c))
    {
        return b;
    }
    else
    {
        return c;
    }
}

void main()
{
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    uv *= 2;

    vec4 textColor = texture2D(u_tex0, uv);

    float colorValue;

    if (uv.x <= 1)
    {
        if (uv.y <= 1)
        {
            // Bottom left quadrant
            colorValue = min(min(textColor.r, textColor.g), textColor.b);
        }
        else
        {
            // Top left quadrant
            colorValue = max(max(textColor.r, textColor.g), textColor.b);
        }
    }
    else
    {
        if (uv.y <= 1)
        {
            // Bottom right quadrant
            colorValue = (textColor.r + textColor.g + textColor.b) / 3.0;
        }
        else
        {
            // Top right quadrant
            colorValue = choose_median(textColor.r, textColor.g, textColor.b);
        }
    }

    textColor.r = colorValue;
    textColor.g = colorValue;
    textColor.b = colorValue;

    gl_FragColor = textColor;
}
