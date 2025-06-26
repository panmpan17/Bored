
#extension GL_OES_standard_derivatives : enable

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;

void main()
{
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    gl_FragColor = vec4(st, 0, 1);
    // // gl_FragColor = vec4(radical_progress(0, vec2(0.5)), 0, 0, 1);
}
