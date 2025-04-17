#version 120
uniform vec2 u_resolution;
uniform float u_time;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
}

vec2 rand2(vec2 co) {
    float r1 = rand(co);
    float r2 = rand(co + 3.14); // offset to avoid correlation
    return vec2(r1, r2);
}

float voronoi(vec2 uv) {
    vec2 p = floor(uv);
    vec2 f = fract(uv);
    float minDist = 10.0;

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 neighbor = vec2(float(i), float(j));
            vec2 point = rand2(p + neighbor);
            point = 0.5 + 0.5 * sin(u_time + 6.2831 * point); // animate points
            vec2 diff = neighbor + point - f;
            float d = dot(diff, diff);
            minDist = min(minDist, d);
        }
    }

    return sqrt(minDist);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution.xy * 10.0; // 10 cells across
    float v = voronoi(uv);
    gl_FragColor = vec4(vec3(v), 1.0);
}
