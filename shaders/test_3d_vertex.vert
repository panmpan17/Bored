#version 120

in vec3 position;
in vec2 texcoord;

out vec2 vUv;

uniform mat4 modelViewProjectionMatrix;

void main() {
    vUv = texcoord;
    gl_Position = modelViewProjectionMatrix * vec4(position, 1.0);
}