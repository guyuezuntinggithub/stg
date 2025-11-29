import * as THREE from "three";
import { XRHandModelFactory } from 'three/addons/webxr/XRHandModelFactory.js';
import { Lines } from "../SceneHelper/lines.js";

const SKINNING_VERTEX_SHADER = `#version 300 es
#define MAX_JOINTS 25
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in uvec4 a_joint_indices;
layout(location = 3) in vec4 a_joint_weights;
uniform mat4 u_projectionMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_modelMatrix;
uniform mat4 u_jointMatrices[MAX_JOINTS];
out vec3 v_normal;
void main() {
    mat4 skinMatrix = a_joint_weights.x * u_jointMatrices[a_joint_indices.x] + a_joint_weights.y * u_jointMatrices[a_joint_indices.y] + a_joint_weights.z * u_jointMatrices[a_joint_indices.z] + a_joint_weights.w * u_jointMatrices[a_joint_indices.w];
    vec4 skinnedPosition = skinMatrix * vec4(a_position, 1.0);
    gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * skinnedPosition;
    mat4 normalMatrix = transpose(inverse(u_modelMatrix * skinMatrix));
    v_normal = normalize((normalMatrix * vec4(a_normal, 0.0)).xyz);
}`;

const LIT_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec3 v_normal;
uniform vec3 u_lightDirection;
out vec4 outColor;
void main() {
    vec3 normal = normalize(v_normal);
    float lightIntensity = max(dot(normal, normalize(u_lightDirection)), 0.0);
    vec3 baseColor = vec3(0.8, 0.8, 0.8);
    vec3 ambient = vec3(0.2);
    vec3 finalColor = ambient + baseColor * lightIntensity;
    outColor = vec4(finalColor, 1.0);
}`;

export class XRScene {
    constructor(graphicsAPI) {
        this.graphicsAPI = graphicsAPI;

        this.debug = true;

        this.virtualScene = new THREE.Scene();
        this.hands = [];
        this.handModels = [];
        this.lineHands = [];
        this.handRenderData = {};

        // webgl
        this.skinningShaderProgram = null;
        this.shaderLocations = {
            attributes: null,
            uniforms: null,
        };;
        this.identityMatrix = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
    }

    addHand(handGroup, index) {
        this.hands[index] = handGroup;
        this.virtualScene.add(handGroup);

        const factory = new XRHandModelFactory();
        const handModel = factory.createHandModel(handGroup, 'mesh');
        this.handModels[index] = handModel;
        this.hands[index].add(handModel);

        if (this.debug) {
            this.lineHands[index] = new Lines(this.graphicsAPI);
        }

        handGroup.addEventListener('connected', (event) => {
            const inputSource = event.data;
            const handedness = inputSource.handedness;
            handModel.userData.handedness = handedness;
            handGroup.userData.handedness = handedness;
        });
    }
    
    addController( index ) {
    }

    addControllerGrip( index ) {
    }

    setupNativeWebGLResources(handedness, skinnedMesh) {
        if (!this.skinningShaderProgram) {
            this.skinningShaderProgram = this.graphicsAPI.setupProgram(SKINNING_VERTEX_SHADER, LIT_FRAGMENT_SHADER);
            this.shaderLocations.attributes = this.graphicsAPI.getAttrib(this.skinningShaderProgram);
            this.shaderLocations.uniforms = this.graphicsAPI.getUniform(this.skinningShaderProgram);
        }

        const geometry = skinnedMesh.geometry;
        const positions = geometry.attributes.position.array;
        const normals = geometry.attributes.normal.array;
        const skinIndices = geometry.attributes.skinIndex.array;
        const skinWeights = geometry.attributes.skinWeight.array;
        const indices = geometry.index.array;

        const attributes = this.shaderLocations.attributes;
        
        const gl = this.graphicsAPI.getContext();
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        
        this.graphicsAPI.createAndBindBuffer(gl.ARRAY_BUFFER, positions, attributes.a_position, 3, gl.FLOAT);
        this.graphicsAPI.createAndBindBuffer(gl.ARRAY_BUFFER, normals, attributes.a_normal, 3, gl.FLOAT);
        this.graphicsAPI.createAndBindBuffer(gl.ARRAY_BUFFER, skinWeights, attributes.a_joint_weights, 4, gl.FLOAT);
        
        const skinIndexVbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, skinIndexVbo);
        gl.bufferData(gl.ARRAY_BUFFER, skinIndices, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(attributes.a_joint_indices);
        gl.vertexAttribIPointer(attributes.a_joint_indices, 4, gl.UNSIGNED_BYTE, 0, 0); // 注意是 IPointer!

        const ebo = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

        gl.bindVertexArray(null);

        this.handRenderData[handedness] = {
            vao,
            indexCount: indices.length,
            skinnedMesh,
            boneMatrices: null,
            visible: false
        };
    }

    updateRenderData() {
        for (const handIdx in this.hands) {
            const hand = this.hands[handIdx];
            const handModel = this.handModels[handIdx];
            const handedness = hand.userData.handedness;
            if (!hand.visible || !handedness) {
                if (handedness && this.handRenderData[handedness]) {
                    this.handRenderData[handedness].visible = false;
                }
                continue;
            }
            if (this.debug) {
                this.updateLines(this.lineHands[handIdx], hand.joints);
            } else {
                let handData = this.handRenderData[handedness];
                if (!handData) {
                    const skinnedMesh = handModel.getObjectByProperty('type', 'SkinnedMesh');
                    if (skinnedMesh) {
                        this.setupNativeWebGLResources(handedness, skinnedMesh);
                        handData = this.handRenderData[handedness];
                    }
                }
                if (handData) {
                    const skinnedMesh = handData.skinnedMesh;

                    handModel.updateMatrixWorld(true);
                    skinnedMesh.updateMatrixWorld(true);
                    skinnedMesh.skeleton.update();
                    handData.boneMatrices = skinnedMesh.skeleton.boneMatrices;
                    handData.visible = true;
                }
            }
        }
    }

    renderHands(view) {
        const gl = this.graphicsAPI.getContext();
        const uniformsLoc = this.shaderLocations.uniforms;

        this.graphicsAPI.disableCull();
        this.graphicsAPI.enableDepth();
        for (const handIdx in this.hands) {
            const hand = this.hands[handIdx];
            if (!hand.visible) {
                continue;
            }

            if (this.debug)  {
                this.lineHands[handIdx].render(view.projectionMatrix, view.transform.inverse.matrix);
            } else {
                const handData = this.handRenderData[this.handModels[handIdx].userData.handedness];
                if (handData && handData.visible) {
                    this.graphicsAPI.updateProgram(this.skinningShaderProgram);
                    this.graphicsAPI.updateVertexInput(handData.vao);

                    this.graphicsAPI.updateUniform(uniformsLoc.u_projectionMatrix, 'Matrix4fv', view.projectionMatrix, false);
                    this.graphicsAPI.updateUniform(uniformsLoc.u_viewMatrix, 'Matrix4fv', view.transform.inverse.matrix, false);
                    this.graphicsAPI.updateUniform(uniformsLoc.u_modelMatrix, 'Matrix4fv', this.identityMatrix, false);
                    this.graphicsAPI.updateUniform(uniformsLoc.u_jointMatrices, 'Matrix4fv', handData.boneMatrices, false);
                    this.graphicsAPI.updateUniform(uniformsLoc.u_lightDirection, '3fv', [0.5, 1.0, 0.5]);
                    // TODO
                    gl.drawElements(gl.TRIANGLES, handData.indexCount, gl.UNSIGNED_SHORT, 0);
                }
            }
        }
        this.graphicsAPI.updateVertexInput(null);
    }

    updateLines = function() {
        const jointsMap = [
            'wrist',
            'thumb-metacarpal',
            'thumb-phalanx-proximal',
            'thumb-phalanx-distal',
            'thumb-tip',
            'index-finger-metacarpal',
            'index-finger-phalanx-proximal',
            'index-finger-phalanx-intermediate',
            'index-finger-phalanx-distal',
            'index-finger-tip',
            'middle-finger-metacarpal',
            'middle-finger-phalanx-proximal',
            'middle-finger-phalanx-intermediate',
            'middle-finger-phalanx-distal',
            'middle-finger-tip',
            'ring-finger-metacarpal',
            'ring-finger-phalanx-proximal',
            'ring-finger-phalanx-intermediate',
            'ring-finger-phalanx-distal',
            'ring-finger-tip',
            'pinky-finger-metacarpal',
            'pinky-finger-phalanx-proximal',
            'pinky-finger-phalanx-intermediate',
            'pinky-finger-phalanx-distal',
            'pinky-finger-tip',
        ];
        const positions = {};
        const thumb = [0, 1, 2, 3, 4];
        const index = [0, 5, 6, 7, 8, 9];
        const middle = [0, 10, 11, 12, 13, 14];
        const ring = [0, 15, 16, 17, 18, 19];
        const pinky = [0, 20, 21, 22, 23, 24];
        const hand = [thumb, index, middle, ring, pinky];
        return function(lines, joints) {
            const color = [1, 0, 0, 1];
            for (const name of jointsMap) {
                positions[name] = [...(joints[name]?.position.toArray() || [0, 0, 0]), ...color];
            }

            lines.clear();
            for (const finger of hand) {
                for (let i = 1; i < finger.length; ++i) {
                    lines.addLine({ start: positions[jointsMap[finger[i - 1]]], end: positions[jointsMap[finger[i]]] });
                }
            }

            lines.updateBuffers(true);
        };
    } ()
}