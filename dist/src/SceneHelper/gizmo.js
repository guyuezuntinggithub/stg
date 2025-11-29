import { Lines } from "./lines.js";
import * as THREE from "three";
import { Tween, Group, Easing } from '@tweenjs/tween.js';

export class Gizmo {
    constructor(camera, graphicsAPI) {
        this.camera = camera;
        this.graphicsAPI = graphicsAPI;

        this.frustumSize = 3.0;
        const left = -this.frustumSize / 2;
        const right = this.frustumSize / 2;
        const top = this.frustumSize / 2;
        const bottom = -this.frustumSize / 2;
        const near = 0.1;
        const far = 100;
        this.orthocamera = new THREE.OrthographicCamera(left, right, top, bottom, near, far);
        this.orthocamera.up.set(0, -1, 0);

        // click
        this.tweenGroup = new Group();
        this.raycaster = new THREE.Raycaster();
        this.radius = 0.3;
        this.circlesCenter = []
        this.circlesCenter.push(new THREE.Vector3(-1, 0, 0));
        this.circlesCenter.push(new THREE.Vector3(0, -1, 0));
        this.circlesCenter.push(new THREE.Vector3(0, 0, -1));
        this.circlesCenter.push(new THREE.Vector3( 1, 0, 0));
        this.circlesCenter.push(new THREE.Vector3(0,  1, 0));
        this.circlesCenter.push(new THREE.Vector3(0, 0,  1));
        this.clickCircleId = -1;

        // Line
        this.lines = new Lines(this.graphicsAPI);

        const lineLength = 0.75;
        this.lines.addLine({ start: [0, 0, 0, 0.84, 0.28, 0.28, 1], end: [lineLength, 0, 0, 0.84, 0.28, 0.28, 1] });
        this.lines.addLine({ start: [0, 0, 0, 0.22, 0.64, 0.39, 1], end: [0, lineLength, 0, 0.22, 0.64, 0.39, 1] });
        this.lines.addLine({ start: [0, 0, 0, 0.20, 0.50, 0.82, 1], end: [0, 0, lineLength, 0.20, 0.50, 0.82, 1] });

        this.lines.updateBuffers();

        // Circle
        this.texture = this.graphicsAPI.loadTexture("/scenes/gizmo.png", true);
        // --- 3D 着色器 ---
        const vsSource = `#version 300 es
            in vec2 a_position;
            in vec2 a_uv;
            uniform mat4 u_projection;
            uniform mat4 u_view;
            uniform int u_highlightIdx;
            out vec3 v_uvw;
            
            void main() {
                int row = gl_InstanceID / 3;
                int col = gl_InstanceID % 3;
                vec4 end = vec4(0, 0, 0, 1);
                end[col] = float(row) * 2.0 - 1.0;

                vec4 screenCenter = u_projection * u_view * end;
                vec3 offset = screenCenter.xyz / screenCenter.w;
                vec2 scale = vec2(${(2 * this.radius / this.frustumSize).toFixed(5)});
                gl_Position = vec4(offset.xy + a_position * scale, offset.z, 1.0);
                v_uvw.x = a_uv.x + float(col) * 0.333333333;
                v_uvw.y = a_uv.y + float(row) * 0.5;
                if (u_highlightIdx == gl_InstanceID) {
                    v_uvw.z = 1.5;  // highlight
                } else {
                    v_uvw.z = 1.0;
                }
            }
        `;
        const fsSource = `#version 300 es
            precision mediump float;
            in vec3 v_uvw;
            uniform sampler2D u_texture;
            out vec4 outColor;
            void main() {
                vec4 color = texture(u_texture, v_uvw.xy) * v_uvw.z;
                if (color.a < 0.40) {
                    discard;
                }
                outColor = color;
            }
        `;
        this.program = this.graphicsAPI.setupProgram(vsSource, fsSource);
        this.uniforms = this.graphicsAPI.getUniform(this.program);
        this.attributes = this.graphicsAPI.getAttrib(this.program);
        this.vao = this.graphicsAPI.setupCircleVAO(this.attributes['a_position'], this.attributes['a_uv']);
        
        this.activeBindID = 7;
        this.graphicsAPI.bindTexture(this.texture, this.activeBindID);
        this.graphicsAPI.updateProgram(this.program);
        this.graphicsAPI.updateUniform(this.uniforms['u_texture'], '1i', this.activeBindID);
    }

    update(currentTime, deltaT) {
        this.tweenGroup.update(currentTime);
    }

    render = function() {
        const radius = 3.0;
        
        return function(){
            this.orthocamera.position.set(
                this.camera.matrixWorld.elements[8] * radius, 
                this.camera.matrixWorld.elements[9] * radius,
                this.camera.matrixWorld.elements[10] * radius
            );
            this.orthocamera.lookAt(0, 0, 0);

            // circles pipeline
            this.graphicsAPI.updateProgram(this.program);
            this.graphicsAPI.updateVertexInput(this.vao.vao);
            this.graphicsAPI.updateUniform(this.uniforms.u_projection, 'Matrix4fv', this.orthocamera.projectionMatrix.elements, false);
            this.graphicsAPI.updateUniform(this.uniforms.u_view, 'Matrix4fv', this.orthocamera.matrixWorldInverse.elements, false);
            this.graphicsAPI.updateUniform(this.uniforms.u_highlightIdx, '1i', this.clickCircleId);
            this.graphicsAPI.drawInstanced("TRIANGLE_FAN", 0, 4, 6);
            this.graphicsAPI.updateVertexInput(null);

            // lines pipeline
            // draw after circles so it won't be alpha-blended with transparent circles
            this.lines.render(this.orthocamera.projectionMatrix.elements, this.orthocamera.matrixWorldInverse.elements);
        }
        
    }();

    mouseMove(mouse) {
        if (mouse.x < 0 || mouse.y < 0) {
            this.clickCircleId = -1;
            return;
        }
        mouse.x = mouse.x * 2 - 1;
        mouse.y = mouse.y * 2 - 1;
        this.raycaster.setFromCamera(mouse, this.orthocamera);

        let minDistance = 1000;
        const ray = this.raycaster.ray;
        let hitIdx = -1;
        for (let i=0; i<6; ++i) {
            const dist = this.raySphereIntersect(ray.origin, ray.direction, this.circlesCenter[i], this.radius);
            if (dist > 0 && dist < minDistance) {
                minDistance = dist;
                hitIdx = i;
            }
        }
        this.clickCircleId = hitIdx;
    }

    mouseClick() {
        if (this.clickCircleId < 0) {
            return;
        }

        if (this.camera.controls.type === 'orbit') {
            const target = this.camera.controls.target;
            const radius = this.camera.position.distanceTo(target);
            const startVec = this.orthocamera.position.clone().normalize();
            const rotationQuat = new THREE.Quaternion().setFromUnitVectors(startVec, this.circlesCenter[this.clickCircleId]);
            const tempQuat = new THREE.Quaternion();

            const progress = { t: 0 };
            const tween = new Tween(progress)
                .to({ t: 1 }, 500) // 用2秒时间完成动画
                .easing(Easing.Cubic.In)
                .onUpdate(() => {
                    tempQuat.slerp(rotationQuat, progress.t);
                    const currentPos = startVec.clone().applyQuaternion(tempQuat).multiplyScalar(radius).add(target);
                    this.camera.position.copy(currentPos);
                })
                .start();
            this.tweenGroup.removeAll();
            this.tweenGroup.add(tween);
        } else if (this.camera.controls.type === 'fly' || this.camera.controls.type === 'pointerLock') {
            const startVec = this.orthocamera.position.clone().normalize();
            const rotationQuat = new THREE.Quaternion().setFromUnitVectors(startVec, this.circlesCenter[this.clickCircleId]);
            const tempQuat = new THREE.Quaternion();

            const progress = { t: 0 };
            const tween = new Tween(progress)
                .to({ t: 1 }, 500) // 用2秒时间完成动画
                .easing(Easing.Cubic.In)
                .onUpdate(() => {
                    tempQuat.slerp(rotationQuat, progress.t);
                    const viewDir = startVec.clone().applyQuaternion(tempQuat).multiplyScalar(-1).add(this.camera.position);
                    this.camera.lookAt(viewDir);
                })
                .start();
            this.tweenGroup.removeAll();
            this.tweenGroup.add(tween);
        }
    }

    /**
     * 计算射线与球体的交点。
     * @param {THREE.Vector3} rayOrigin - 射线的起点。
     * @param {THREE.Vector3} rayDirection - 射线的方向（必须是单位向量）。
     * @param {THREE.Vector3} sphereCenter - 球体的中心。
     * @param {number} sphereRadius - 球体的半径。
     * @returns {number} 如果相交，返回从射线起点到最近交点的距离；如果不相交，返回 -1。
     */
    raySphereIntersect(rayOrigin, rayDirection, sphereCenter, sphereRadius) {
        // 计算从球心指向射线起点的向量 L
        const L = new THREE.Vector3().subVectors(rayOrigin, sphereCenter);

        // 解一元二次方程 at^2 + bt + c = 0
        // a = rayDirection · rayDirection (因为方向是单位向量，所以 a = 1)
        const a = 1; 

        // b = 2 * (rayDirection · L)
        const b = 2 * rayDirection.dot(L);

        // c = (L · L) - r^2
        const c = L.dot(L) - sphereRadius * sphereRadius;

        // 计算判别式 (b^2 - 4ac)
        const discriminant = b * b - 4 * a * c;

        // 如果判别式 < 0，说明射线没有与球体相交
        if (discriminant < 0) {
            return -1;
        }

        // 计算两个可能的交点距离 t0 和 t1
        const sqrtDiscriminant = Math.sqrt(discriminant);
        const t0 = (-b - sqrtDiscriminant) / (2 * a);
        const t1 = (-b + sqrtDiscriminant) / (2 * a);

        // 我们需要找到最近的、在射线前进方向上的交点 (t > 0)

        // 如果 t0 > 0，说明 t0 是最近的有效交点
        if (t0 > 0) {
            return t0;
        }

        // 如果 t0 < 0 而 t1 > 0，说明射线起点在球体内部，t1 是前方的交点
        if (t1 > 0) {
            return t1;
        }

        // 如果 t0 和 t1 都 <= 0，说明整个球体都在射线的后面
        return -1;
    }
}