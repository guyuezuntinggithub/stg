import { WebGL } from "./backend/webgl.js";
import { EventBus } from "./EventBus.js";
import { GSLoader } from "./GSLoader/Loader.js";
import { GSScene } from "./GSScene.js";
import { ShaderManager } from "./ShaderManager.js";
import { GSSorter } from "./sorter/GSSorter.js";
import { Utils } from "./Utils.js";
import { OrbitControls } from './controls/OrbitControls.js';
import { PointerLockControls } from './controls/PointerLockCotrols.js';
import { SceneHelper } from './SceneHelper/sceneHelper.js';
import { GSType, RenderMode } from "./Global.js";
import * as THREE from "three"
import { XRManager } from "./WebXR/XRManager.js";


export default class GSViewer {
    constructor() {
        this.canvas = document.getElementById('drop-zone');
        this.graphicsAPI = new WebGL(this.canvas);
        this.eventBus = new EventBus();
        this.options = {
            debug: false,
            enableDebugOutput: false,
            destroyOnLoad: false,
            sharedMemoryForWorkers: false,
            enableSIMDInSort: true,
            cacheShaders: true,
            enablePointerLock: true,

            initialCameraPosition: undefined,
            cameraUp: undefined,
            initialCameraLookAt: undefined,
            cameraFOV: undefined,

            isMobile: undefined,
        }
        this.__resolveOptions();

        this.devicePixelRatio = window.devicePixelRatio;
        this.canvas.width  = Math.round(this.canvas.clientWidth  * this.devicePixelRatio);
        this.canvas.height = Math.round(this.canvas.clientHeight * this.devicePixelRatio);
        this.perspectiveCamera = null;
        this.camera = null;
        this.initialCameraPosition = this.options.initialCameraPosition;
        this.cameraUp = this.options.cameraUp;
        this.initialCameraLookAt = this.options.initialCameraLookAt;
        this.cameraFOV = this.options.cameraFOV;
        this.backgroundColor = [0.15, 0.15, 0.15];

        // mobile: only support orbitControls
        // pc: support orbit and pointerLock without lock(useAsFlyControls = true)
        //     only when enablePointerLockControls = true and useAsFlyControls = false, we lock pointer.
        this.orbitControls = null;
        this.pointerLockControls = null;
        this.useAsFlyControls = true;   // a flag to decide two states of this.pointerLockControls
        this.controls = null;

        // Movement State
        this.isWDown = false;
        this.isSDown = false;
        this.isADown = false;
        this.isDDown = false;
        this.isYDown = false;   // freeze Y
        this.isQDown = false;   // camera rotation
        this.isEDown = false;   // camera rotation
        this.isSpaceDown = false;
        this.isLeftCtrlDown = false;
        this.freezeY = false;
        this.velocity = new THREE.Vector3();
        this.direction = new THREE.Vector3();
        this.isLeftMouseDown = false;

        this.sortForSkipFrame = false;
        this.skipTimestamps = [];
        this.eventBus.on('sortForSkipFrameDone', this.__onSortForSkipFrameDone.bind(this));
        this.startTime = performance.now();
        this.loopedTime = 0;
        this.videoDuration = 1;
        this.videostartTime = 0;
        this.vedioendTime=1;
        this.lastFPSTime = 0;
        this.frameCount = 0;
        this.fps = 30;
        this.lastFrameTime = 0;
        this.deltaT = 0;
        this.pause = false;
        this.isDraggingTimeline = false;
        this.playSpeed = 1.0;

        this.alphaCullThreshold = 3 / 255;
        this.renderMode = RenderMode.splat;
        this.showGrid = true;
        this.showGizmo = true;

        this.gsloader = new GSLoader(this.eventBus);
        this.gsScene = new GSScene(this.options, this.eventBus, this.graphicsAPI);
        this.shaderManager = new ShaderManager(this.options, this.eventBus, this.graphicsAPI);
        this.sorter = new GSSorter(this.options, this.eventBus);
        this.sceneHelper = new SceneHelper(this.canvas, this.graphicsAPI);

        this.webxr = new XRManager(this.eventBus, this.canvas, this.graphicsAPI);
        this.normalRenderLoopHandle = null;
        this.xrRenderLoopHandle = null;
        this.cyclopeanCamera = new THREE.PerspectiveCamera();
        this.eventBus.on('xrSessionEnd', this.__onXrSessionEnd.bind(this));

        this.eventBus.on('buffersReady', this.__onBuffersReady.bind(this));
        this.__setupCamera();
        this.__setupControls();

        this.sceneHelper.init(this.camera)
    }

    togglePlayPause() {
        this.pause = !this.pause;
    }

    setTimestamp(timestamp) {
        this.skipTimestamps.push(timestamp);
        if (this.skipTimestamps.length === 3) {
            this.skipTimestamps.shift();
        }
    }

    playFromStart() {
        this.setTimestamp(this.videostartTime);
        //console.warn("🔥 setTimeRange 被调用了：", this.videostartTime);
        //debugger;
    }

    setPlaybackSpeed(speed) {
        this.playSpeed = speed;
    }

    setIsDraggingTimeline(res) {
        this.isDraggingTimeline = res;
    }

    exportGlbFile() {
        this.gsScene.export();
    }

    setControlMode(mode) {
        if (mode === 'orbit') {
            if (this.pointerLockControls) {
                this.pointerLockControls.enabled = false;
            }

            this.orbitControls.enabled = true;
            this.controls = this.orbitControls;
            this.camera.controls = this.controls;
            this.controls.type = 'orbit';
            return true;
        } else if (mode === 'pointerLock') {
            if (this.options.enablePointerLock) {
                this.orbitControls.enabled = false;

                this.useAsFlyControls = false;
                this.pointerLockControls.useAsFlyControls = this.useAsFlyControls;
                this.controls = this.pointerLockControls;
                this.pointerLockControls.enabled = true;
                this.camera.controls = this.controls;
                this.controls.type = 'pointerLock';
                return true;
            } else {
                return false;
            }
        } else if (mode === 'fly') {
            if (this.pointerLockControls) {
                this.orbitControls.enabled = false;

                this.useAsFlyControls = true;
                this.pointerLockControls.useAsFlyControls = this.useAsFlyControls;
                this.controls = this.pointerLockControls;
                this.pointerLockControls.enabled = true;
                this.camera.controls = this.controls;
                this.controls.type = 'fly';
                return true;
            } else {
                return false;
            }
        }
    }

    async switchToXR(sessionType) {
        if (this.normalRenderLoopHandle) {
            cancelAnimationFrame(this.normalRenderLoopHandle);
            this.normalRenderLoopHandle = null;
        }
        const result = await this.webxr.initSession(sessionType);
        if (!result) {
            this.run();
            return false;
        }
        this.controls.enabled = false;
        this.runXR();
        switch (this.gsScene.gsType) {
            case 'ThreeD': this.alphaCullThreshold = 15 / 255; break;
            case 'SPACETIME': this.alphaCullThreshold = 35 / 255; break;
            default: break;
        }

        return true;
    }

    recheckXR() {
        this.webxr.checkEnv();
    }

    setRenderMode(mode) {
        const value = RenderMode[mode];
        if (value && value > 0) {
            this.renderMode = value;
        }
    }

    setGridVisibility(visible) {
        this.showGrid = visible;
    }

    setGizmoVisibility(visible) {
        this.showGizmo = visible;
    }

    setAlphaCullThreshold(threshold) {
        this.alphaCullThreshold = threshold;
    }
    setTimeRange(s,e) {
        //console.warn("🔥 setTimeRange 被调用了：", s, e);
        //debugger;
        this.videostartTime = s;
        this.vedioendTime = e;
        this.loopedTime =s;
        this.videoDuration = e;
        this.eventBus.emit('noteExternalListener', {
            updateTimestamp: true,
            timestamp: this.loopedTime,
            duration: this.videoDuration,
        });
    }

    lockPointer() {
        if (this.options.enablePointerLock && !this.useAsFlyControls) {
            this.pointerLockControls.lock();
        }
    }

    getFPS() {
        return this.fps;
    }

    getFrameTime() {
        return this.deltaT;
    }

    getSplatNum() {
        return this.gsScene.splatNum;
    }

    getLastSortTime() {
        return this.sorter.getLastSortTime();
    }

    getLastCullTime() {
        return this.sorter.getLastCullTime();
    }

    getCullingPercentage() {    // which means visible splats number
        // trick: for static 3dgs scene, force to sort once per sec
        // this.__runSplatSort(true);
        return this.sorter.getSplatSortCount() / this.gsScene.splatNum;
    }

    getResolution() {
        return { width: this.canvas.width, height: this.canvas.height, dpr: this.devicePixelRatio };
    }

    fetchSceneWithURL(url) {
        // TODO: this is just for show
        this.gsloader.readFileFromServer(url);
    }

    fetchSceneWithNative(file) {
        this.gsloader.readFileFromNative(file);
    }

    removeScene(uid) {
        this.gsScene.removeScene(uid);
    }

    switchToScene(uid) {
        this.gsScene.switchToScene(uid);
    }

    attemptToSwitchQuality(target) {

    }

    addExternalListener(func) {
        this.eventBus.on('noteExternalListener', func);
    }

    applyTransform() {
        this.gsScene.applyTransform();
    }

    resetTransform() {
        this.gsScene.resetTransform();
    }

    updateTransform() {
        this.gsScene.updateTransform();
        // force to sort for new transformed scene
        this.__runSplatSort(false, true);
    }

    setBackgroundColor(r, g, b) {
        this.backgroundColor = [r, g, b];
    }

    setDPR(dpr) {
        if (this.devicePixelRatio != dpr) {
            this.devicePixelRatio = Math.floor(dpr*100)/100;
        }
    }

    updateCamera = function() {
        const target = new THREE.Vector3();

        return function(cameraSettings) {
            this.setDPR(cameraSettings.dpr);
            const rgb = Utils.hex2rgb(cameraSettings.backgroundColor);
            if (Utils.valueChanged(this.backgroundColor, rgb)) {
                this.setBackgroundColor(rgb[0], rgb[1], rgb[2]);
            }
            this.camera.fov = cameraSettings.fov;
            this.camera.near = cameraSettings.clip.n;
            this.camera.far = cameraSettings.clip.f;
            this.camera.updateProjectionMatrix();
            if (this.controls.type === 'orbit') {
                this.camera.up.x = cameraSettings.up.x;
                this.camera.up.y = cameraSettings.up.y;
                this.camera.up.z = cameraSettings.up.z;
                this.camera.position.x = cameraSettings.pos.x;
                this.camera.position.y = cameraSettings.pos.y;
                this.camera.position.z = cameraSettings.pos.z;
                this.controls.target.x = cameraSettings.look.x;
                this.controls.target.y = cameraSettings.look.y;
                this.controls.target.z = cameraSettings.look.z;
            } else {
                this.camera.up.x = cameraSettings.up.x;
                this.camera.up.y = cameraSettings.up.y;
                this.camera.up.z = cameraSettings.up.z;
                this.camera.position.x = cameraSettings.pos.x;
                this.camera.position.y = cameraSettings.pos.y;
                this.camera.position.z = cameraSettings.pos.z;
                target.set(cameraSettings.look.x, cameraSettings.look.y, cameraSettings.look.z).add(this.camera.position);
                this.camera.lookAt(target);
            }
        }
    }();

    resetCamera() {
        this.camera.position.copy(this.initialCameraPosition);
        this.camera.up.copy(this.cameraUp).normalize();
        if (this.camera.controls.type === 'orbit') {
            this.controls.target.copy(this.initialCameraLookAt);
        }
        this.camera.lookAt(this.initialCameraLookAt);
    }

    run() {

        const animate = (currentTime) => {
            this.normalRenderLoopHandle = requestAnimationFrame(animate);

            this.__updateFPS(currentTime);
            this.__updateControls();
            this.__runSplatSort(this.gsScene.forceSort());
            this.__updateForRendererSizeChanges();
            this.sceneHelper.update(currentTime, this.deltaT);
            if (this.sortForSkipFrame) {
                // skip rendering when sorting for the new frame to escape from flash
                return;
            }
            this.graphicsAPI.updateClearColor(this.backgroundColor[0], this.backgroundColor[1], this.backgroundColor[2], 1);
            this.graphicsAPI.updateViewport();
            if (this.showGrid) {
                this.sceneHelper.renderGrid();
            }
            if (this.__shouldRender()) {
                // pass 0: gaussian splatting
                this.shaderManager.setPipeline();
                this.__updateUniforms();

                if (this.options.debug) {
                    this.graphicsAPI.drawInstanced('TRIANGLE_FAN', 0, 4, this.gsScene.splatNum, this.shaderManager.debugTF);
                    this.shaderManager.debugLog();
                }
                this.graphicsAPI.drawInstanced('TRIANGLE_FAN', 0, 4, this.sorter.getSplatSortCount());
            }
            if (this.showGizmo) {
                this.sceneHelper.renderGizmo();
            }
        }

        animate(performance.now());
    }

    runXR() {

        const animateXR = (currentTime, xrFrame) => {
            this.xrRenderLoopHandle = this.webxr.session.requestAnimationFrame(animateXR);

            const pose = xrFrame.getViewerPose(this.webxr.refSpace);
            if (!pose) return;

            this.__updateFPS(currentTime + this.startTime);
            this.__updateCyclopeanCamera(pose);
            this.__runSplatSort(this.gsScene.forceSort(), false, true);
            this.__updateForRendererSizeChanges();
            this.webxr.updateControllers(xrFrame);

            if (this.__shouldRender()) {
                this.graphicsAPI.bindFrameBuffer(this.webxr.framebuffer);
                this.webxr.updateClearColor(0, 0, 0, 1);
                for (const view of pose.views) {
                    const viewport = this.webxr.getViewport(view);
                    this.graphicsAPI.updateViewport({x:viewport.x, y:viewport.y}, {x:viewport.width, y:viewport.height});
                    this.shaderManager.setPipeline();
                    this.__updateUniforms(Array.from(view.projectionMatrix), Array.from(view.transform.inverse.matrix), viewport);
                    this.graphicsAPI.drawInstanced('TRIANGLE_FAN', 0, 4, this.sorter.getSplatSortCount());

                    this.webxr.render(view);
                }
            }
        }

        this.webxr.frameRateControl(0, true);
        this.xrRenderLoopHandle = this.webxr.session.requestAnimationFrame(animateXR);
    }

    __shouldRender = function() {
        let isSet = false;
        return function(reset = false) {
            if (reset) {
                isSet = false;
                return;
            }
            const res = Boolean(this.gsScene.ready && this.sorter.ready && this.shaderManager.ready);
            if (!isSet && res) {
                // these states only need to set once
                this.graphicsAPI.setBlendState();
                this.shaderManager.setPipeline();
                // this.gsScene.updateTextureBindings();
                this.shaderManager.updateUniformTextures(this.gsScene.buffers);
                this.shaderManager.updateUniforms(true);
                this.gsScene.setSceneReady();
                isSet = true;
                this.eventBus.emit('noteExternalListener', {
                    sceneLoaded: true,
                    uid: this.gsScene.currentScene.uid,
                    name: this.gsScene.currentScene.name,   // do not use this.gsScene.name
                    transform: this.gsScene.currentScene.transform,
                    gsType: GSType[this.gsScene.currentScene.gsType],
                    sequential: this.gsScene.currentScene.sequential,
                });
            }

            if (this.webxr.running && this.frameCount === 0) {
                this.webxr.frameRateControl(this.fps);
            }

            return res;
        }
    }();

    __onSortForSkipFrameDone({}) {
        this.sortForSkipFrame = false;
        this.loopedTime = this.skipTimestamps.shift() || 0;
    }

    __onBuffersReady({ data, sceneName }) {
        const sceneType = data.sceneType;

        // we reset camera each time we switch scene
        if (!sceneType.generalSequentialThreeD) {
            this.resetCamera();
        } else {
            // to trigger for sorting for skip frame
            // so we skip rendering new frame until sorting done
            this.setTimestamp(this.gsScene.currentScene.currentFrame + 0.001)
        }
        this.__shouldRender(true);
        this.__runSplatSort(false, true);
        
        if (this.options.isMobile || this.webxr.running) {
            switch (data.gsType) {
                case 'ThreeD': this.alphaCullThreshold = 15 / 255; break;
                case 'SPACETIME': this.alphaCullThreshold = 35 / 255; break;
                default: break;
            }
        }

        if (sceneType.ThreeD) {
            // currently all video duration is 1 sec
            this.videoDuration = 1.0;
            this.pause = true;
            this.setTimestamp(0);
        } else if (sceneType.STG) {
            // currently all video duration is 1 sec
            this.videoDuration = 1.0;
            this.pause = false;
            this.setTimestamp(0);
        } else if (sceneType.virtualSequentialThreeD) {
            this.videoDuration = data.frameNum;
            this.pause = false;
            this.setTimestamp(0);
        }

        this.eventBus.emit('noteExternalListener', {
            updateTimestamp: true,
            timestamp: this.loopedTime,
            duration: this.videoDuration,
        });
    }

    __onXrSessionEnd({}) {
        this.graphicsAPI.bindFrameBuffer(null);
        this.xrRenderLoopHandle = null;
        this.controls.enabled = true;
        this.__shouldRender(true);
        this.__runSplatSort(false, true);
        this.run();
    }

    __updateForRendererSizeChanges = function() {

        const lastRendererSize = new THREE.Vector2();
        const currentRendererSize = new THREE.Vector2();

        return function() {
            currentRendererSize.x = Math.round(this.canvas.clientWidth * this.devicePixelRatio);
            currentRendererSize.y = Math.round(this.canvas.clientHeight * this.devicePixelRatio);

            if (currentRendererSize.x !== lastRendererSize.x || currentRendererSize.y !== lastRendererSize.y) {
                this.canvas.width = currentRendererSize.x;
                this.canvas.height = currentRendererSize.y;
                this.camera.aspect = currentRendererSize.x / currentRendererSize.y;
                this.camera.updateProjectionMatrix();
                lastRendererSize.copy(currentRendererSize);
                this.sceneHelper._onAspectChanged();
            }
        };
    }();

    __updateUniforms = function() {
        const newViewMatrix = new THREE.Matrix4();
        const xrViewMatrix = new THREE.Matrix4();

        return function(proj = null, view = null, viewport = null) {
            if (this.options.enablePointerLock && this.controls === this.pointerLockControls) {
                //this.camera.updateMatrixWorld();
            }
            const projMat = proj || this.camera.projectionMatrix.elements;
            newViewMatrix.copy(this.gsScene.currentScene.modelMatrix);
            newViewMatrix.premultiply(Boolean(view) ? xrViewMatrix.fromArray(view) : this.camera.matrixWorldInverse);

            this.shaderManager.updateUniform('viewMatrix', newViewMatrix.elements);
            this.shaderManager.updateUniform('projectionMatrix', projMat);
            // this.shaderManager.updateUniform('cameraPosition', this.camera.position.toArray(), true);
            const viewportSize = viewport || { width: this.canvas.width, height: this.canvas.height };
            const focalX = projMat[0] * 0.5 * viewportSize.width;
            const focalY = projMat[5] * 0.5 * viewportSize.height;
            this.shaderManager.updateUniform('focal', [focalX, focalY], true);
            this.shaderManager.updateUniform('invViewport', [1 / viewportSize.width, 1 / viewportSize.height], true);
            this.shaderManager.updateUniform('timestamp', this.loopedTime);
            this.shaderManager.updateUniform('renderMode', this.renderMode, true);
            this.shaderManager.updateUniform('alphaCullThreshold', this.alphaCullThreshold, true);

            this.shaderManager.updateUniforms();
        }
    }();

    __runSplatSort = function() {
        let sortOnceForNewScene = true;
        const mvpMatrix = new THREE.Matrix4();
        const cameraPositionArray = [];
        const lastSortViewDir = new THREE.Vector3(0, 0, -1);
        const sortViewDir = new THREE.Vector3(0, 0, -1);
        const lastSortViewPos = new THREE.Vector3();
        const sortViewOffset = new THREE.Vector3();

        return function(force = false, reset = false, useCyclopeanCamera = false) {
            if (reset) {
                sortOnceForNewScene = true;
                return Promise.resolve(false);
            }
            if (!this.sorter.ready) return Promise.resolve(false);
            if (this.sorter.sortRunning) return Promise.resolve(true);
            // we sort all splats
            // culling on wasm if chunkBased, or we just sort all splats
            if (this.gsScene.splatNum <= 0) {
                return Promise.resolve(false);
            }
            const camera = useCyclopeanCamera ? this.cyclopeanCamera : this.camera;

            sortViewDir.set(0, 0, -1).applyQuaternion(camera.quaternion);
            const angleDiff = sortViewDir.dot(lastSortViewDir);
            const positionDiff = sortViewOffset.copy(camera.position).sub(lastSortViewPos).length();
            if ((angleDiff < 0.995 || positionDiff >= 0.5) && !useCyclopeanCamera) {
                this.eventBus.emit('noteExternalListener', {
                    cameraUpdate: true,
                    position: camera.position,
                    look: this.controls.type === 'orbit' ? this.controls.target : sortViewDir,
                    up: camera.up,
                });
            }

            if (!(force || sortOnceForNewScene)) {
                let needsRefreshForRotation = false;
                let needsRefreshForPosition = false;
                if (angleDiff < 0.995) needsRefreshForRotation = true;
                if (positionDiff >= 1.0) needsRefreshForPosition = true;
                if (!needsRefreshForRotation && !needsRefreshForPosition) return Promise.resolve(false);
            }
            sortOnceForNewScene = false;

            // start to sort
            this.sorter.sortRunning = true;

            mvpMatrix.copy(this.gsScene.currentScene.modelMatrix);
            mvpMatrix.premultiply(camera.matrixWorldInverse);
            mvpMatrix.premultiply(camera.projectionMatrix);

            cameraPositionArray[0] = camera.position.x;
            cameraPositionArray[1] = camera.position.y;
            cameraPositionArray[2] = camera.position.z;

            const timestamp = this.sortForSkipFrame ? (this.skipTimestamps[0] || 0) : this.loopedTime;
            this.sorter.sort(mvpMatrix, cameraPositionArray, timestamp, this.sortForSkipFrame);

            lastSortViewPos.copy(camera.position);
            lastSortViewDir.copy(sortViewDir);
        };

    }();

    __resolveOptions() {
        // iOS makes choosing the right WebAssembly configuration tricky :(
        //const iOSSemVer = Utils.isIOS() ? Utils.getIOSSemever() : null;
        //if (iOSSemVer) {
        //    this.options.sharedMemoryForWorkers = this.options.sharedMemoryForWorkers && !(iOSSemVer.major <= 16 && iOSSemVer.minor < 4);
        //}
        // sharedMemoryForWorkers feature has not finished
        this.options.sharedMemoryForWorkers = false;

        this.options.initialCameraPosition = new THREE.Vector3().fromArray(this.options.initialCameraPosition || [0, 0, -2]);
        this.options.cameraUp = new THREE.Vector3().fromArray(this.options.cameraUp || [0, -1, 0]);
        this.options.initialCameraLookAt = new THREE.Vector3().fromArray(this.options.initialCameraLookAt || [0, 0, 0]);
        this.options.cameraFOV = this.options.cameraFOV || 60;

        this.options.isMobile = Utils.isMobile();

        this.options.destroyOnLoad = this.options.isMobile;
        this.options.cacheShaders = !this.options.isMobile;

        if (this.options.enablePointerLock) {
            if (!document.getElementById('blocker') || this.options.isMobile) {
                console.warn("Warn: Blocker element with ID 'blocker' not found. Set 'enablePointerLock' to False");
                this.options.enablePointerLock = false;
            }
        }
    }

    __setupCamera() {
        const renderDimensions = new THREE.Vector2(this.width, this.height);

        this.perspectiveCamera = new THREE.PerspectiveCamera(this.cameraFOV, renderDimensions.x / renderDimensions.y, 0.1, 1000);
        this.camera = this.perspectiveCamera;
        this.camera.position.copy(this.initialCameraPosition);
        this.camera.up.copy(this.cameraUp).normalize();
        this.camera.lookAt(this.initialCameraLookAt);
    }

    __onKeyDown(event) {
        switch (event.code) {
            case 'KeyQ': this.isQDown = true; break;
            case 'KeyE': this.isEDown = true; break;
        }
        if (this.controls !== this.pointerLockControls) return;
        switch (event.code) {
            case 'KeyW': this.isWDown = true; break;
            case 'KeyA': this.isADown = true; break;
            case 'KeyS': this.isSDown = true; break;
            case 'KeyD': this.isDDown = true; break;
            case 'KeyY': this.isYDown = true; break;
            case 'Space': this.isSpaceDown = true; break;
            case 'ControlLeft': this.isLeftCtrlDown = true; break;
        }
    }

    __onKeyUp(event) {
        switch (event.code) {
            case 'KeyW': this.isWDown = false; break;
            case 'KeyA': this.isADown = false; break;
            case 'KeyS': this.isSDown = false; break;
            case 'KeyD': this.isDDown = false; break;
            case 'KeyY': this.isYDown = false; this.freezeY = !this.freezeY; break;
            case 'KeyQ': this.isQDown = false; break;
            case 'KeyE': this.isEDown = false; break;
            case 'Space': this.isSpaceDown = false; break;
            case 'ControlLeft': this.isLeftCtrlDown = false; break;
        }
    }

    __onMouseDown(event) {
        // event.button === 0 is the left mouse button
        if (event.button === 0) {
            this.isLeftMouseDown = true;
            if (this.pointerLockControls) {
                this.pointerLockControls.isLeftMouseDown = this.isLeftMouseDown;
            }
            this.eventBus.emit('noteExternalListener', {
                startDrag: true,
            });
        }
    }

    __onMouseUp(event) {
        // event.button === 0 is the left mouse button
        if (event.button === 0) {
            this.isLeftMouseDown = false;
            if (this.pointerLockControls) {
                this.pointerLockControls.isLeftMouseDown = this.isLeftMouseDown;
            }
            this.sceneHelper._onMouseUp();
            this.eventBus.emit('noteExternalListener', {
                endDrag: true,
            });
        }
    }

    __onMouseMove(event) {
        if (this.pointerLockControls) {
            this.pointerLockControls._onMouseMove(event);
        }
        this.sceneHelper._onMouseMove(event);
    }

    __setupControls() {
        this.orbitControls = new OrbitControls(this.camera, this.canvas);
        this.orbitControls.listenToKeyEvents(window);
        this.orbitControls.rotateSpeed = 0.5;
        //this.orbitControls.maxPolarAngle = Math.PI * .75;
        //this.orbitControls.minPolarAngle = 0.1;
        this.orbitControls.enableDamping = true;
        this.orbitControls.dampingFactor = 0.05;
        this.orbitControls.target.copy(this.initialCameraLookAt);
        this.orbitControls.update();

        // TODO: add tuo luo yi for mobile device
        if (!this.options.isMobile) {
            this.pointerLockControls = new PointerLockControls(this.camera, this.canvas);
            if (this.options.enablePointerLock) {
                this.pointerLockControls.addEventListener('lock', () => {
                    document.getElementById('blocker').style.display = 'none';
                });
                this.pointerLockControls.addEventListener('unlock', () => {
                    document.getElementById('blocker').style.display = 'block';
                });
            }
            this.pointerLockControls.enabled = false;
        }
        document.addEventListener('keydown', this.__onKeyDown.bind(this));
        document.addEventListener('keyup', this.__onKeyUp.bind(this));
        this.canvas.addEventListener('mousedown', this.__onMouseDown.bind(this));
        this.canvas.addEventListener('mouseup', this.__onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.__onMouseMove.bind(this));
        
        this.controls = this.orbitControls;
        this.controls.type = 'orbit';
        this.controls.update();
        this.camera.controls = this.controls;

    }

    __updatePointerLockMovement() {
        const t = this.deltaT / 1000; // Convert delta time from ms to seconds

        // Apply friction/decay to all three velocity components
        this.velocity.x -= this.velocity.x * 10.0 * t;
        this.velocity.z -= this.velocity.z * 10.0 * t;
        this.velocity.y -= this.velocity.y * 10.0 * t; // Added for vertical movement

        // --- Horizontal Movement (XZ Plane) ---
        this.direction.z = Number(this.isWDown) - Number(this.isSDown);
        this.direction.x = Number(this.isDDown) - Number(this.isADown);
        this.direction.normalize(); // Ensure consistent speed in all horizontal directions

        if (this.isWDown || this.isSDown) {
            this.velocity.z -= this.direction.z * 100.0 * t;
        }
        if (this.isADown || this.isDDown) {
            this.velocity.x -= this.direction.x * 100.0 * t;
        }

        // --- Vertical Movement (Y Axis) ---
        // Calculate vertical direction based on Space and Left Ctrl keys
        const yDirection = Number(this.isSpaceDown) - Number(this.isLeftCtrlDown);

        if (this.isSpaceDown || this.isLeftCtrlDown) {
            // Apply force to the vertical velocity
            this.velocity.y -= yDirection * 100.0 * t;
        }

        // --- Apply all movements to the controls ---
        this.pointerLockControls.moveRight(-this.velocity.x * t);
        this.pointerLockControls.moveForward(-this.velocity.z * t, this.freezeY);
        this.pointerLockControls.moveUp(-this.velocity.y * t); // Call the new moveUp method
    }

    __updateFPS(currentTime) {
        this.frameCount++;
        if (currentTime - this.lastFPSTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastFPSTime = currentTime;
        }
        this.deltaT = currentTime - this.lastFrameTime;
        this.lastFrameTime = currentTime;
        if (!this.pause && !this.isDraggingTimeline) {
            this.loopedTime += this.deltaT / 1000 * this.playSpeed; // ms => s
            // for 4dgs, sorting cannot catch up with rendering
            // therefore, the rendering of the first frame may use the sorted indices of the last frame
            // which may cause inconsistency and flash
            // To solve this, we wait for the sorting for the first frame is done, meanwhile keep rendering the last frame
            if (!this.sortForSkipFrame && this.loopedTime > this.videoDuration) {
                this.sortForSkipFrame = true;
            }
            this.loopedTime = Math.min(this.videoDuration, this.loopedTime);

            this.eventBus.emit('noteExternalListener', {
                updateTimestamp: true,
                timestamp: this.loopedTime,
                duration: this.videoDuration,
            });
        }

        // we also update current frame for sequential 3dgs, 'cause it's similar with this.loopedTime
        this.gsScene.updateVirtualSequentialThreeDFrame(this.loopedTime);
        if (!this.sortForSkipFrame && this.skipTimestamps[0] !== undefined) {
            this.sortForSkipFrame = true;
        }
    }

    __updateCyclopeanCamera = function() {
        const leftPos = new THREE.Vector3();
        const rightPos = new THREE.Vector3();
        const midPos = new THREE.Vector3();
        const viewDir = new THREE.Vector3();

        return function(poses) {
            const views = poses.views;

            if (views.length === 1) {
                const view = views[0];

                this.cyclopeanCamera.matrixWorldInverse.fromArray(Array.from(view.transform.inverse.matrix));
                this.cyclopeanCamera.projectionMatrix.fromArray(Array.from(view.projectionMatrix));
            } else if (views.length === 2) {
                const leftViewIndex = views[0].eye === 'left' ? 0 : 1;
                const leftView = views[leftViewIndex];
                const rightView = views[1 - leftViewIndex];

                const leftViewAtan = Utils.getTanHalfFovFromProj(leftView.projectionMatrix);
                const rightViewAtan = Utils.getTanHalfFovFromProj(rightView.projectionMatrix);

                // view matrix
                const leftViewMat = leftView.transform.inverse.matrix;  // Float32Array
                const rightViewMat = rightView.transform.inverse.matrix; // Float32Array
                const targetViewMat = this.cyclopeanCamera.matrixWorldInverse;

                leftPos.set(-leftViewMat[12], -leftViewMat[13], -leftViewMat[14]);
                rightPos.set(-rightViewMat[12], -rightViewMat[13], -rightViewMat[14]);
                midPos.copy(leftPos).add(rightPos).multiplyScalar(0.5);

                viewDir.set(leftViewMat[8], leftViewMat[9], leftViewMat[10]);
                const halfDistBetweenEyes = leftPos.distanceTo(rightPos) * 0.5;
                const dirOffset = halfDistBetweenEyes / leftViewAtan.left;
                midPos.sub(viewDir.multiplyScalar(dirOffset));

                targetViewMat.fromArray(Array.from(leftViewMat));
                targetViewMat.elements[12] = -midPos.x;
                targetViewMat.elements[13] = -midPos.y;
                targetViewMat.elements[14] = -midPos.z;

                // proj matrix
                const near = this.webxr.depthNear + dirOffset;
                const far = this.webxr.depthFar + dirOffset;
                this.cyclopeanCamera.projectionMatrix.makePerspective(
                    -near * leftViewAtan.left,
                    near * rightViewAtan.right,
                    near * Math.max(leftViewAtan.top, rightViewAtan.top),
                    -near * Math.max(leftViewAtan.bottom, rightViewAtan.bottom),
                    near,
                    far
                );
            }

            this.cyclopeanCamera.matrixWorld.copy(this.cyclopeanCamera.matrixWorldInverse).invert();
            this.cyclopeanCamera.matrixWorld.decompose(
                this.cyclopeanCamera.position,
                this.cyclopeanCamera.quaternion,
                this.cyclopeanCamera.scale
            );
        }
    }();

    __updateControls() {
        this.__updateCameraRotate();
        if (this.controls == this.orbitControls) {
            this.controls.update();
        } else if (this.controls == this.pointerLockControls && (this.useAsFlyControls || (!this.useAsFlyControls && this.controls.isLocked))) {
            this.__updatePointerLockMovement();
        }
    }

    __updateCameraRotate = function() {
        const forwardVector = new THREE.Vector3();
        const target = new THREE.Vector3(); 
        const rollSpeed = 0.001;

        return function() {
            this.camera.getWorldDirection(forwardVector);
            let update = false;
            if (this.isQDown) {
                this.camera.up.applyAxisAngle(forwardVector, rollSpeed * this.deltaT);
                update = true;
            }
            if (this.isEDown) {
                this.camera.up.applyAxisAngle(forwardVector, -rollSpeed * this.deltaT);
                update = true;
            }

            if (update) {
                if (this.controls === this.pointerLockControls) {
                    target.copy(this.camera.position).add(forwardVector);
                    this.camera.lookAt(target);
                }
                this.eventBus.emit('noteExternalListener', {
                    cameraUpdate: true,
                    position: this.camera.position,
                    look: this.controls.type === 'orbit' ? this.controls.target : forwardVector,
                    up: this.camera.up,
                })
            }
        }
    }();
}