import { Lines } from "./lines.js";
import * as THREE from "three";
import { Tween, Group, Easing } from '@tweenjs/tween.js';

export class Grid {
    constructor(camera, graphicsAPI) {
        this.camera = camera;
        this.graphicsAPI = graphicsAPI;

        // Line
        this.lines = new Lines(this.graphicsAPI);

        const halfNum = 50;
        const minF = -halfNum;
        const maxF = halfNum;
        this.lines.addLine({ start: [minF, 0, 0, 0.84, 0.28, 0.28, 1], end: [maxF, 0, 0, 0.84, 0.28, 0.28, 1] });
        this.lines.addLine({ start: [0, 0, minF, 0.20, 0.50, 0.82, 1], end: [0, 0, maxF, 0.20, 0.50, 0.82, 1] });

        for (let i = 1; i<halfNum; ++i) {
            this.lines.addLine({ start: [minF, 0, i, 0.29, 0.29, 0.29, 1], end: [maxF, 0, i, 0.29, 0.29, 0.29, 1] });
            this.lines.addLine({ start: [i, 0, minF, 0.29, 0.29, 0.29, 1], end: [i, 0, maxF, 0.29, 0.29, 0.29, 1] });
            this.lines.addLine({ start: [minF, 0, -i, 0.29, 0.29, 0.29, 1], end: [maxF, 0, -i, 0.29, 0.29, 0.29, 1] });
            this.lines.addLine({ start: [-i, 0, minF, 0.29, 0.29, 0.29, 1], end: [-i, 0, maxF, 0.29, 0.29, 0.29, 1] });
        }

        this.lines.updateBuffers();
    }

    render = function() {
        
        return function(){
            // lines pipeline
            this.lines.render(this.camera.projectionMatrix.elements, this.camera.matrixWorldInverse.elements);
        }
        
    }();
}