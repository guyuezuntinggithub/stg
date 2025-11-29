export class Lines {
    constructor(graphicsAPI) {
        this.graphicsAPI = graphicsAPI;
        this.vertices = [];
        // --- 3D 着色器 ---
        const vsSource = `#version 300 es
            in vec3 a_position;
            in vec4 a_color;
            uniform mat4 u_projection;
            uniform mat4 u_view;

            out vec4 v_color;
            
            void main() {
                gl_Position = u_projection * u_view * vec4(a_position, 1.0);
                v_color = a_color;
            }
        `;
        const fsSource = `#version 300 es
            precision mediump float;
            in vec4 v_color;
            out vec4 outColor;
            void main() {
                outColor = v_color;
            }
        `;
        this.program = this.graphicsAPI.setupProgram(vsSource, fsSource);
        this.uniforms = this.graphicsAPI.getUniform(this.program);
        this.attributes = this.graphicsAPI.getAttrib(this.program);

        this.vao = this.graphicsAPI.setupLineVAO(this.attributes['a_position'], this.attributes['a_color']);
    }

    /**
     * 添加一条3D线段。
     * @param {{start: [number, number, number], end: [number, number, number]}} line
     */
    addLine({ start, end }) {
        this.vertices.push(...start, ...end);
    }

    clear() {
        this.vertices = [];
    }

    updateBuffers(dynamic = false) {
        this.graphicsAPI.initBuffer(this.vao.buffer, new Float32Array(this.vertices), dynamic ? 'DYNAMIC_DRAW' : 'STATIC_DRAW');
    }

    /**
     * 渲染所有线段。
     * @param {{projectionMatrix: number[], viewMatrix: number[], color: number[], lineWidth: number}} options
     */
    render(projectionMatrix, viewMatrix) {
        // 每个顶点现在是3个分量
        const vertexCount = this.vertices.length / 7;
        if (vertexCount === 0) return;
        this.graphicsAPI.updateProgram(this.program);
        this.graphicsAPI.updateVertexInput(this.vao.vao);
        this.graphicsAPI.graphicsAPI.lineWidth(1.0);
        
        this.graphicsAPI.updateUniform(this.uniforms.u_projection, 'Matrix4fv', projectionMatrix, false);
        this.graphicsAPI.updateUniform(this.uniforms.u_view, 'Matrix4fv', viewMatrix, false);
        
        this.graphicsAPI.draw('LINES', 0, vertexCount);
        this.graphicsAPI.updateVertexInput(null);
    }

    dispose() {
        this.graphicsAPI.deleteVertexArray(this.vao.vao);
        this.graphicsAPI.deleteBuffer(this.vao.buffer);
        this.graphicsAPI.deleteProgram(this.program);
    }
}