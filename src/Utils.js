
// Fast Half Float Conversions, http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
const _tables = /*@__PURE__*/ _generateTables();
function _generateTables() {
	// float32 to float16 helpers
	const buffer = new ArrayBuffer( 4 );
	const floatView = new Float32Array( buffer );
	const uint32View = new Uint32Array( buffer );
	const baseTable = new Uint32Array( 512 );
	const shiftTable = new Uint32Array( 512 );
	for ( let i = 0; i < 256; ++ i ) {
		const e = i - 127;
		// very small number (0, -0)
		if ( e < - 27 ) {
			baseTable[ i ] = 0x0000;
			baseTable[ i | 0x100 ] = 0x8000;
			shiftTable[ i ] = 24;
			shiftTable[ i | 0x100 ] = 24;
			// small number (denorm)
		} else if ( e < - 14 ) {
			baseTable[ i ] = 0x0400 >> ( - e - 14 );
			baseTable[ i | 0x100 ] = ( 0x0400 >> ( - e - 14 ) ) | 0x8000;
			shiftTable[ i ] = - e - 1;
			shiftTable[ i | 0x100 ] = - e - 1;
			// normal number
		} else if ( e <= 15 ) {
			baseTable[ i ] = ( e + 15 ) << 10;
			baseTable[ i | 0x100 ] = ( ( e + 15 ) << 10 ) | 0x8000;
			shiftTable[ i ] = 13;
			shiftTable[ i | 0x100 ] = 13;
			// large number (Infinity, -Infinity)
		} else if ( e < 128 ) {
			baseTable[ i ] = 0x7c00;
			baseTable[ i | 0x100 ] = 0xfc00;
			shiftTable[ i ] = 24;
			shiftTable[ i | 0x100 ] = 24;
			// stay (NaN, Infinity, -Infinity)
		} else {
			baseTable[ i ] = 0x7c00;
			baseTable[ i | 0x100 ] = 0xfc00;
			shiftTable[ i ] = 13;
			shiftTable[ i | 0x100 ] = 13;
		}
	}
	// float16 to float32 helpers
	const mantissaTable = new Uint32Array( 2048 );
	const exponentTable = new Uint32Array( 64 );
	const offsetTable = new Uint32Array( 64 );
	for ( let i = 1; i < 1024; ++ i ) {
		let m = i << 13; // zero pad mantissa bits
		let e = 0; // zero exponent
		// normalized
		while ( ( m & 0x00800000 ) === 0 ) {
			m <<= 1;
			e -= 0x00800000; // decrement exponent
		}
		m &= ~ 0x00800000; // clear leading 1 bit
		e += 0x38800000; // adjust bias
		mantissaTable[ i ] = m | e;
	}
	for ( let i = 1024; i < 2048; ++ i ) {
		mantissaTable[ i ] = 0x38000000 + ( ( i - 1024 ) << 13 );
	}
	for ( let i = 1; i < 31; ++ i ) {
		exponentTable[ i ] = i << 23;
	}
	exponentTable[ 31 ] = 0x47800000;
	exponentTable[ 32 ] = 0x80000000;
	for ( let i = 33; i < 63; ++ i ) {
		exponentTable[ i ] = 0x80000000 + ( ( i - 32 ) << 23 );
	}
	exponentTable[ 63 ] = 0xc7800000;
	for ( let i = 1; i < 64; ++ i ) {
		if ( i !== 32 ) {
			offsetTable[ i ] = 1024;
		}
	}
	return {
		floatView: floatView,
		uint32View: uint32View,
		baseTable: baseTable,
		shiftTable: shiftTable,
		mantissaTable: mantissaTable,
		exponentTable: exponentTable,
		offsetTable: offsetTable
	};

}

// float32 to float16
function toHalfFloat( val ) {
	//if ( Math.abs( val ) > 65504 ) console.warn( 'THREE.DataUtils.toHalfFloat(): Value out of range.' );
	val = Utils.clamp( val, - 65504, 65504 );
	_tables.floatView[ 0 ] = val;
	const f = _tables.uint32View[ 0 ];
	const e = ( f >> 23 ) & 0x1ff;
	return _tables.baseTable[ e ] + ( ( f & 0x007fffff ) >> _tables.shiftTable[ e ] );
}

// float16 to float32
function fromHalfFloat( val ) {
	const m = val >> 10;
	_tables.uint32View[ 0 ] = _tables.mantissaTable[ _tables.offsetTable[ m ] + ( val & 0x3ff ) ] + _tables.exponentTable[ m ];
	return _tables.floatView[ 0 ];
}

let getFloat16Available;
if ('getFloat16' in DataView.prototype) {
    getFloat16Available = true;
} else {
    getFloat16Available = false;
}

export class Utils {
    static f2fp162uint16 = toHalfFloat;
    static uint162fp162f = fromHalfFloat;

    static readFp16(dataview, byteOffset, littleEndian) {
        if (getFloat16Available) {
            return dataview.getFloat16(byteOffset, littleEndian);
        } else {
            return fromHalfFloat(dataview.getUint16(byteOffset, littleEndian));
        }
    }

    static valueChanged(oldValue, newValue) {
        if (typeof oldValue !== typeof newValue) {
            return true;
        }

        if (Array.isArray(newValue)) {
            if (!Array.isArray(oldValue) || oldValue.length !== newValue.length) {
                return true;
            }

            for (let i = 0; i < newValue.length; i++) {
                if (oldValue[i] !== newValue[i]) {
                    return true;
                }
            }
            return false;
        }

        return oldValue !== newValue;
    }

    static getRandomUID() {
        return crypto.randomUUID();
    }

    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    static clamp(x, min, max) {
        return Math.max(Math.min(x, max), min);
    }

    static exactDivideUp(a, divide) {
        return Math.floor((a + divide - 1) / divide);
    }

    static alignUp(a, alignment) {
        return Utils.exactDivideUp(a, alignment) * alignment;
    }

    // x is expected to be between [0, 1]
    static float2uint8(x, min = 0, max = 1) {
        return Utils.float2uintX(x, 8, min, max);
    }

    static uint82float(x, min = 0, max = 1) {
        return Utils.uintX2float(x, 8, min, max);
    }

    static float2uintX(x, X, min = 0, max = 1) {
        const range = (1 << X) - 1;
        return Utils.clamp(Math.round((x - min) / (max - min) * range), 0, range);
    }

    static uintX2float(x, X, min = 0, max = 1) {
        const range = (1 << X) - 1;
        return Utils.clamp(x, 0, range) / range * (max - min) + min;
    }

    static packFloat2rgba(r, g, b, a, out, offset = 0) {
        Utils.pack4Float2Uint32(r, g, b, a, 0, 1, out, offset);
    }

    static pack4Float2Uint32(x, y, z, w, min, max, out, offset = 0) {
        if (Array.isArray(min) && Array.isArray(max)) {
            if (min.length !== 4 || max.length !== 4) {
                throw new Error("min.length, max.length should be 4");
            }
            out.setUint8(offset + 0, Utils.float2uint8(x, min[0], max[0]));
            out.setUint8(offset + 1, Utils.float2uint8(y, min[1], max[1]));
            out.setUint8(offset + 2, Utils.float2uint8(z, min[2], max[2]));
            out.setUint8(offset + 3, Utils.float2uint8(w, min[3], max[3]));
        } 
        else if (typeof min === 'number' && typeof max === 'number') {
            out.setUint8(offset + 0, Utils.float2uint8(x, min, max));
            out.setUint8(offset + 1, Utils.float2uint8(y, min, max));
            out.setUint8(offset + 2, Utils.float2uint8(z, min, max));
            out.setUint8(offset + 3, Utils.float2uint8(w, min, max));
        } 
        else {
            throw new Error("min, max should be both either Number or Array");
        }
    }

    static computeCov3dPack2fp16 = function() {
        class Matrix3 {
            constructor( n11, n12, n13, n21, n22, n23, n31, n32, n33 ) {Matrix3.prototype.isMatrix3 = true;this.elements = [1, 0, 0, 0, 1, 0, 0, 0, 1];if ( n11 !== undefined ) {this.set( n11, n12, n13, n21, n22, n23, n31, n32, n33 );}}
            set( n11, n12, n13, n21, n22, n23, n31, n32, n33 ) {const te = this.elements;te[ 0 ] = n11; te[ 1 ] = n21; te[ 2 ] = n31;te[ 3 ] = n12; te[ 4 ] = n22; te[ 5 ] = n32;te[ 6 ] = n13; te[ 7 ] = n23; te[ 8 ] = n33;return this;}
            identity() {this.set(1, 0, 0,0, 1, 0,0, 0, 1);return this;}
            copy( m ) {const te = this.elements;const me = m.elements;te[ 0 ] = me[ 0 ]; te[ 1 ] = me[ 1 ]; te[ 2 ] = me[ 2 ];te[ 3 ] = me[ 3 ]; te[ 4 ] = me[ 4 ]; te[ 5 ] = me[ 5 ];te[ 6 ] = me[ 6 ]; te[ 7 ] = me[ 7 ]; te[ 8 ] = me[ 8 ];return this;}
            multiply( m ) {return this.multiplyMatrices( this, m );}
            premultiply( m ) {return this.multiplyMatrices( m, this );}
            multiplyMatrices( a, b ) {const ae = a.elements;const be = b.elements;const te = this.elements;const a11 = ae[ 0 ], a12 = ae[ 3 ], a13 = ae[ 6 ];const a21 = ae[ 1 ], a22 = ae[ 4 ], a23 = ae[ 7 ];const a31 = ae[ 2 ], a32 = ae[ 5 ], a33 = ae[ 8 ];const b11 = be[ 0 ], b12 = be[ 3 ], b13 = be[ 6 ];const b21 = be[ 1 ], b22 = be[ 4 ], b23 = be[ 7 ];const b31 = be[ 2 ], b32 = be[ 5 ], b33 = be[ 8 ];te[ 0 ] = a11 * b11 + a12 * b21 + a13 * b31;te[ 3 ] = a11 * b12 + a12 * b22 + a13 * b32;te[ 6 ] = a11 * b13 + a12 * b23 + a13 * b33;te[ 1 ] = a21 * b11 + a22 * b21 + a23 * b31;te[ 4 ] = a21 * b12 + a22 * b22 + a23 * b32;te[ 7 ] = a21 * b13 + a22 * b23 + a23 * b33;te[ 2 ] = a31 * b11 + a32 * b21 + a33 * b31;te[ 5 ] = a31 * b12 + a32 * b22 + a33 * b32;te[ 8 ] = a31 * b13 + a32 * b23 + a33 * b33;return this;}
            transpose() {let tmp;const m = this.elements;tmp = m[ 1 ]; m[ 1 ] = m[ 3 ]; m[ 3 ] = tmp;tmp = m[ 2 ]; m[ 2 ] = m[ 6 ]; m[ 6 ] = tmp;tmp = m[ 5 ]; m[ 5 ] = m[ 7 ]; m[ 7 ] = tmp;return this;}
            fromArray( array, offset = 0 ) {for ( let i = 0; i < 9; i ++ ) {this.elements[ i ] = array[ i + offset ];}return this;}
            clone() {return new this.constructor().fromArray( this.elements );}
        }
        const scaleMatrix = new Matrix3();
        const rotationMatrix = new Matrix3();
        const covarianceMatrix = new Matrix3();
        const transformedCovariance = new Matrix3();
        const transform3x3 = new Matrix3();
        const transform3x3Transpose = new Matrix3();

        return function(sx, sy, sz, rx, ry, rz, rw, out, offset = 0, transform = null) {
            scaleMatrix.elements[0] = sx;
            scaleMatrix.elements[4] = sy;
            scaleMatrix.elements[8] = sz;

            const inv_length = 1 / Math.sqrt( rx * rx + ry * ry + rz * rz + rw * rw );
            const x = rx * inv_length, y = ry * inv_length, z = rz * inv_length, w = rw * inv_length;
            const x2 = x + x,	y2 = y + y, z2 = z + z;
		    const xx = x * x2, xy = x * y2, xz = x * z2;
		    const yy = y * y2, yz = y * z2, zz = z * z2;
		    const wx = w * x2, wy = w * y2, wz = w * z2;
            const te = rotationMatrix.elements;
            te[0] = ( 1 - ( yy + zz ) );
		    te[1] = ( xy + wz );
		    te[2] = ( xz - wy );
		    te[3] = ( xy - wz );
		    te[4] = ( 1 - ( xx + zz ) );
		    te[5] = ( yz + wx );
		    te[6] = ( xz + wy );
		    te[7] = ( yz - wx );
		    te[8] = ( 1 - ( xx + yy ) );

            covarianceMatrix.copy(rotationMatrix).multiply(scaleMatrix);
            transformedCovariance.copy(covarianceMatrix).transpose().premultiply(covarianceMatrix);

            if (transform) {
                transform3x3.setFromMatrix4(transform);
                transform3x3Transpose.copy(transform3x3).transpose();
                transformedCovariance.multiply(transform3x3Transpose);
                transformedCovariance.premultiply(transform3x3);
            }
            out.setUint16(offset +  0, Utils.f2fp162uint16(transformedCovariance.elements[0]), true);
            out.setUint16(offset +  2, Utils.f2fp162uint16(transformedCovariance.elements[3]), true);
            out.setUint16(offset +  4, Utils.f2fp162uint16(transformedCovariance.elements[6]), true);
            out.setUint16(offset +  6, Utils.f2fp162uint16(transformedCovariance.elements[4]), true);
            out.setUint16(offset +  8, Utils.f2fp162uint16(transformedCovariance.elements[7]), true);
            out.setUint16(offset + 10, Utils.f2fp162uint16(transformedCovariance.elements[8]), true);
        };

    }();

    static computeTexSize(texelNum) {
        let log2TexelNum = Math.max(Math.ceil(Math.log2(texelNum)), 0);
        if (log2TexelNum > 24) {
            console.warn(`texelNum ${texelNum} exceeds maximum 4096 * 4096 and was clamped to maximum`);
            log2TexelNum = 24;
        }
        if (log2TexelNum % 2 === 0) {
            const sideLength = Math.pow(2, log2TexelNum / 2);
            return { width: sideLength, height: sideLength };
        } else {
            const height = Math.pow(2, Math.floor(log2TexelNum / 2));
            const width = height * 2;
            return { width, height };
        }
    }

    static compute4dgsBoundingBox(x, m1, t_center, t_radius) {
        // we just ignore m2 and m3 if there are
        // 'cause it is not worth solving quadratic equation for minor error
        const X1 = x + m1 * (t_center - t_radius);
        const X2 = x + m1 * (t_center + t_radius);
        return { min: Math.min(X1, X2), max: Math.max(X1, X2) };
    }

    static getTanHalfFovFromProj(projArray) {
        const projMatrix = projArray;
        
        const left   = (1 - projMatrix[8]) / projMatrix[0];
        const right  = (1 + projMatrix[8]) / projMatrix[0];
        const top    = (1 + projMatrix[9]) / projMatrix[5];
        const bottom = (1 - projMatrix[9]) / projMatrix[5];

        return {
            top: top,
            bottom: bottom,
            left: left,
            right: right,
        };
    }

    static getFovFromProj(projArray, toDeg = false) {
        const atan = Utils.getAtanHalfFovFromProj(projArray);

        const radToDeg = toDeg ? 180 / Math.PI : 1;
        atan.left   = Math.atan(atan.left   ) * radToDeg;
        atan.right  = Math.atan(atan.right  ) * radToDeg;
        atan.top    = Math.atan(atan.top    ) * radToDeg;
        atan.bottom = Math.atan(atan.bottom ) * radToDeg;
        atan.horizon = atan.left + atan.right;
        atan.verticle = atan.bottom + atan.top;

        return atan;
    }

    static getSceneType = function() {
        const type  = {
            ThreeD: false,
            STG: false,
            generalSequentialThreeD: false,
            virtualSequentialThreeD: false,
        }
        return function(data) {
            const res = {...type};
            if (data.sequential) {
                if (data.virtual) {
                    res.virtualSequentialThreeD = true;
                } else {
                    res.generalSequentialThreeD = true;
                }
            } else {
                if (data.gsType === "ThreeD") {
                    res.ThreeD = true;
                } else if (data.gsType === "SPACETIME") {
                    res.STG = true;
                }
            }
            return res;
        };
    } ()

    static hex2rgb(hex_) {
        const hex = hex_.slice(1);
        const r = parseInt(hex.substring(0, 2), 16);
        const g = parseInt(hex.substring(2, 4), 16);
        const b = parseInt(hex.substring(4, 6), 16);
        return [r/255, g/255, b/255];
    }

    static extractFileExtension(fileName) {
        return fileName.substring(fileName.lastIndexOf('.') + 1).toLowerCase();
    }
    
    static extractFileName(fileName) {
        const fileNameWithExtension = fileName.split('/').pop().split('\\').pop();
        const name = fileNameWithExtension.split('.').slice(0, -1).join('.');
        return name;
    }

    static isFrameIdxSequential(files) {
        if (files.length <= 1) {
            return { isSequential: true, startFrame: files.length > 0 ? files[0].frameIdx : null };
        }

        const startFrame = files[0].frameIdx;

        for (let i = 1; i < files.length; i++) {
            if (files[i].frameIdx !== files[i - 1].frameIdx + 1) {
                return { isSequential: false, startFrame: startFrame };
            }
        }

        return { isSequential: true, startFrame: startFrame };
    }

    static extractFileNameIdx = function() {
        const regex = /(\d+)$/;
        let match;
        let number;
        return function(filename) {
            match = regex.exec(filename);
            number = match ? parseInt(match[1]) : -1;
            return number;
        };
    } ()

    static getAllEntries(directoryReader) {
        return new Promise(resolve => {
            const entries = [];
            const readEntries = () => {
                directoryReader.readEntries(newEntries => {
                    if (newEntries.length === 0) {
                        resolve(entries);
                    } else {
                        entries.push(...newEntries);
                        readEntries();
                    }
                });
            };
            readEntries();
        });
    }

    static async readFirstLevelDirectory(directoryEntry) {
        const directoryReader = directoryEntry.createReader();
        const entries = await Utils.getAllEntries(directoryReader);

        return entries;
    }

    static isIOS() {
        const ua = navigator.userAgent;
        return ua.indexOf('iPhone') > 0 || ua.indexOf('iPad') > 0;
    }

    static getIOSSemever() {
        if (Utils.isIOS()) {
            const extract = navigator.userAgent.match(/OS (\d+)_(\d+)_?(\d+)?/);
            return new Semver(
                parseInt(extract[1] || 0, 10),
                parseInt(extract[2] || 0, 10),
                parseInt(extract[3] || 0, 10)
            );
        } else {
            return null; // or [0,0,0]
        }
    }

    /**
     * 综合检测当前设备是否为移动端。
     * 优先使用 navigator.userAgentData，其次是CSS媒体查询，最后回退到User Agent字符串。
     * @returns {boolean} 如果是移动设备则返回true，否则返回false。
     */
    static isMobile() {
        // 1. 优先使用最新的 User-Agent Client Hints API
        if (navigator.userAgentData) {
            return navigator.userAgentData.mobile;
        }

        // 2. 其次，使用CSS媒体查询的组合判断
        // 检查主要输入设备是否为手指，且不支持悬停
        const hasCoarsePointer = window.matchMedia('(pointer: coarse)').matches;
        const lacksHoverSupport = window.matchMedia('(hover: none)').matches;
        if (hasCoarsePointer && lacksHoverSupport) {
            return true;
        }

        // 检查是否有精密的指针（鼠标），如果有，则不太可能是手机
        const hasFinePointer = window.matchMedia('(pointer: fine)').matches;
        if (hasFinePointer) {
            return false;
        }

        // 3. 作为最后的备用方案，使用传统的User Agent字符串检查
        // （虽然不完美，但能覆盖一些边缘情况）
        return /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(navigator.userAgent.toLowerCase());
    }
}