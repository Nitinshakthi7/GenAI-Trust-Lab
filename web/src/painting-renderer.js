const vertexSource = `
  attribute vec2 position;
  varying vec2 uv;

  void main() {
    gl_Position = vec4(position, 0., 1.);
    uv = position * vec2(.5, -.5) + .5;
  }
`;

export function createPaintingRenderer(canvas, fragmentSource) {
  const gl = canvas.getContext('webgl', { alpha: false, antialias: false });
  if (!gl || gl.isContextLost()) return;

  const program = gl.createProgram();
  const shaders = [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]];
  for (const [type, source] of shaders) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    gl.attachShader(program, shader);
    gl.deleteShader(shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return;
  }
  gl.useProgram(program);

  const bounds = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bounds);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.uniform1i(gl.getUniformLocation(program, 'painting'), 0);
  const time = gl.getUniformLocation(program, 'seconds');
  let source;

  return {
    draw(image, seconds) {
      if (source !== image.currentSrc) {
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
        source = image.currentSrc;
      }
      gl.uniform1f(time, seconds);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    },
    dispose() {
      gl.deleteTexture(texture);
      gl.deleteBuffer(bounds);
      gl.deleteProgram(program);
    },
  };
}
