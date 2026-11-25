// Chess-board vision for the iPhone app: a port of chesslab/vision (board_detection.py,
// squares.py, piece_recognition.py, templates.py, orientation.py, turn_detection.py)
// without OpenCV. Same candidates, scores and thresholds; the Python docstrings explain
// the ideas. Runs in a Web Worker in the app and under Node for the tests.
//
// Images: RGBA input {w, h, data: Uint8ClampedArray}; internally RGB pyramids
// {w, h, data: Uint8ClampedArray (w*h*3)}; rectified boards are Float32Array RGB.
// Channel order is RGB throughout (the Python code is BGR; only grey conversion cares).

export const SCORE_WARP = 256;
export const FINE_WARP = 512;
export const MIN_BOARD_PX = 64;
export const SQUARE_PX = 64;
export const PIECE_SYMBOLS = "PNBRQKpnbrqk";

// ============================================================================ helpers

function median(values, n = values.length) {
  const a = Float64Array.from(values.subarray ? values.subarray(0, n) : values.slice(0, n)).sort();
  const m = n >> 1;
  return n % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

function clip(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function norm3(a, b) {
  const d0 = a[0] - b[0], d1 = a[1] - b[1], d2 = a[2] - b[2];
  return Math.sqrt(d0 * d0 + d1 * d1 + d2 * d2);
}

function reflect101(i, n) {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

// OpenCV's getGaussianKernel(ksize, 0)
function gaussKernel(ksize) {
  const sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;
  const k = new Float64Array(ksize);
  let sum = 0;
  for (let i = 0; i < ksize; i++) {
    const x = i - (ksize - 1) / 2;
    k[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += k[i];
  }
  for (let i = 0; i < ksize; i++) k[i] /= sum;
  return k;
}
const K5 = gaussKernel(5);
const K3 = gaussKernel(3);

// separable Gaussian blur of an interleaved float image (BORDER_REFLECT_101)
export function blur(src, w, h, ch, kernel = K5) {
  const r = kernel.length >> 1;
  const tmp = new Float32Array(w * h * ch);
  const out = new Float32Array(w * h * ch);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < ch; c++) {
        let s = 0;
        for (let i = -r; i <= r; i++) s += kernel[i + r] * src[(y * w + reflect101(x + i, w)) * ch + c];
        tmp[(y * w + x) * ch + c] = s;
      }
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < ch; c++) {
        let s = 0;
        for (let i = -r; i <= r; i++) s += kernel[i + r] * tmp[(reflect101(y + i, h) * w + x) * ch + c];
        out[(y * w + x) * ch + c] = s;
      }
    }
  }
  return out;
}

// OpenCV INTER_AREA style resize of an interleaved float image (any scale)
export function resizeArea(src, sw, sh, ch, dw, dh) {
  const weights = (sn, dn) => {
    const scale = sn / dn, rows = [];
    for (let d = 0; d < dn; d++) {
      const a = d * scale, b = (d + 1) * scale, row = [];
      for (let s = Math.floor(a); s < Math.ceil(b) && s < sn; s++) {
        const wgt = Math.min(b, s + 1) - Math.max(a, s);
        if (wgt > 1e-9) row.push([s, wgt / scale]);
      }
      rows.push(row);
    }
    return rows;
  };
  const wx = weights(sw, dw), wy = weights(sh, dh);
  const tmp = new Float32Array(dw * sh * ch);
  for (let y = 0; y < sh; y++)
    for (let x = 0; x < dw; x++)
      for (let c = 0; c < ch; c++) {
        let s = 0;
        for (const [sx, wgt] of wx[x]) s += wgt * src[(y * sw + sx) * ch + c];
        tmp[(y * dw + x) * ch + c] = s;
      }
  const out = new Float32Array(dw * dh * ch);
  for (let y = 0; y < dh; y++)
    for (let x = 0; x < dw; x++)
      for (let c = 0; c < ch; c++) {
        let s = 0;
        for (const [sy, wgt] of wy[y]) s += wgt * tmp[(sy * dw + x) * ch + c];
        out[(y * dw + x) * ch + c] = s;
      }
  return out;
}

// ============================================================================ pyramid

function rgbaToRgb(img) {
  const n = img.w * img.h, out = new Uint8ClampedArray(n * 3), d = img.data;
  for (let i = 0; i < n; i++) {
    out[i * 3] = d[i * 4];
    out[i * 3 + 1] = d[i * 4 + 1];
    out[i * 3 + 2] = d[i * 4 + 2];
  }
  return { w: img.w, h: img.h, data: out };
}

function halve(img) {
  const w = img.w >> 1, h = img.h >> 1, s = img.data, out = new Uint8ClampedArray(w * h * 3);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let c = 0; c < 3; c++) {
        const i0 = ((2 * y) * img.w + 2 * x) * 3 + c, i1 = i0 + img.w * 3;
        out[(y * w + x) * 3 + c] = (s[i0] + s[i0 + 3] + s[i1] + s[i1 + 3] + 2) >> 2;
      }
  return { w, h, data: out };
}

/** Image pyramid used for anti-aliased board warps (replaces the Python pre-blur). */
export function makePyramid(rgba) {
  const levels = [rgbaToRgb(rgba)];
  while (Math.min(levels[levels.length - 1].w, levels[levels.length - 1].h) > 64) {
    levels.push(halve(levels[levels.length - 1]));
  }
  return levels;
}

// ============================================================================ geometry

export function orderCorners(pts) {
  const p = pts.map(q => [q[0], q[1]]);
  const s = p.map(q => q[0] + q[1]), d = p.map(q => q[1] - q[0]);
  const argmin = a => a.indexOf(Math.min(...a)), argmax = a => a.indexOf(Math.max(...a));
  let out = [p[argmin(s)], p[argmin(d)], p[argmax(s)], p[argmax(d)]];
  const uniq = new Set(out.map(q => q.join(",")));
  if (uniq.size < 4) {                       // degenerate ordering -> sort by angle
    const cx = p.reduce((a, q) => a + q[0], 0) / 4, cy = p.reduce((a, q) => a + q[1], 0) / 4;
    const sorted = p.slice().sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
    const sums = sorted.map(q => q[0] + q[1]);
    const start = sums.indexOf(Math.min(...sums));
    out = sorted.slice(start).concat(sorted.slice(0, start));
  }
  return out;
}

export function rectCorners(x, y, w, h) {
  return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
}

function sides(c) {
  return [0, 1, 2, 3].map(i => Math.hypot(c[i][0] - c[(i + 1) % 4][0], c[i][1] - c[(i + 1) % 4][1]));
}

// homography taking the square [0,size]^2 (TL,TR,BR,BL) onto quad c
function squareToQuad(c, size) {
  const src = [[0, 0], [size, 0], [size, size], [0, size]];
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [u, v] = src[i], [x, y] = c[i];
    A.push([u, v, 1, 0, 0, 0, -u * x, -v * x]); b.push(x);
    A.push([0, 0, 0, u, v, 1, -u * y, -v * y]); b.push(y);
  }
  for (let col = 0; col < 8; col++) {        // Gaussian elimination with partial pivoting
    let piv = col;
    for (let r = col + 1; r < 8; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    [A[col], A[piv]] = [A[piv], A[col]];
    [b[col], b[piv]] = [b[piv], b[col]];
    for (let r = 0; r < 8; r++) {
      if (r === col) continue;
      const f = A[r][col] / A[col][col];
      for (let k = col; k < 8; k++) A[r][k] -= f * A[col][k];
      b[r] -= f * b[col];
    }
  }
  const h = b.map((v, i) => v / A[i][i]);
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

/** Perspective-rectify the board into a size x size Float32 RGB image. */
export function warpBoard(pyr, corners, size = FINE_WARP) {
  const c = orderCorners(corners);
  const side = sides(c).reduce((a, b) => a + b, 0) / 4;
  const factor = side / size;
  const level = factor > 1.3 ? Math.min(pyr.length - 1, Math.ceil(Math.log2(factor / 1.3))) : 0;
  const img = pyr[level], W = img.w, H = img.h, s = img.data, k = 2 ** level;
  const m = squareToQuad(c, size);
  const out = new Float32Array(size * size * 3);
  for (let v = 0; v < size; v++) {
    for (let u = 0; u < size; u++) {
      const den = m[6] * u + m[7] * v + 1;
      let x = (m[0] * u + m[1] * v + m[2]) / den;
      let y = (m[3] * u + m[4] * v + m[5]) / den;
      x = (x + 0.5) / k - 0.5;
      y = (y + 0.5) / k - 0.5;
      x = x < 0 ? 0 : x > W - 1 ? W - 1 : x;   // BORDER_REPLICATE
      y = y < 0 ? 0 : y > H - 1 ? H - 1 : y;
      const x0 = x | 0, y0 = y | 0, x1 = x0 + 1 < W ? x0 + 1 : x0, y1 = y0 + 1 < H ? y0 + 1 : y0;
      const fx = x - x0, fy = y - y0;
      const i00 = (y0 * W + x0) * 3, i01 = (y0 * W + x1) * 3, i10 = (y1 * W + x0) * 3, i11 = (y1 * W + x1) * 3;
      const o = (v * size + u) * 3;
      for (let ch = 0; ch < 3; ch++) {
        const top = s[i00 + ch] + (s[i01 + ch] - s[i00 + ch]) * fx;
        const bot = s[i10 + ch] + (s[i11 + ch] - s[i10 + ch]) * fx;
        out[o + ch] = Math.round(top + (bot - top) * fy);
      }
    }
  }
  return out;
}

function isConvex(c) {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = c[i], b = c[(i + 1) % 4], d = c[(i + 2) % 4];
    const cross = (b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0]);
    if (cross === 0) continue;
    const sg = Math.sign(cross);
    if (sign === 0) sign = sg;
    else if (sg !== sign) return false;
  }
  return sign !== 0;
}

function plausible(c, w, h) {
  const sd = sides(c), mn = Math.min(...sd), mx = Math.max(...sd);
  if (mn < MIN_BOARD_PX) return false;
  if (mx / Math.max(mn, 1e-6) > 1.25) return false;
  const margin = 0.08 * mx;
  const xs = c.map(p => p[0]), ys = c.map(p => p[1]);
  if (Math.min(...xs) < -margin || Math.min(...ys) < -margin) return false;
  if (Math.max(...xs) > w + margin || Math.max(...ys) > h + margin) return false;
  return isConvex(c);
}

// ============================================================================ squares

const PARITY = (r, c) => (r + c) % 2 === 0;

/** Each cell's background colour: medoid of the medians of its four corner patches. */
export function cornerBackground(F, n) {
  const s = Math.floor(n / 8);
  const inset = Math.max(1, Math.round(s * 0.06)), p = Math.max(2, Math.round(s * 0.14));
  const a0 = inset, b0 = s - inset - p;
  const starts = [[a0, a0], [a0, b0], [b0, a0], [b0, b0]];
  const buf = new Float32Array(p * p);
  const bg = new Float32Array(64 * 3);
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const meds = starts.map(([py, px]) => [0, 1, 2].map(ch => {
        let k = 0;
        for (let y = 0; y < p; y++)
          for (let x = 0; x < p; x++) buf[k++] = F[((r * s + py + y) * n + c * s + px + x) * 3 + ch];
        return median(buf, k);
      }));
      let best = 0, bestD = Infinity;
      for (let i = 0; i < 4; i++) {
        let d = 0;
        for (let j = 0; j < 4; j++) d += norm3(meds[i], meds[j]);
        if (d < bestD) { bestD = d; best = i; }
      }
      bg.set(meds[best], (r * 8 + c) * 3);
    }
  }
  return bg;
}

function bgAt(bg, r, c) {
  return bg.subarray((r * 8 + c) * 3, (r * 8 + c) * 3 + 3);
}

export function palette(bg) {
  const a = [[], [], []], b = [[], [], []];
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++)
      for (let ch = 0; ch < 3; ch++) (PARITY(r, c) ? a : b)[ch].push(bg[(r * 8 + c) * 3 + ch]);
  const ma = a.map(v => median(v)), mb = b.map(v => median(v));
  return [ma, mb, norm3(ma, mb)];
}

// ============================================================================ scoring

function gray(F, n) {
  const g = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) g[i] = 0.299 * F[i * 3] + 0.587 * F[i * 3 + 1] + 0.114 * F[i * 3 + 2];
  return g;
}

function gridEdgeRatio(g, n) {
  const s = Math.floor(n / 8);
  const colProf = new Float64Array(n), rowProf = new Float64Array(n);
  for (let y = 0; y < n; y++) {
    const ym = reflect101(y - 1, n) * n, y0 = y * n, yp = reflect101(y + 1, n) * n;
    for (let x = 0; x < n; x++) {
      const xm = reflect101(x - 1, n), xp = reflect101(x + 1, n);
      const gx = (g[ym + xp] + 2 * g[y0 + xp] + g[yp + xp]) - (g[ym + xm] + 2 * g[y0 + xm] + g[yp + xm]);
      const gy = (g[yp + xm] + 2 * g[yp + x] + g[yp + xp]) - (g[ym + xm] + 2 * g[ym + x] + g[ym + xp]);
      colProf[x] += Math.abs(gx);
      rowProf[y] += Math.abs(gy);
    }
  }
  let total = 0;
  for (const prof of [colProf, rowProf]) {
    for (let i = 0; i < n; i++) prof[i] /= n;
    let peaks = 0;
    for (let k = 1; k < 8; k++) peaks += (prof[k * s - 1] + prof[k * s]) * 0.5;
    total += (peaks / 7) / (median(prof) + 1.0);
  }
  return total / 2;
}

/** How much a rectified image looks like an 8x8 checkerboard. */
export function boardMetrics(F, n) {
  const bg = cornerBackground(F, n);
  const [ma, mb, sep] = palette(bg);
  let correct = 0;
  const mins = new Float64Array(64);
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const col = bgAt(bg, r, c), da = norm3(col, ma), db = norm3(col, mb);
      if (PARITY(r, c) ? da < db : db < da) correct++;
      mins[r * 8 + c] = Math.min(da, db);
    }
  correct /= 64;
  const spread = median(mins);
  const checker = correct ** 2 * clip((sep - 10) / 25, 0, 1) * clip(1 - 1.5 * spread / Math.max(sep, 1), 0, 1);
  const ratio = gridEdgeRatio(gray(F, n), n);
  const edge = clip((ratio - 1.3) / 2.5, 0, 1);
  const score = checker * (0.55 + 0.45 * edge);
  const sharpness = correct ** 2 * Math.max(0, 1 - 1.5 * spread / Math.max(sep, 1)) * Math.min(sep, 60) / 60 * ratio;
  return { score, checker, edge, separation: sep, sharpness };
}

export function scoreCorners(pyr, c, size = SCORE_WARP) {
  return boardMetrics(warpBoard(pyr, c, size), size).score;
}

// ============================================================================ candidates

function toGray(img) {
  const n = img.w * img.h, g = new Float32Array(n), d = img.data;
  for (let i = 0; i < n; i++) g[i] = Math.round(0.299 * d[i * 3] + 0.587 * d[i * 3 + 1] + 0.114 * d[i * 3 + 2]);
  return g;
}

// cv2.Canny(img, lo, hi) with a 3x3 Sobel and L1 magnitude, followed by a 3x3 dilation
function cannyDilated(g, w, h, lo, hi) {
  const mag = new Float32Array(w * h), dir = new Uint8Array(w * h);
  const T22 = Math.tan(Math.PI / 8), T67 = Math.tan(3 * Math.PI / 8);
  for (let y = 0; y < h; y++) {
    const ym = reflect101(y - 1, h) * w, y0 = y * w, yp = reflect101(y + 1, h) * w;
    for (let x = 0; x < w; x++) {
      const xm = reflect101(x - 1, w), xp = reflect101(x + 1, w);
      const gx = (g[ym + xp] + 2 * g[y0 + xp] + g[yp + xp]) - (g[ym + xm] + 2 * g[y0 + xm] + g[yp + xm]);
      const gy = (g[yp + xm] + 2 * g[yp + x] + g[yp + xp]) - (g[ym + xm] + 2 * g[ym + x] + g[ym + xp]);
      const ax = Math.abs(gx), ay = Math.abs(gy);
      mag[y0 + x] = ax + ay;
      dir[y0 + x] = ay <= ax * T22 ? 0 : ay > ax * T67 ? 1 : (gx > 0) === (gy > 0) ? 2 : 3;
    }
  }
  const state = new Uint8Array(w * h);    // 0 none, 1 weak, 2 strong
  const stack = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x, m = mag[i];
      if (m <= lo) continue;
      let a, b;
      switch (dir[i]) {
        case 0: a = mag[i - 1]; b = mag[i + 1]; break;
        case 1: a = mag[i - w]; b = mag[i + w]; break;
        case 2: a = mag[i - w - 1]; b = mag[i + w + 1]; break;
        default: a = mag[i - w + 1]; b = mag[i + w - 1];
      }
      if (m > a && m >= b) {
        if (m > hi) { state[i] = 2; stack.push(i); } else state[i] = 1;
      }
    }
  }
  const edge = new Uint8Array(w * h);
  while (stack.length) {
    const i = stack.pop();
    if (edge[i]) continue;
    edge[i] = 1;
    const x = i % w, y = (i / w) | 0;
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const j = yy * w + xx;
        if (state[j] && !edge[j]) stack.push(j);
      }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!edge[y * w + x]) continue;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx >= 0 && yy >= 0 && xx < w && yy < h) out[yy * w + xx] = 1;
        }
    }
  return out;
}

// connected components of pixels where mask === value; returns bboxes + pixel counts
function components(mask, w, h, value, eight) {
  const label = new Int32Array(w * h).fill(-1);
  const comps = [];
  const stack = new Int32Array(w * h);
  for (let start = 0; start < w * h; start++) {
    if (mask[start] !== value || label[start] >= 0) continue;
    const id = comps.length;
    let top = 0, count = 0, x0 = w, y0 = h, x1 = -1, y1 = -1;
    stack[top++] = start;
    label[start] = id;
    while (top) {
      const i = stack[--top], x = i % w, y = (i / w) | 0;
      count++;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          if (!eight && dx && dy) continue;
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const j = yy * w + xx;
          if (mask[j] === value && label[j] < 0) { label[j] = id; stack[top++] = j; }
        }
    }
    comps.push({ x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1, count });
  }
  return comps;
}

function contourCandidates(g, w, h) {
  const out = [];
  const bl = blur(g, w, h, 1, K5);
  for (const [lo, hi] of [[20, 60], [50, 150]]) {
    const edges = cannyDilated(bl, w, h, lo, hi);
    // outlines (edge blobs) and the regions they enclose (holes) stand in for findContours
    for (const [value, eight] of [[1, true], [0, false]]) {
      for (const cc of components(edges, w, h, value, eight)) {
        const area = value === 1 ? cc.w * cc.h : cc.count;
        if (area < 0.01 * h * w) continue;
        if (cc.w >= MIN_BOARD_PX && cc.w / Math.max(cc.h, 1) > 0.8 && cc.w / Math.max(cc.h, 1) < 1.25) {
          out.push(rectCorners(cc.x, cc.y, cc.w, cc.h));
        }
      }
    }
  }
  return out;
}

function polyfit1(xs, ys) {
  const n = xs.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  return [slope, (sy - slope * sx) / n];
}

function latticeFromCells(group, size) {
  const n = group.length, neigh = [];
  const lo = 0.9 * size, hi = 1.6 * size + 8;
  for (let a = 0; a < n; a++)
    for (let b = 0; b < n; b++) {
      const dx = group[b][0] - group[a][0], dy = group[b][1] - group[a][1];
      if (Math.abs(dy) < 0.3 * size && dx > lo && dx < hi) neigh.push(dx);
      if (Math.abs(dx) < 0.3 * size && dy > lo && dy < hi) neigh.push(dy);
    }
  if (!neigh.length) return [];
  const pitch = median(neigh);
  let ref = 0;
  for (let i = 1; i < n; i++) if (Math.abs(group[i][2] - size) < Math.abs(group[ref][2] - size)) ref = i;
  const I = [], J = [], CX = [], CY = [];
  for (const [cx, cy] of group) {
    const i = Math.round((cx - group[ref][0]) / pitch), j = Math.round((cy - group[ref][1]) / pitch);
    if (Math.abs(cx - group[ref][0] - i * pitch) < 0.2 * pitch && Math.abs(cy - group[ref][1] - j * pitch) < 0.2 * pitch) {
      I.push(i); J.push(j); CX.push(cx); CY.push(cy);
    }
  }
  if (I.length < 6) return [];
  const fit = (idx, val) => new Set(idx).size > 1 ? polyfit1(idx, val)
    : [pitch, val.reduce((a, v, k) => a + v - idx[k] * pitch, 0) / val.length];
  const [px, ax] = fit(I, CX), [py, ay] = fit(J, CY);
  const imin = Math.min(...I), imax = Math.max(...I), jmin = Math.min(...J), jmax = Math.max(...J);
  const range = (a, b) => Array.from({ length: Math.max(0, b - a) }, (_, k) => a + k);
  const xs = imax - imin <= 7 ? range(imax - 7, imin + 1) : range(imin, imax - 6);
  const ys = jmax - jmin <= 7 ? range(jmax - 7, jmin + 1) : range(jmin, jmax - 6);
  const out = [];
  for (const ox of xs)
    for (const oy of ys) out.push(rectCorners(ax + px * (ox - 0.5), ay + py * (oy - 0.5), 8 * px, 8 * py));
  return out;
}

function latticeCandidates(g, w, h) {
  const edges = cannyDilated(g, w, h, 30, 90);
  const minSide = Math.max(8, Math.min(h, w) / 90), maxSide = Math.min(h, w) / 7;
  const cells = [];
  for (const cc of components(edges, w, h, 0, false)) {
    if (cc.w < minSide || cc.h < minSide || cc.w > maxSide || cc.h > maxSide) continue;
    if (!(cc.w / cc.h > 0.85 && cc.w / cc.h < 1.18)) continue;
    if (cc.count < 0.8 * cc.w * cc.h) continue;
    cells.push([cc.x + cc.w / 2, cc.y + cc.h / 2, (cc.w + cc.h) / 2]);
  }
  if (cells.length < 6) return [];
  const out = [], used = new Array(cells.length).fill(false);
  for (let round = 0; round < 2; round++) {            // up to two size clusters
    let bestCount = 0, ref = -1;
    cells.forEach((cell, i) => {
      if (used[i]) return;
      const cnt = cells.filter(o => Math.abs(o[2] - cell[2]) < 0.12 * cell[2]).length;
      if (cnt > bestCount) { bestCount = cnt; ref = i; }
    });
    if (bestCount < 6) break;
    const refSize = cells[ref][2], members = [];
    cells.forEach((cell, i) => {
      if (!used[i] && Math.abs(cell[2] - refSize) < 0.12 * refSize) { members.push(cell); used[i] = true; }
    });
    out.push(...latticeFromCells(members, refSize));
  }
  return out;
}

function dedupe(cands) {
  const kept = [];
  for (const cand of cands) {
    const c = cand.c, side = Math.hypot(c[0][0] - c[1][0], c[0][1] - c[1][1]);
    const dup = kept.some(k => {
      let mx = 0;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 2; j++) mx = Math.max(mx, Math.abs(c[i][j] - k.c[i][j]));
      return mx < 0.015 * side;
    });
    if (!dup) kept.push(cand);
  }
  return kept;
}

// ============================================================================ refinement

function isAxisAligned(c, tol = 0.02) {
  const side = Math.hypot(c[0][0] - c[1][0], c[0][1] - c[1][1]);
  return Math.abs(c[0][1] - c[1][1]) < tol * side && Math.abs(c[3][1] - c[2][1]) < tol * side
    && Math.abs(c[0][0] - c[3][0]) < tol * side && Math.abs(c[1][0] - c[2][0]) < tol * side;
}

function coordinateDescent(params, f, step, minStep, maxEvals) {
  let best = f(params), evals = 1;
  while (step >= minStep && evals < maxEvals) {
    let improved = false;
    outer: for (let k = 0; k < params.length; k++) {
      for (const sgn of [1, -1]) {
        const trial = params.slice();
        trial[k] += sgn * step;
        const s = f(trial);
        evals++;
        if (s > best + 1e-5) { best = s; params = trial; improved = true; break outer; }
      }
    }
    if (!improved) step *= 0.5;
  }
  return [params, best];
}

/** Nudge the board edges to maximise the checkerboard score (stage 1 coarse, stage 2 fine). */
export function refineCorners(pyr, corners, { maxEvals = 160, fine = true, free = false } = {}) {
  const img = pyr[0];
  const c = orderCorners(corners);
  const side = Math.hypot(c[0][0] - c[1][0], c[0][1] - c[1][1]);
  let toC, params;
  if (!free && isAxisAligned(c)) {
    const xs = c.map(p => p[0]), ys = c.map(p => p[1]);
    params = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    toC = p => rectCorners(p[0], p[1], p[2] - p[0], p[3] - p[1]);
  } else {
    params = c.flat();
    toC = p => [[p[0], p[1]], [p[2], p[3]], [p[4], p[5]], [p[6], p[7]]];
  }
  const fAt = size => p => {
    const cc = toC(p);
    if (!plausible(cc, img.w, img.h)) return -1;
    return boardMetrics(warpBoard(pyr, cc, size), size).sharpness;
  };
  [params] = coordinateDescent(params, fAt(SCORE_WARP), side / 32, side / 256, maxEvals);
  if (fine) [params] = coordinateDescent(params, fAt(FINE_WARP), side / 256, Math.max(0.5, side / 1024), maxEvals >> 1);
  const finalC = toC(params);
  return [finalC, plausible(finalC, img.w, img.h) ? scoreCorners(pyr, finalC) : 0];
}

// ============================================================================ detection

/** Locate the most board-like region, or null. Returns {corners, score, method}. */
export function detectBoard(pyr, { refine = true } = {}) {
  const img = pyr[0];
  let level = 0;
  while (level < pyr.length - 1 && Math.max(pyr[level].w, pyr[level].h) > 1400) level++;
  const small = pyr[level], scale = 2 ** level;
  const g = toGray(small);

  const raw = [];
  for (const c of latticeCandidates(g, small.w, small.h)) raw.push({ c: c.map(p => [p[0] * scale, p[1] * scale]), m: "lattice" });
  for (const c of contourCandidates(g, small.w, small.h)) raw.push({ c: c.map(p => [p[0] * scale, p[1] * scale]), m: "contour" });
  raw.push({ c: rectCorners(0, 0, img.w, img.h), m: "full-image" });

  const scored = [];
  for (const { c: c0, m } of dedupe(raw)) {
    const c = orderCorners(c0);
    if (!plausible(c, img.w, img.h)) continue;
    scored.push({ s: scoreCorners(pyr, c), c, m });
  }
  if (!scored.length) return null;
  scored.sort((a, b) => b.s - a.s);

  let results = [scored[0]];
  if (refine) {
    const top = scored[0].s;
    // coarse refinement of the best three, fine refinement only for the winner (phone CPU budget)
    const coarse = [];
    for (const { s, c, m } of scored.slice(0, 3)) {
      if (s < 0.15 && s < top * 0.5) continue;
      const [rc, rs] = refineCorners(pyr, c, { fine: false });
      coarse.push(rs >= s ? { s: rs, c: rc, m } : { s, c, m });
    }
    coarse.sort((a, b) => b.s - a.s);
    if (coarse.length) {
      const [rc, rs] = refineCorners(pyr, coarse[0].c, { maxEvals: 40 });
      results = results.concat(coarse, rs >= coarse[0].s ? [{ s: rs, c: rc, m: coarse[0].m }] : []);
    }
  }
  const best = results.reduce((a, b) => (b.s > a.s ? b : a));
  if (best.s <= 0) return null;
  return { corners: orderCorners(best.c), score: best.s, method: best.m };
}

/** Board corners chosen by the user (dragged handles); snap them onto the grid. */
export function detectionFromQuad(pyr, quad) {
  const c = orderCorners(quad);
  const img = pyr[0];
  if (!plausible(c, img.w, img.h)) return { corners: c, score: 0, method: "manual" };
  const base = scoreCorners(pyr, c);
  const [rc, rs] = refineCorners(pyr, c, { maxEvals: 120, free: !isAxisAligned(c) });
  return rs > base ? { corners: rc, score: rs, method: "manual" } : { corners: c, score: base, method: "manual" };
}

// ============================================================================ templates

/**
 * Prepare a template set for matching. `set` = {name, templates: [{symbol, rgb: Float32Array(S*S*3),
 * alpha: Float32Array(S*S)}]}; returns the precomputed blurred premultiplied colour and alpha.
 */
export function prepareTemplates(set) {
  const S = SQUARE_PX;
  return {
    name: set.name,
    items: set.templates.map(t => {
      const pm = new Float32Array(S * S * 3);
      for (let i = 0; i < S * S; i++) for (let c = 0; c < 3; c++) pm[i * 3 + c] = t.alpha[i] * t.rgb[i * 3 + c];
      return { symbol: t.symbol, premul: blur(pm, S, S, 3), alpha: blur(t.alpha, S, S, 1) };
    }),
  };
}

/** Decode the binary template format written by build.py: per template RGB (S*S*3) then alpha (S*S). */
export function decodeTemplateSet(name, symbols, bytes) {
  const S = SQUARE_PX, per = S * S * 4, templates = [];
  for (let t = 0; t < symbols.length; t++) {
    const off = t * per;
    const rgb = new Float32Array(S * S * 3), alpha = new Float32Array(S * S);
    for (let i = 0; i < S * S * 3; i++) rgb[i] = bytes[off + i];
    for (let i = 0; i < S * S; i++) alpha[i] = bytes[off + S * S * 3 + i] / 255;
    templates.push({ symbol: symbols[t], rgb, alpha });
  }
  return { name, templates };
}

export function encodeTemplateSet(set) {
  const S = SQUARE_PX, per = S * S * 4;
  const bytes = new Uint8Array(per * set.templates.length);
  set.templates.forEach((t, k) => {
    const off = k * per;
    for (let i = 0; i < S * S * 3; i++) bytes[off + i] = clip(Math.round(t.rgb[i]), 0, 255);
    for (let i = 0; i < S * S; i++) bytes[off + S * S * 3 + i] = clip(Math.round(t.alpha[i] * 255), 0, 255);
  });
  return { symbols: set.templates.map(t => t.symbol).join(""), bytes };
}

// ============================================================================ piece recognition

const SHIFT = 3, DIST_OK = 16, DIST_MAX = 42, EMPTY_OK = 8, EMPTY_MAX = 22, GAP_LOW = 0.06, GAP_HIGH = 0.34;
const KEEP_PER_SYMBOL = 3;           // templates per piece type searched over all shifts

function cellOf(F, n, r, c) {
  const S = n / 8, out = new Float32Array(S * S * 3);
  for (let y = 0; y < S; y++) out.set(F.subarray(((r * S + y) * n + c * S) * 3, ((r * S + y) * n + c * S + S) * 3), y * S * 3);
  return out;
}

function confidence(d1, d2, empty) {
  const gap = (d2 - d1) / Math.max(d2, 1e-6);
  const confGap = clip((gap - GAP_LOW) / (GAP_HIGH - GAP_LOW), 0, 1);
  const [ok, worst] = empty ? [EMPTY_OK, EMPTY_MAX] : [DIST_OK, DIST_MAX];
  return Math.min(confGap, clip((worst - d1) / (worst - ok), 0, 1));
}

// sum of squared differences of the S-2k template centre against the crop at offset (dx, dy),
// giving up once it exceeds `limit`
function ssd(crop, comp, S, dx, dy, limit) {
  const k = SHIFT, inner = S - 2 * k;
  let sum = 0;
  for (let y = 0; y < inner; y++) {
    let ci = ((y + dy) * S + dx) * 3, ti = ((y + k) * S + k) * 3;
    for (let x = 0; x < inner * 3; x++) {
      const d = crop[ci + x] - comp[ti + x];
      sum += d * d;
    }
    if (sum > limit) return sum;
  }
  return sum;
}

/** Classify the 64 squares of a FINE_WARP board. Returns readings row-major (row 0 = top). */
export function readSquares(F, n, prepared) {
  const S = SQUARE_PX;
  if (n !== 8 * S) throw new Error("readSquares expects an 8*SQUARE_PX board");
  const bg = cornerBackground(F, n);
  const [pa, pb, sep] = palette(bg);
  const m = Math.floor(S * 0.1), k = SHIFT, npx = (S - 2 * k) ** 2 * 3;
  const comp = new Float32Array(S * S * 3);
  const readings = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const b = bgAt(bg, r, c);
      const highlighted = norm3(b, PARITY(r, c) ? pa : pb) > Math.max(14, 0.2 * sep);
      const crop = blur(cellOf(F, n, r, c), S, S, 3);
      let se = 0, fgCount = 0;
      for (let y = k; y < S - k; y++)
        for (let x = k; x < S - k; x++)
          for (let ch = 0; ch < 3; ch++) { const d = crop[(y * S + x) * 3 + ch] - b[ch]; se += d * d; }
      const dEmpty = Math.sqrt(se / npx);
      for (let y = m; y < S - m; y++)
        for (let x = m; x < S - m; x++) {
          const i = (y * S + x) * 3;
          if (Math.max(Math.abs(crop[i] - b[0]), Math.abs(crop[i + 1] - b[1]), Math.abs(crop[i + 2] - b[2])) > 30) fgCount++;
        }
      const fg = fgCount / ((S - 2 * m) ** 2);
      const base = { row: r, col: c, highlighted, background: [b[0], b[1], b[2]] };
      if (fg < 0.008 && dEmpty < 6) {
        readings.push({ ...base, symbol: null, confidence: 1, distance: dEmpty, runnerUp: null });
        continue;
      }
      // pass 1: every template, centred; keep the best few per piece type
      const centred = prepared.items.map((t, idx) => {
        for (let i = 0; i < S * S; i++) {
          const a = 1 - t.alpha[i];
          comp[i * 3] = t.premul[i * 3] + a * b[0];
          comp[i * 3 + 1] = t.premul[i * 3 + 1] + a * b[1];
          comp[i * 3 + 2] = t.premul[i * 3 + 2] + a * b[2];
        }
        return { idx, symbol: t.symbol, d: ssd(crop, comp, S, k, k, Infinity) };
      });
      const bySymbol = {};
      for (const e of centred.sort((a, b2) => a.d - b2.d)) {
        (bySymbol[e.symbol] ||= []).length < KEEP_PER_SYMBOL && bySymbol[e.symbol].push(e);
      }
      // pass 2: full +/-SHIFT search for those (same result as cv2.matchTemplate TM_SQDIFF min)
      const dists = { empty: dEmpty };
      for (const [sym, list] of Object.entries(bySymbol)) {
        let bestSq = Infinity;
        for (const e of list) {
          const t = prepared.items[e.idx];
          for (let i = 0; i < S * S; i++) {
            const a = 1 - t.alpha[i];
            comp[i * 3] = t.premul[i * 3] + a * b[0];
            comp[i * 3 + 1] = t.premul[i * 3 + 1] + a * b[1];
            comp[i * 3 + 2] = t.premul[i * 3 + 2] + a * b[2];
          }
          bestSq = Math.min(bestSq, e.d);
          for (let dy = 0; dy <= 2 * k; dy++)
            for (let dx = 0; dx <= 2 * k; dx++) {
              if (dx === k && dy === k) continue;
              bestSq = Math.min(bestSq, ssd(crop, comp, S, dx, dy, bestSq));
            }
        }
        dists[sym] = Math.sqrt(Math.max(bestSq, 0) / npx);
      }
      const ranked = Object.entries(dists).sort((a, b2) => a[1] - b2[1]);
      const [[best, d1], [second, d2]] = ranked;
      readings.push({
        ...base, symbol: best === "empty" ? null : best, confidence: confidence(d1, d2, best === "empty"),
        distance: d1, runnerUp: second,
      });
    }
  }
  return readings;
}

export function readingsGrid(readings) {
  const grid = Array.from({ length: 8 }, () => new Array(8).fill(null));
  for (const rd of readings) grid[rd.row][rd.col] = rd.symbol;
  return grid;
}

// ============================================================================ orientation + turn

const WEIGHTS = { k: 3.0, p: 1.5, q: 0.6, r: 1.0, b: 0.8, n: 0.8 };

export function inferOrientation(grid) {
  const sums = { w: [0, 0], b: [0, 0] };
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const sym = grid[r][c];
      if (!sym) continue;
      const wgt = WEIGHTS[sym.toLowerCase()], acc = sums[sym === sym.toUpperCase() ? "w" : "b"];
      acc[0] += wgt * r;
      acc[1] += wgt;
    }
  if (!sums.w[1] || !sums.b[1]) return { whiteAtBottom: true, confidence: 0, method: "pieces" };
  const s = sums.w[0] / sums.w[1] - sums.b[0] / sums.b[1];
  return { whiteAtBottom: s >= 0, confidence: Math.tanh(Math.abs(s) / 2), method: "pieces" };
}

export function resolveOrientation(grid, mode = "auto", sticky = null, minConf = 0.35) {
  if (mode === "white") return { whiteAtBottom: true, confidence: 1, method: "manual" };
  if (mode === "black") return { whiteAtBottom: false, confidence: 1, method: "manual" };
  const guess = inferOrientation(grid);
  if (guess.confidence >= minConf) return guess;
  if (sticky !== null && sticky !== undefined) {
    return { whiteAtBottom: sticky, confidence: Math.max(guess.confidence, 0.6), method: "sticky" };
  }
  return guess;
}

export function turnFromHighlights(readings) {
  const lit = readings.filter(rd => rd.highlighted);
  if (!lit.length || lit.length > 4) return { whiteToMove: null, confidence: 0, method: "unknown" };
  const occupied = lit.filter(rd => rd.symbol);
  const nonKings = occupied.filter(rd => rd.symbol.toLowerCase() !== "k");
  const pool = nonKings.length ? nonKings : occupied;
  const colours = new Set(pool.map(rd => rd.symbol === rd.symbol.toUpperCase()));
  if (colours.size !== 1) return { whiteToMove: null, confidence: 0, method: "unknown" };
  const moverIsWhite = [...colours][0];
  return { whiteToMove: !moverIsWhite, confidence: lit.length <= 3 ? 0.85 : 0.6, method: "highlight" };
}

// ============================================================================ calibration

function silhouetteAlpha(cell, bg, S, threshold = 28) {
  const mask = new Uint8Array(S * S);
  for (let i = 0; i < S * S; i++) {
    const d = Math.max(Math.abs(cell[i * 3] - bg[0]), Math.abs(cell[i * 3 + 1] - bg[1]), Math.abs(cell[i * 3 + 2] - bg[2]));
    mask[i] = d > threshold ? 1 : 0;
  }
  const morph = (src, dilate) => {           // 3x3, out-of-image pixels ignored (OpenCV default)
    const out = new Uint8Array(S * S);
    for (let y = 0; y < S; y++)
      for (let x = 0; x < S; x++) {
        let v = dilate ? 0 : 1;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= S || yy >= S) continue;
            v = dilate ? v | src[yy * S + xx] : v & src[yy * S + xx];
          }
        out[y * S + x] = v;
      }
    return out;
  };
  const closed = morph(morph(mask, true), false);
  const outside = new Uint8Array(S * S);    // flood fill (4-connected) from the four corners
  const stack = [];
  for (const [x, y] of [[0, 0], [S - 1, 0], [0, S - 1], [S - 1, S - 1]]) {
    if (!closed[y * S + x]) stack.push(y * S + x);
  }
  while (stack.length) {
    const i = stack.pop();
    if (outside[i] || closed[i]) continue;
    outside[i] = 1;
    const x = i % S, y = (i / S) | 0;
    if (x > 0) stack.push(i - 1);
    if (x < S - 1) stack.push(i + 1);
    if (y > 0) stack.push(i - S);
    if (y < S - 1) stack.push(i + S);
  }
  const filled = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) filled[i] = outside[i] ? 0 : 255;
  const sm = blur(filled, S, S, 1, K3);
  for (let i = 0; i < S * S; i++) sm[i] = Math.round(clip(sm[i], 0, 255)) / 255;
  return sm;
}

function startLayout(whiteAtBottom) {
  const back = "RNBQKBNR", layout = [];
  for (let file = 0; file < 8; file++) {
    const col = whiteAtBottom ? file : 7 - file;
    const [wBack, wPawn, bPawn, bBack] = whiteAtBottom ? [7, 6, 1, 0] : [0, 1, 6, 7];
    layout.push([wBack, col, back[file]], [wPawn, col, "P"], [bPawn, col, "p"], [bBack, col, back[file].toLowerCase()]);
  }
  return layout;
}

function rescaledTemplate(symbol, rgb, alpha, S, k) {
  const inner = Math.round(S * k), off = (S - inner) >> 1;
  const rs = resizeArea(rgb, S, S, 3, inner, inner), ra = resizeArea(alpha, S, S, 1, inner, inner);
  const outRgb = new Float32Array(S * S * 3), outA = new Float32Array(S * S);
  for (let y = 0; y < inner; y++)
    for (let x = 0; x < inner; x++) {
      const di = (y + off) * S + x + off, si = y * inner + x;
      outRgb.set(rs.subarray(si * 3, si * 3 + 3), di * 3);
      outA[di] = ra[si];
    }
  return { symbol, rgb: outRgb, alpha: outA };
}

/**
 * Learn a piece set from a FINE_WARP board in the starting position.
 * Returns {set, whiteAtBottom}; throws if the board doesn't look like the start.
 */
export function calibrateFromStart(F, n, whiteAtBottom = null, name = "calibrated") {
  const S = SQUARE_PX;
  if (n !== 8 * S) throw new Error("calibrateFromStart expects an 8*SQUARE_PX board");
  const bg = cornerBackground(F, n);
  const cells = [];
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) cells.push(cellOf(F, n, r, c));
  const m = Math.floor(S * 0.12), rowOcc = new Array(8).fill(0);
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const cell = cells[r * 8 + c], b = bgAt(bg, r, c);
      let cnt = 0;
      for (let y = m; y < S - m; y++)
        for (let x = m; x < S - m; x++) {
          const i = (y * S + x) * 3;
          if (Math.max(Math.abs(cell[i] - b[0]), Math.abs(cell[i + 1] - b[1]), Math.abs(cell[i + 2] - b[2])) > 30) cnt++;
        }
      rowOcc[r] += cnt / ((S - 2 * m) ** 2) / 8;
    }
  }
  if (!(Math.min(rowOcc[0], rowOcc[1], rowOcc[6], rowOcc[7]) > 0.08 && Math.max(...rowOcc.slice(2, 6)) < 0.05)) {
    throw new Error("The board does not look like the starting position "
      + "(pieces must fill the first two and last two rows, middle empty).");
  }
  const alphas = cells.map((cell, i) => silhouetteAlpha(cell, bgAt(bg, i >> 3, i & 7), S));
  if (whiteAtBottom === null || whiteAtBottom === undefined) {
    const lum = rows => {
      const vals = [];
      for (const r of rows)
        for (let c = 0; c < 8; c++) {
          const cell = cells[r * 8 + c], a = alphas[r * 8 + c];
          let s = 0, cnt = 0;
          for (let i = 0; i < S * S; i++) if (a[i] > 0.5) { s += cell[i * 3] + cell[i * 3 + 1] + cell[i * 3 + 2]; cnt += 3; }
          if (cnt) vals.push(s / cnt);
        }
      return vals.length ? vals.reduce((a, v) => a + v, 0) / vals.length : 0;
    };
    whiteAtBottom = lum([6, 7]) >= lum([0, 1]);
  }
  const templates = [];
  for (const [r, c, sym] of startLayout(whiteAtBottom)) {
    const rgb = cells[r * 8 + c], alpha = alphas[r * 8 + c];
    templates.push({ symbol: sym, rgb, alpha });
    templates.push(rescaledTemplate(sym, rgb, alpha, S, 0.92));
  }
  return { set: { name, templates }, whiteAtBottom };
}
