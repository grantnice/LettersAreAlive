// toyanimals.js — procedural, hand-tuned "toy" farm animals for Letters Are Alive.
// Every animal is built from smooth signed-distance shapes (meshed with surface nets), coloured per
// vertex, given big glossy eyes and a velvety fur sheen, and animated with keyframe clips generated
// here (Idle, Idle_2, Idle_Headlow, Eating, Walk, Gallop, Jump_toIdle, Idle_HitReact_Left/Right).
//
//   buildToyAnimal(key, { height, speed }) -> { model, clips }
//     model  : THREE.Group, facing +z, feet on y = 0 (a clone that shares geometry with a cached template)
//     clips  : { name: THREE.AnimationClip } (shared between clones)
//   TOY_SPECIES : list of keys

import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { clone as skeletonClone } from 'three/addons/utils/SkeletonUtils.js';

// meshoptimizer's simplifier (optional): if it can't load, meshes are just used at full density
let _simp = null;
const simplifierReady = import('meshoptimizer/meshopt_simplifier.module.js')
  .then(async (m) => { await m.MeshoptSimplifier.ready; m.MeshoptSimplifier.useExperimentalFeatures = true; _simp = m.MeshoptSimplifier; })
  .catch(() => { _simp = null; });

// ---------------------------------------------------------------------------------------------
// Noise + small helpers
// ---------------------------------------------------------------------------------------------
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
function hash3(x, y, z) { let h = Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(z, 1274126177); h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; }
function vnoise(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  let xf = x - xi, yf = y - yi, zf = z - zi; xf = xf * xf * (3 - 2 * xf); yf = yf * yf * (3 - 2 * yf); zf = zf * zf * (3 - 2 * zf);
  const l = (a, b, t) => a + (b - a) * t;
  return l(l(l(hash3(xi, yi, zi), hash3(xi + 1, yi, zi), xf), l(hash3(xi, yi + 1, zi), hash3(xi + 1, yi + 1, zi), xf), yf),
    l(l(hash3(xi, yi, zi + 1), hash3(xi + 1, yi, zi + 1), xf), l(hash3(xi, yi + 1, zi + 1), hash3(xi + 1, yi + 1, zi + 1), xf), yf), zf);
}
function fbm(x, y, z) { return vnoise(x, y, z) * 0.6 + vnoise(x * 2.1 + 5, y * 2.1, z * 2.1) * 0.3 + vnoise(x * 4.3, y * 4.3 + 3, z * 4.3) * 0.1; }
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// ---------------------------------------------------------------------------------------------
// Signed-distance primitives. Each has d(x,y,z), an AABB and a colour tag.
// sx/sy/sz squash the shape (approximate distance, fine for meshing).
// ---------------------------------------------------------------------------------------------
function roundCone(a, b, r1, r2, tag, o = {}) {
  const ba = sub(b, a), l2 = ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2], rr = r1 - r2, a2 = l2 - rr * rr, il2 = 1 / l2;
  const s = o.squash || null; // [sx, sy, sz] scale about `a`
  const d = (px, py, pz) => {
    let x = px - a[0], y = py - a[1], z = pz - a[2];
    if (s) { x /= s[0]; y /= s[1]; z /= s[2]; }
    const yy = x * ba[0] + y * ba[1] + z * ba[2], zz = yy - l2;
    const qx = x * l2 - ba[0] * yy, qy = y * l2 - ba[1] * yy, qz = z * l2 - ba[2] * yy;
    const x2 = qx * qx + qy * qy + qz * qz, y2 = yy * yy * l2, z2 = zz * zz * l2;
    const k = Math.sign(rr) * rr * rr * x2;
    let r;
    if (Math.sign(zz) * a2 * z2 > k) r = Math.sqrt(x2 + z2) * il2 - r2;
    else if (Math.sign(yy) * a2 * y2 < k) r = Math.sqrt(x2 + y2) * il2 - r1;
    else r = (Math.sqrt(x2 * a2 * il2) + yy * rr) * il2 - r1;
    return s ? r * Math.min(s[0], s[1], s[2]) : r;
  };
  const m = Math.max(r1, r2) * (s ? Math.max(...s) : 1);
  const lo = [Math.min(a[0], b[0]) - m, Math.min(a[1], b[1]) - m, Math.min(a[2], b[2]) - m];
  const hi = [Math.max(a[0], b[0]) + m, Math.max(a[1], b[1]) + m, Math.max(a[2], b[2]) + m];
  if (s) for (let i = 0; i < 3; i++) { const ext = Math.abs(b[i] - a[i]) * s[i] + m; lo[i] = Math.min(lo[i], a[i] - ext); hi[i] = Math.max(hi[i], a[i] + ext); }
  return { d, lo, hi, tag, k: o.k ?? 0.06, bump: o.bump || null };
}
function sphere(c, r, tag, o = {}) { return roundCone(c, add(c, [0, 1e-4, 0]), r, r, tag, o); }
function ellipsoid(c, r, tag, o = {}) {
  const d = (px, py, pz) => {
    const x = (px - c[0]) / r[0], y = (py - c[1]) / r[1], z = (pz - c[2]) / r[2];
    const k0 = Math.hypot(x, y, z), k1 = Math.hypot(x / r[0], y / r[1], z / r[2]);
    return k1 > 1e-9 ? (k0 * (k0 - 1)) / k1 : -Math.min(...r);
  };
  const m = Math.max(...r);
  return { d, lo: [c[0] - m, c[1] - m, c[2] - m], hi: [c[0] + m, c[1] + m, c[2] + m], tag, k: o.k ?? 0.06, bump: o.bump || null };
}
function smin(a, b, k) { if (k <= 0) return Math.min(a, b); const h = Math.max(k - Math.abs(a - b), 0) / k; return Math.min(a, b) - h * h * k * 0.25; }

// ---------------------------------------------------------------------------------------------
// Surface nets mesher (smooth normals from the SDF gradient, colours blended from nearby shapes)
// ---------------------------------------------------------------------------------------------
function meshPart(prims, colorAt, res = 52) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of prims) for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], p.lo[i]); hi[i] = Math.max(hi[i], p.hi[i]); }
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  // smooth blending (k) and bumps grow the surface past each primitive's own bounds
  const grow = Math.max(...prims.map((p) => p.k * 0.3 + (p.bump ? 0.06 : 0)));
  const cell = (span + grow * 2) / res, pad = cell * 2 + grow;
  for (let i = 0; i < 3; i++) { lo[i] -= pad; hi[i] += pad; }
  const n = [0, 1, 2].map((i) => Math.ceil((hi[i] - lo[i]) / cell) + 1);
  const sdf = (x, y, z) => {
    let d = Infinity;
    for (const p of prims) {
      let e = p.d(x, y, z);
      if (p.bump && e < cell * 6) e -= p.bump(x, y, z);
      d = d === Infinity ? e : smin(d, e, p.k);
    }
    return d;
  };
  const N = n[0] * n[1] * n[2], F = new Float32Array(N);
  const idx = (i, j, k) => i + n[0] * (j + n[1] * k);
  for (let k = 0; k < n[2]; k++) for (let j = 0; j < n[1]; j++) for (let i = 0; i < n[0]; i++) F[idx(i, j, k)] = sdf(lo[0] + i * cell, lo[1] + j * cell, lo[2] + k * cell);
  const vid = new Int32Array(N).fill(-1);
  const pos = [];
  const corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const cv = new Float32Array(8);
  for (let k = 0; k < n[2] - 1; k++) for (let j = 0; j < n[1] - 1; j++) for (let i = 0; i < n[0] - 1; i++) {
    let neg = 0;
    for (let c = 0; c < 8; c++) { const [a, b, d] = corners[c]; cv[c] = F[idx(i + a, j + b, k + d)]; if (cv[c] < 0) neg++; }
    if (neg === 0 || neg === 8) continue;
    let sx = 0, sy = 0, sz = 0, cnt = 0;
    for (const [e0, e1] of edges) {
      const v0 = cv[e0], v1 = cv[e1]; if ((v0 < 0) === (v1 < 0)) continue;
      const t = v0 / (v0 - v1); const c0 = corners[e0], c1 = corners[e1];
      sx += c0[0] + (c1[0] - c0[0]) * t; sy += c0[1] + (c1[1] - c0[1]) * t; sz += c0[2] + (c1[2] - c0[2]) * t; cnt++;
    }
    vid[idx(i, j, k)] = pos.length / 3;
    pos.push(lo[0] + (i + sx / cnt) * cell, lo[1] + (j + sy / cnt) * cell, lo[2] + (k + sz / cnt) * cell);
  }
  const tri = [];
  const quad = (a, b, c, d, flip) => { if (a < 0 || b < 0 || c < 0 || d < 0) return; if (flip) tri.push(a, c, b, a, d, c); else tri.push(a, b, c, a, c, d); };
  for (let k = 1; k < n[2] - 1; k++) for (let j = 1; j < n[1] - 1; j++) for (let i = 1; i < n[0] - 1; i++) {
    const f0 = F[idx(i, j, k)] < 0;
    if (f0 !== (F[idx(i + 1, j, k)] < 0)) quad(vid[idx(i, j - 1, k - 1)], vid[idx(i, j, k - 1)], vid[idx(i, j, k)], vid[idx(i, j - 1, k)], f0);
    if (f0 !== (F[idx(i, j + 1, k)] < 0)) quad(vid[idx(i - 1, j, k - 1)], vid[idx(i - 1, j, k)], vid[idx(i, j, k)], vid[idx(i, j, k - 1)], f0);
    if (f0 !== (F[idx(i, j, k + 1)] < 0)) quad(vid[idx(i - 1, j - 1, k)], vid[idx(i, j - 1, k)], vid[idx(i, j, k)], vid[idx(i - 1, j, k)], f0);
  }
  // normals from the SDF gradient + colours
  const nv = pos.length / 3, nrm = new Float32Array(nv * 3), col = new Float32Array(nv * 3), e = cell * 0.5;
  const c = new THREE.Color();
  for (let v = 0; v < nv; v++) {
    const x = pos[v * 3], y = pos[v * 3 + 1], z = pos[v * 3 + 2];
    const g = norm([sdf(x + e, y, z) - sdf(x - e, y, z), sdf(x, y + e, z) - sdf(x, y - e, z), sdf(x, y, z + e) - sdf(x, y, z - e)]);
    nrm[v * 3] = g[0]; nrm[v * 3 + 1] = g[1]; nrm[v * 3 + 2] = g[2];
    // soft-min colour blend over the shapes near this point
    let wsum = 0, r = 0, gg = 0, b = 0, dmin = Infinity;
    const ds = prims.map((p) => { const dd = p.d(x, y, z); if (dd < dmin) dmin = dd; return dd; });
    for (let q = 0; q < prims.length; q++) {
      const w = Math.exp(-(ds[q] - dmin) / (cell * 1.2)); if (w < 1e-3) continue;
      colorAt(prims[q].tag, x, y, z, c); r += c.r * w; gg += c.g * w; b += c.b * w; wsum += w;
    }
    col[v * 3] = r / wsum; col[v * 3 + 1] = gg / wsum; col[v * 3 + 2] = b / wsum;
  }
  // make sure triangles face outward (agree with the gradient)
  if (tri.length) {
    const t0 = tri[0] * 3, t1 = tri[1] * 3, t2 = tri[2] * 3;
    const u = sub([pos[t1], pos[t1 + 1], pos[t1 + 2]], [pos[t0], pos[t0 + 1], pos[t0 + 2]]);
    const w = sub([pos[t2], pos[t2 + 1], pos[t2 + 2]], [pos[t0], pos[t0 + 1], pos[t0 + 2]]);
    const cr = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    if (cr[0] * nrm[t0] + cr[1] * nrm[t0 + 1] + cr[2] * nrm[t0 + 2] < 0) for (let q = 0; q < tri.length; q += 3) { const t = tri[q + 1]; tri[q + 1] = tri[q + 2]; tri[q + 2] = t; }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setIndex(nv > 65535 ? new THREE.Uint32BufferAttribute(tri, 1) : new THREE.Uint16BufferAttribute(tri, 1));
  geo.computeBoundingSphere();
  return geo;
}

// ---------------------------------------------------------------------------------------------
// Species
// ---------------------------------------------------------------------------------------------
// Dimensions are in "model metres"; the farm rescales each animal to its roster height.
// colors: tag -> hex. Tags used: base, belly, snout, nose, hoof, ear, earIn, mane, tail, tailTip, horn, wool, face, legs, sock
const SPECIES = {
  sheep: { // Gotland sheep: silver-grey curly fleece, black face and legs
    legLen: 0.42, bodyLen: 0.62, bodyR: 0.36, neck: { len: 0.22, ang: 0.85, r: 0.15 }, head: { r: 0.16, snout: 0.14, snoutR: 0.09, drop: 0.05 },
    ears: { type: 'side', len: 0.15, w: 0.06 }, tail: { type: 'stub', len: 0.12, r: 0.08 }, legR: 0.06, eye: { s: 1.05, side: 0.62 },
    wool: { body: true, neck: true, amp: 0.035, freq: 13 }, poll: { r: 0.12 },
    colors: { base: '#9fa3a8', belly: '#8d9196', wool: '#b9bcc0', face: '#2b2a2c', snout: '#3a383b', nose: '#151416', hoof: '#1d1c1e', legs: '#2b2a2c', ear: '#2b2a2c', earIn: '#5d4a4f', tail: '#b0b3b7' },
  },
  cow: { // Fjällko: white with black patches and black ears/muzzle
    legLen: 0.62, bodyLen: 1.05, bodyR: 0.42, neck: { len: 0.44, ang: 0.78, r: 0.19 }, head: { r: 0.2, snout: 0.24, snoutR: 0.15, drop: 0.1 },
    ears: { type: 'side', len: 0.17, w: 0.08 }, tail: { type: 'tuft', len: 0.62, r: 0.03 }, legR: 0.085, eye: { s: 1.15, side: 0.72 }, udder: true,
    colors: { base: '#f4f1ea', belly: '#ebe6dc', snout: '#e8b3ab', nose: '#4a2f31', hoof: '#2d2624', ear: '#1f1d1f', earIn: '#d9a0a0', tail: '#f4f1ea', tailTip: '#1f1d1f', legs: '#f4f1ea' },
    pattern: 'patches', patchColor: '#1f1d1f',
  },
  pig: { // Linderöd: pink with dark spots
    legLen: 0.26, bodyLen: 0.78, bodyR: 0.36, neck: { len: 0.12, ang: 1.3, r: 0.22 }, head: { r: 0.22, snout: 0.16, snoutR: 0.11, drop: 0.0 },
    ears: { type: 'flop', len: 0.17, w: 0.1 }, tail: { type: 'curl', len: 0.16, r: 0.025 }, legR: 0.07, eye: { s: 0.95, side: 0.55 }, pigNose: true,
    colors: { base: '#f2b6b0', belly: '#f6c6c0', snout: '#f2a4a2', nose: '#e88d8f', hoof: '#6b4a45', ear: '#eea29e', earIn: '#e58d8d', tail: '#f2b6b0', legs: '#f2b6b0' },
    pattern: 'spots', patchColor: '#3d3436',
  },
  pony: { // Gotland pony: bay/dun with dark mane and legs
    legLen: 0.72, bodyLen: 0.95, bodyR: 0.34, neck: { len: 0.58, ang: 0.58, r: 0.17 }, head: { r: 0.2, snout: 0.3, snoutR: 0.13, drop: 0.12 },
    ears: { type: 'point', len: 0.15, w: 0.06 }, tail: { type: 'hair', len: 0.62, r: 0.09 }, legR: 0.07, eye: { s: 1.1, side: 0.74 }, mane: true, hooves: true,
    colors: { base: '#8a5a36', belly: '#9c6c45', snout: '#5d4031', nose: '#2c211c', hoof: '#2a2422', ear: '#7b4f30', earIn: '#4b3326', mane: '#2b211d', tail: '#2b211d', legs: '#3a2b24' },
  },
  donkey: { // grey with pale muzzle and belly, long ears
    legLen: 0.6, bodyLen: 0.88, bodyR: 0.33, neck: { len: 0.4, ang: 0.75, r: 0.16 }, head: { r: 0.2, snout: 0.3, snoutR: 0.14, drop: 0.1 },
    ears: { type: 'long', len: 0.34, w: 0.085 }, tail: { type: 'tuft', len: 0.5, r: 0.03 }, legR: 0.07, eye: { s: 1.2, side: 0.7 }, mane: true, maneShort: true, hooves: true,
    colors: { base: '#8f8c8a', belly: '#d9d4cc', snout: '#e3ddd3', nose: '#3a3433', hoof: '#2d2a28', ear: '#7f7b78', earIn: '#d8cfc7', mane: '#3c3735', tail: '#8f8c8a', tailTip: '#3c3735', legs: '#8f8c8a' },
  },
  shiba: { // farm dog: red-tan with white cheeks, chest and curled tail
    legLen: 0.32, bodyLen: 0.58, bodyR: 0.22, neck: { len: 0.2, ang: 0.6, r: 0.13 }, head: { r: 0.17, snout: 0.15, snoutR: 0.075, drop: 0.04 },
    ears: { type: 'point', len: 0.14, w: 0.07 }, tail: { type: 'curlBushy', len: 0.24, r: 0.07 }, legR: 0.05, eye: { s: 1.15, side: 0.5 }, paws: true,
    colors: { base: '#d6853e', belly: '#f6eadb', snout: '#f6eadb', nose: '#1e1a1a', hoof: '#f2e4d2', ear: '#c97834', earIn: '#f3dccb', tail: '#d6853e', tailTip: '#f6eadb', legs: '#d6853e' },
    pattern: 'whiteChest',
  },
  llama: { // cream with brown patches, banana ears
    legLen: 0.8, bodyLen: 0.78, bodyR: 0.3, neck: { len: 0.95, ang: 0.12, r: 0.11 }, head: { r: 0.15, snout: 0.17, snoutR: 0.1, drop: 0.04 },
    ears: { type: 'banana', len: 0.2, w: 0.055 }, tail: { type: 'stub', len: 0.16, r: 0.07 }, legR: 0.065, eye: { s: 1.25, side: 0.6 },
    wool: { body: true, neck: false, amp: 0.02, freq: 9 },
    colors: { base: '#efe3cc', belly: '#e8dbc2', wool: '#efe3cc', snout: '#e6d6bd', nose: '#3a2c26', hoof: '#3a2c26', ear: '#8a5a3c', earIn: '#d9b8a1', tail: '#efe3cc', legs: '#efe3cc' },
    pattern: 'llama', patchColor: '#8a5a3c',
  },
  alpaca: { // fluffy cream with a pom-pom head
    legLen: 0.55, bodyLen: 0.68, bodyR: 0.32, neck: { len: 0.72, ang: 0.15, r: 0.13 }, head: { r: 0.15, snout: 0.12, snoutR: 0.085, drop: 0.03 },
    ears: { type: 'point', len: 0.12, w: 0.05 }, tail: { type: 'stub', len: 0.14, r: 0.08 }, legR: 0.07, eye: { s: 1.3, side: 0.6 },
    wool: { body: true, neck: true, amp: 0.04, freq: 10 }, poll: { r: 0.15 },
    colors: { base: '#f3e6cf', belly: '#efe0c6', wool: '#f6ead6', snout: '#d9c3a3', nose: '#4a3a33', hoof: '#4a3a33', ear: '#e9d8bc', earIn: '#cfb194', tail: '#f3e6cf', legs: '#efe0c6' },
  },
  deer: { // roe deer: red-brown, white rump and throat
    legLen: 0.55, bodyLen: 0.66, bodyR: 0.26, neck: { len: 0.38, ang: 0.45, r: 0.1 }, head: { r: 0.13, snout: 0.16, snoutR: 0.07, drop: 0.05 },
    ears: { type: 'big', len: 0.17, w: 0.08 }, tail: { type: 'stub', len: 0.07, r: 0.05 }, legR: 0.04, eye: { s: 1.35, side: 0.7 }, hooves: true,
    colors: { base: '#b0673a', belly: '#dcb48f', snout: '#3b2b26', nose: '#1b1515', hoof: '#2a201d', ear: '#a5603a', earIn: '#ecd9c6', tail: '#f4ece2', legs: '#a5603a' },
    pattern: 'rump',
  },
  fox: { // red fox: orange, white chest and tail tip, black socks and ear backs
    legLen: 0.28, bodyLen: 0.56, bodyR: 0.17, neck: { len: 0.18, ang: 0.7, r: 0.1 }, head: { r: 0.135, snout: 0.17, snoutR: 0.055, drop: 0.03 },
    ears: { type: 'point', len: 0.15, w: 0.07 }, tail: { type: 'bushy', len: 0.5, r: 0.1 }, legR: 0.035, eye: { s: 1.2, side: 0.5 }, paws: true,
    colors: { base: '#e2762c', belly: '#f8efe4', snout: '#f8efe4', nose: '#161212', hoof: '#2a2120', ear: '#e2762c', earIn: '#2a2120', tail: '#e2762c', tailTip: '#f8efe4', legs: '#e2762c', sock: '#2a2120' },
    pattern: 'whiteChest', socks: true,
  },
  pug: { // fawn pug with black mask and curly tail
    legLen: 0.2, bodyLen: 0.4, bodyR: 0.19, neck: { len: 0.12, ang: 0.7, r: 0.13 }, head: { r: 0.15, snout: 0.05, snoutR: 0.09, drop: 0.04 },
    ears: { type: 'flop', len: 0.1, w: 0.075 }, tail: { type: 'curl', len: 0.11, r: 0.035 }, legR: 0.045, eye: { s: 1.3, side: 0.55 }, paws: true,
    colors: { base: '#d8b98f', belly: '#e6cfab', snout: '#2a2422', nose: '#151212', hoof: '#d8b98f', ear: '#2a2422', earIn: '#2a2422', tail: '#d8b98f', legs: '#d8b98f', face: '#2a2422' },
    pattern: 'mask',
  },
  bull: { // dark brown bull with horns and a little gold nose ring
    legLen: 0.6, bodyLen: 1.1, bodyR: 0.48, neck: { len: 0.36, ang: 0.95, r: 0.26 }, head: { r: 0.23, snout: 0.22, snoutR: 0.16, drop: 0.07 },
    ears: { type: 'side', len: 0.16, w: 0.08 }, tail: { type: 'tuft', len: 0.62, r: 0.035 }, legR: 0.1, eye: { s: 1.05, side: 0.72 }, horns: true, ring: true,
    colors: { base: '#4a2c22', belly: '#5a3a2e', snout: '#8a6a5e', nose: '#2a1c18', hoof: '#1d1716', ear: '#4a2c22', earIn: '#a87a6c', horn: '#efe4cc', tail: '#4a2c22', tailTip: '#1f1614', legs: '#3e251d' },
  },
  husky: { // grey and white with a face mask
    legLen: 0.36, bodyLen: 0.62, bodyR: 0.23, neck: { len: 0.22, ang: 0.6, r: 0.15 }, head: { r: 0.17, snout: 0.16, snoutR: 0.07, drop: 0.03 },
    ears: { type: 'point', len: 0.13, w: 0.07 }, tail: { type: 'curlBushy', len: 0.32, r: 0.08 }, legR: 0.052, eye: { s: 1.15, side: 0.5, iris: '#5aa6e6' }, paws: true,
    colors: { base: '#6d727a', belly: '#f4f4f2', snout: '#f4f4f2', nose: '#161414', hoof: '#f4f4f2', ear: '#5a5f66', earIn: '#f0e2e0', tail: '#6d727a', tailTip: '#f4f4f2', legs: '#e9e9e6', face: '#f4f4f2' },
    pattern: 'huskyMask',
  },
  horse: { // white horse with silver mane
    legLen: 0.82, bodyLen: 1.02, bodyR: 0.36, neck: { len: 0.72, ang: 0.55, r: 0.17 }, head: { r: 0.2, snout: 0.36, snoutR: 0.13, drop: 0.15 },
    ears: { type: 'point', len: 0.16, w: 0.06 }, tail: { type: 'hair', len: 0.72, r: 0.1 }, legR: 0.075, eye: { s: 1.1, side: 0.74 }, mane: true, hooves: true,
    colors: { base: '#f2efea', belly: '#e9e5de', snout: '#d8cfc8', nose: '#8d7d78', hoof: '#6b625d', ear: '#ece8e2', earIn: '#d6b9b4', mane: '#d4d6dc', tail: '#d4d6dc', legs: '#ece8e2' },
  },
  stag: { // red deer stag with antlers
    legLen: 0.82, bodyLen: 0.92, bodyR: 0.32, neck: { len: 0.55, ang: 0.42, r: 0.15 }, head: { r: 0.16, snout: 0.22, snoutR: 0.085, drop: 0.07 },
    ears: { type: 'big', len: 0.18, w: 0.08 }, tail: { type: 'stub', len: 0.08, r: 0.06 }, legR: 0.055, eye: { s: 1.2, side: 0.7 }, antlers: true, hooves: true, ruff: true,
    colors: { base: '#9a5a35', belly: '#c79a74', snout: '#4a3328', nose: '#1b1515', hoof: '#2a201d', ear: '#8c5231', earIn: '#e6d2bf', antler: '#e7d6b6', tail: '#e9dccb', legs: '#7a4a2e', mane: '#6a3f27' },
    pattern: 'rump',
  },
  wolf: { // friendly grey wolf
    legLen: 0.4, bodyLen: 0.68, bodyR: 0.23, neck: { len: 0.24, ang: 0.65, r: 0.16 }, head: { r: 0.165, snout: 0.2, snoutR: 0.07, drop: 0.04 },
    ears: { type: 'point', len: 0.14, w: 0.07 }, tail: { type: 'bushy', len: 0.48, r: 0.1 }, legR: 0.05, eye: { s: 1.1, side: 0.5, iris: '#c9a23a' }, paws: true, ruff: true,
    colors: { base: '#8a8780', belly: '#ddd8cf', snout: '#ddd8cf', nose: '#161414', hoof: '#cfcac0', ear: '#6f6c66', earIn: '#d8cfc8', tail: '#8a8780', tailTip: '#3a3836', legs: '#a8a49b', mane: '#9a968e' },
    pattern: 'whiteChest',
  },
  zebra: { // stripes!
    legLen: 0.7, bodyLen: 0.92, bodyR: 0.34, neck: { len: 0.56, ang: 0.6, r: 0.17 }, head: { r: 0.2, snout: 0.3, snoutR: 0.13, drop: 0.12 },
    ears: { type: 'point', len: 0.17, w: 0.075 }, tail: { type: 'tuft', len: 0.55, r: 0.03 }, legR: 0.068, eye: { s: 1.15, side: 0.74 }, mane: true, maneShort: true, hooves: true,
    colors: { base: '#f6f4ef', belly: '#f6f4ef', snout: '#2a2628', nose: '#151314', hoof: '#1d1b1c', ear: '#f6f4ef', earIn: '#f6f4ef', mane: '#f6f4ef', tail: '#f6f4ef', tailTip: '#1d1b1c', legs: '#f6f4ef' },
    pattern: 'stripes', patchColor: '#1f1c1e',
  },
};
export const TOY_SPECIES = Object.keys(SPECIES);

// ---------------------------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------------------------
const _tmpC = new THREE.Color();
function lin(hex) { return new THREE.Color(hex); } // THREE.Color(hex) is already converted to linear

const tick = () => new Promise((r) => setTimeout(r, 0));
const meshPartAsync = async (prims, colorAt, res, keep = 0.3) => { await tick(); return simplify(meshPart(prims, colorAt, res), keep); };
// Reduce triangles where the surface is flat; colour and normal changes are weighted so patches,
// stripes and creases keep their shape. Unused vertices are dropped.
function simplify(geo, keep) {
  if (!_simp || keep >= 1) return geo;
  const P = geo.attributes.position.array, N = geo.attributes.normal.array, Cc = geo.attributes.color.array, nv = P.length / 3;
  const idx = new Uint32Array(geo.index.array);
  const attr = new Float32Array(nv * 6); for (let i = 0; i < nv; i++) { attr.set([Cc[i * 3], Cc[i * 3 + 1], Cc[i * 3 + 2], N[i * 3], N[i * 3 + 1], N[i * 3 + 2]], i * 6); }
  let out;
  try { [out] = _simp.simplifyWithAttributes(idx, P, 3, attr, 6, [1.2, 1.2, 1.2, 0.35, 0.35, 0.35], null, Math.floor(idx.length * keep / 3) * 3, 0.004); }
  catch (e) { return geo; }
  if (!out || out.length < 12) return geo;
  const remap = new Int32Array(nv).fill(-1); let k = 0;
  for (const i of out) if (remap[i] < 0) remap[i] = k++;
  const np = new Float32Array(k * 3), nn = new Float32Array(k * 3), nc = new Float32Array(k * 3);
  for (let i = 0; i < nv; i++) { const r = remap[i]; if (r < 0) continue; for (let j = 0; j < 3; j++) { np[r * 3 + j] = P[i * 3 + j]; nn[r * 3 + j] = N[i * 3 + j]; nc[r * 3 + j] = Cc[i * 3 + j]; } }
  const ni = k > 65535 ? new Uint32Array(out.length) : new Uint16Array(out.length); for (let i = 0; i < out.length; i++) ni[i] = remap[out[i]];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(np, 3)); g.setAttribute('normal', new THREE.BufferAttribute(nn, 3)); g.setAttribute('color', new THREE.BufferAttribute(nc, 3));
  g.setIndex(new THREE.BufferAttribute(ni, 1)); geo.dispose(); return g;
}
async function buildTemplate(key) {
  await simplifierReady;
  const S = SPECIES[key];
  const C = {}; for (const [k, v] of Object.entries(S.colors)) C[k] = lin(v);
  const col = (tag) => C[tag] || C.base;
  const L = S.legLen, BL = S.bodyLen, R = S.bodyR;
  const yc = L + R * 0.62;                       // torso axis height
  const shoulder = [0, yc + R * 0.35, BL / 2 - R * 0.05];
  const nA = S.neck.ang, ndir = [0, Math.cos(nA), Math.sin(nA)];
  const neckTip = mul(ndir, S.neck.len);
  const hr = S.head.r * 1.12; // toy proportions: slightly big heads

  const parts = {}; // name -> { geo, pivot, parent, restRot }
  const groups = {};
  const root = new THREE.Group(); root.name = 'toy_' + key;
  const rig = new THREE.Group(); rig.name = 'rig'; root.add(rig);

  const material = new THREE.MeshPhysicalMaterial({
    vertexColors: true, roughness: 0.66, metalness: 0, sheen: 0.7, sheenRoughness: 0.55, sheenColor: new THREE.Color('#fff6ea'),
    clearcoat: 0.0,
  });
  const mkNode = (name, parent, pos, rot = [0, 0, 0]) => {
    const g = new THREE.Group(); g.name = name; g.position.set(...pos); g.rotation.set(...rot);
    g.userData.restQ = g.quaternion.clone(); g.userData.restP = g.position.clone();
    parent.add(g); groups[name] = g; return g;
  };
  const addMesh = (parent, geo, name) => {
    const m = new THREE.Mesh(geo, material); m.name = name + '_mesh'; m.castShadow = true; m.receiveShadow = true; m.userData.shared = true; parent.add(m); return m;
  };

  // ---- colour functions per part (x,y,z in the part's local frame)
  const patchNoise = (x, y, z, f) => fbm(x * f + 3.1, y * f + 1.7, z * f + 7.3);
  const pattern = (part, tag, x, y, z, wx, wy, wz, c) => {
    // wx,wy,wz: approximate model-space position for patterns that should flow across parts
    switch (S.pattern) {
      case 'patches': if (tag === 'base' || tag === 'legs' || tag === 'belly') { const n = patchNoise(wx, wy, wz, 2.6); if (n > 0.58) c.lerp(lin(S.patchColor), smooth(0.58, 0.64, n)); } break;
      case 'spots': if (tag === 'base' || tag === 'belly') { const n = patchNoise(wx, wy, wz, 3.4); if (n > 0.62) c.lerp(lin(S.patchColor), smooth(0.62, 0.68, n) * 0.85); } break;
      case 'llama': if (part === 'torso' || part === 'neck' || part === 'head' || part === 'ear') { const n = patchNoise(wx, wy, wz, 2.0); if (n > 0.58 || part === 'head' && wy > yc + 0.9 && n > 0.45) c.lerp(lin(S.patchColor), smooth(0.56, 0.6, n)); } break;
      case 'stripes': {
        if (tag === 'snout' || tag === 'nose' || tag === 'hoof' || tag === 'tailTip' || tag === 'earIn') break;
        let s;
        if (part === 'torso') s = Math.sin(wz * 26 + Math.sin(wy * 8) * 1.6 + (wz < -BL * 0.25 ? (wy - yc) * 14 * smooth(-BL * 0.25, -BL * 0.5, wz) : 0));
        else if (part === 'neck' || part === 'mane') s = Math.sin(y * 30);
        else if (part === 'head') s = Math.sin(y * 34 + z * 8);
        else if (part.startsWith('leg') || part.startsWith('knee')) s = Math.sin(y * 34);
        else if (part === 'ear' || part === 'tail') s = Math.sin(y * 30);
        else s = 1;
        if (part === 'torso' && wy < yc - R * 0.55) s = 1;
        c.lerp(lin(S.patchColor), smooth(0.05, -0.15, s));
        break;
      }
      case 'whiteChest': if (part === 'neck' || part === 'torso') { const front = part === 'neck' ? z - y * 0.3 : wz - BL * 0.2; if (part === 'neck' ? (front > 0.0 && y < S.neck.len * 0.8) : (wy < yc - R * 0.2 && front > -BL * 0.6)) c.lerp(C.belly, 0.9); } if (part === 'head' && y < -hr * 0.15 && tag !== 'nose') c.lerp(C.belly, 0.9); break;
      case 'mask': if (part === 'head' && (tag === 'base') && z > hr * 0.25) c.lerp(C.face, smooth(hr * 0.25, hr * 0.55, z)); break;
      case 'huskyMask': {
        if (part === 'head' && tag === 'base') { const top = y > hr * 0.25 + Math.abs(x) * 0.3 && Math.abs(x) < hr * 0.75 ? 1 : 0; const eyes = (Math.abs(x) > hr * 0.25 && y > hr * 0.05 && z > 0) ? 0 : 1; c.lerp(C.face, (1 - top * 0.9) * (z > -hr * 0.2 ? 1 : 0) * 0.9 * (y < hr * 0.15 || !eyes ? 1 : 0.2)); }
        if (part === 'neck' && z > 0.0 && y < S.neck.len * 0.8) c.lerp(C.belly, 0.9);
        if (part === 'torso' && wy < yc - R * 0.15) c.lerp(C.belly, 0.9);
        break;
      }
      case 'rump': if (part === 'torso' && wz < -BL * 0.38 && wy > yc - R * 0.6) c.lerp(lin('#f3ebe0'), smooth(-BL * 0.38, -BL * 0.5, wz)); if (part === 'neck' && z > 0.02 && y < S.neck.len * 0.5) c.lerp(lin('#efe4d6'), 0.6); break;
      default: break;
    }
    if (S.socks && (part.startsWith('knee'))) c.lerp(C.sock, smooth(L * 0.32, L * 0.18, y + L * 0.5) );
  };
  // belly + subtle top-light gradient, shared by every part
  const shade = (part, tag, x, y, z, wx, wy, wz, out) => {
    out.copy(col(tag));
    if (tag === 'base' && part === 'torso') out.lerp(C.belly, smooth(yc - R * 0.1, yc - R * 0.85, wy));
    pattern(part, tag, x, y, z, wx, wy, wz, out);
    // fur variation
    const n = vnoise(wx * 9, wy * 9, wz * 9); out.multiplyScalar(0.94 + 0.12 * n);
  };

  // ---- torso
  const torso = [];
  torso.push(roundCone([0, yc + 0.0, -BL / 2], [0, yc + R * 0.08, BL / 2], R * 0.86, R * 0.9, 'base', { k: R * 0.25 }));
  torso.push(ellipsoid([0, yc - R * 0.12, 0.02], [R * 0.9, R * 0.88, BL * 0.52], 'base', { k: R * 0.35 }));
  torso.push(ellipsoid([0, yc - R * 0.05, BL / 2 - R * 0.25], [R * 0.92, R * 1.02, R * 0.85], 'base', { k: R * 0.35 }));   // deep chest
  torso.push(ellipsoid([0, yc + R * 0.08, -BL / 2 + R * 0.3], [R * 0.95, R * 0.95, R * 0.85], 'base', { k: R * 0.35 }));  // round rump
  if (S.udder) torso.push(ellipsoid([0, yc - R * 0.95, -BL * 0.28], [R * 0.32, R * 0.2, R * 0.28], 'snout', { k: 0.08 }));
  if (S.wool && S.wool.body) {
    const W = S.wool; const bump = (x, y, z) => W.amp * (fbm(x * W.freq, y * W.freq, z * W.freq) - 0.15);
    for (const p of torso) { p.tag = 'wool'; p.bump = bump; }
  }
  if (S.ruff) torso.push(ellipsoid([0, yc + R * 0.05, BL / 2 - R * 0.05], [R * 0.85, R * 0.95, R * 0.75], S.colors.mane ? 'mane' : 'base', { k: 0.1, bump: (x, y, z) => 0.02 * fbm(x * 14, y * 14, z * 14) }));
  // legs blend into the torso a little (top stubs)
  const hipX = R * 0.52, frontZ = BL / 2 - R * 0.25, backZ = -BL / 2 + R * 0.3, hipY = yc - R * 0.35;
  for (const sx of [-1, 1]) for (const z of [frontZ, backZ]) torso.push(roundCone([sx * hipX * 0.85, yc - R * 0.2, z], [sx * hipX, hipY - 0.01, z], Math.min(S.legR * 1.6, R * 0.38), S.legR * 1.35, 'base', { k: R * 0.2 }));
  const torsoGeo = await meshPartAsync(torso, (tag, x, y, z, c) => shade('torso', tag, x, y, z, x, y, z, c), 42);
  addMesh(rig, torsoGeo, 'torso');

  // ---- legs (upper from hip to knee, lower from knee to the ground)
  const kneeY = L * 0.5;
  const upperLen = hipY - kneeY;
  const lowerPrims = (front) => {
    const r0 = S.legR * (front ? 1.0 : 1.05), r1 = S.legR * 0.78;
    const p = [roundCone([0, 0, 0], [0, -kneeY + S.legR * 0.9, front ? 0.0 : -0.01], r0, r1, 'legs', { k: 0.03 })];
    if (S.hooves) p.push(roundCone([0, -kneeY + S.legR * 1.1, 0.0], [0, -kneeY + S.legR * 0.55, S.legR * 0.15], S.legR * 0.95, S.legR * 1.1, 'hoof', { k: 0.02 }));
    else if (S.paws) p.push(ellipsoid([0, -kneeY + S.legR * 0.7, S.legR * 0.45], [S.legR * 1.25, S.legR * 0.75, S.legR * 1.6], 'hoof', { k: 0.05 }));
    else p.push(roundCone([0, -kneeY + S.legR * 1.0, 0], [0, -kneeY + S.legR * 0.5, S.legR * 0.1], S.legR * 0.9, S.legR * 0.95, 'hoof', { k: 0.02 }));
    return p;
  };
  const upperPrims = (front) => [roundCone([0, 0.02, 0], [0, -upperLen, 0], S.legR * (front ? 1.45 : 1.7), S.legR * 1.02, 'legs', { k: 0.05 })];
  const legShade = (part) => (tag, x, y, z, c) => shade(part, tag, x, y, z, x, y + (part.startsWith('knee') ? kneeY : hipY), z, c);
  const geoUF = await meshPartAsync(upperPrims(true), legShade('legU'), 28), geoUB = await meshPartAsync(upperPrims(false), legShade('legU'), 28);
  const geoLF = await meshPartAsync(lowerPrims(true), legShade('knee'), 34), geoLB = await meshPartAsync(lowerPrims(false), legShade('knee'), 34);
  for (const [nm, sx, z, front] of [['FL', 1, frontZ, true], ['FR', -1, frontZ, true], ['BL', 1, backZ, false], ['BR', -1, backZ, false]]) {
    const leg = mkNode('leg' + nm, rig, [sx * hipX, hipY, z]);
    addMesh(leg, front ? geoUF : geoUB, 'leg' + nm);
    const knee = mkNode('knee' + nm, leg, [0, -upperLen, 0]);
    addMesh(knee, front ? geoLF : geoLB, 'knee' + nm);
  }

  // ---- neck (pivot at the shoulder)
  const neck = mkNode('neck', rig, shoulder);
  const nk = [roundCone([0, -R * 0.25, -R * 0.15], neckTip, S.neck.r * 1.45, S.neck.r, 'base', { k: 0.08 })];
  const back = [0, Math.sin(nA), -Math.cos(nA)];
  if (S.mane) {
    const mr = S.maneShort ? 0.045 : 0.06;
    nk.push(roundCone(add(mul(back, S.neck.r * 0.6), [0, 0.02, 0]), add(neckTip, mul(back, S.neck.r * 0.7)), mr, mr * 0.8, 'mane', { k: 0.04, squash: [S.maneShort ? 0.7 : 0.55, 1, 1] }));
  }
  if (S.wool && S.wool.neck) { const W = S.wool; for (const p of nk) if (p.tag === 'base') { p.tag = 'wool'; p.bump = (x, y, z) => W.amp * (fbm(x * W.freq, y * W.freq, z * W.freq) - 0.15); } }
  if (S.ruff) nk.push(ellipsoid(mul(neckTip, 0.35), [S.neck.r * 1.5, S.neck.r * 1.7, S.neck.r * 1.5], S.colors.mane ? 'mane' : 'base', { k: 0.1, bump: (x, y, z) => 0.02 * fbm(x * 14, y * 14, z * 14) }));
  addMesh(neck, await meshPartAsync(nk, (tag, x, y, z, c) => shade('neck', tag, x, y, z, x + shoulder[0], y + shoulder[1], z + shoulder[2], c), 32), 'neck');

  // ---- head (pivot at the neck tip). Cranium centred just above/behind the pivot, snout towards +z.
  const head = mkNode('head', neck, neckTip, [-(nA - 0.3) * 0.0, 0, 0]);
  const cr = [0, hr * 0.25, hr * 0.15];
  const sTip = [0, cr[1] - hr * 0.35 - S.head.drop, cr[2] + hr * 0.55 + S.head.snout];
  const hd = [];
  hd.push(sphere(cr, hr, 'base', { k: 0.08 }));
  hd.push(roundCone([0, cr[1] - hr * 0.25, cr[2] + hr * 0.35], sTip, S.head.snoutR * 1.2, S.head.snoutR, 'snout', { k: 0.1 }));
  hd.push(ellipsoid([0, cr[1] - hr * 0.35, cr[2] - hr * 0.05], [hr * 0.8, hr * 0.75, hr * 0.85], 'base', { k: 0.1 })); // cheeks/jaw
  if (S.pigNose) hd.push(roundCone(add(sTip, [0, 0, -0.01]), add(sTip, [0, 0, S.head.snoutR * 0.55]), S.head.snoutR * 1.05, S.head.snoutR * 1.0, 'nose', { k: 0.03 }));
  else {
    hd.push(ellipsoid(add(sTip, [0, S.head.snoutR * 0.35, S.head.snoutR * 0.82]), [S.head.snoutR * 0.55, S.head.snoutR * 0.36, S.head.snoutR * 0.3], 'nose', { k: 0.03 }));
  }
  if (S.poll) hd.push(sphere([0, cr[1] + hr * 0.75, cr[2] - hr * 0.05], S.poll.r, 'wool', { k: 0.08, bump: (x, y, z) => 0.03 * (fbm(x * 14, y * 14, z * 14) - 0.2) }));
  if (S.mane) hd.push(ellipsoid([0, cr[1] + hr * 0.9, cr[2] + hr * 0.25], [hr * 0.28, hr * 0.32, hr * 0.42], 'mane', { k: 0.05 })); // forelock
  if (S.horns) for (const sx of [-1, 1]) {
    const b0 = [sx * hr * 0.7, cr[1] + hr * 0.6, cr[2]], b1 = [sx * hr * 1.35, cr[1] + hr * 0.75, cr[2] + hr * 0.1], b2 = [sx * hr * 1.6, cr[1] + hr * 1.25, cr[2] + hr * 0.3];
    hd.push(roundCone(b0, b1, hr * 0.17, hr * 0.12, 'horn', { k: 0.02 }), roundCone(b1, b2, hr * 0.12, hr * 0.04, 'horn', { k: 0.03 }));
  }
  if (S.antlers) for (const sx of [-1, 1]) {
    const hr = S.head.r * 1.7; // antlers are big and proud
    const a0 = [sx * S.head.r * 0.45, cr[1] + S.head.r * 0.7, cr[2] - S.head.r * 0.1];
    const a1 = add(a0, [sx * hr * 0.7, hr * 1.4, -hr * 0.5]), a2 = add(a1, [sx * hr * 0.5, hr * 1.3, hr * 0.2]), a3 = add(a2, [sx * hr * 0.1, hr * 0.9, hr * 0.5]);
    hd.push(roundCone(a0, a1, hr * 0.13, hr * 0.1, 'antler', { k: 0.02 }), roundCone(a1, a2, hr * 0.1, hr * 0.075, 'antler', { k: 0.02 }), roundCone(a2, a3, hr * 0.075, hr * 0.04, 'antler', { k: 0.02 }));
    hd.push(roundCone(lerp3(a0, a1, 0.35), add(lerp3(a0, a1, 0.35), [sx * hr * 0.1, hr * 0.5, hr * 0.75]), hr * 0.07, hr * 0.035, 'antler', { k: 0.02 }));
    hd.push(roundCone(lerp3(a1, a2, 0.5), add(lerp3(a1, a2, 0.5), [sx * hr * 0.35, hr * 0.55, hr * 0.55]), hr * 0.065, hr * 0.03, 'antler', { k: 0.02 }));
    hd.push(roundCone(a2, add(a2, [sx * hr * 0.55, hr * 0.6, -hr * 0.35]), hr * 0.06, hr * 0.03, 'antler', { k: 0.02 }));
  }
  const headW = [shoulder[0] + neckTip[0], shoulder[1] + neckTip[1], shoulder[2] + neckTip[2]];
  const headGeo = await meshPartAsync(hd, (tag, x, y, z, c) => {
    shade('head', tag, x, y, z, x + headW[0], y + headW[1], z + headW[2], c);
    if (tag === 'snout' && S.colors.face && S.pattern !== 'mask' && S.pattern !== 'huskyMask') c.copy(C.face).multiplyScalar(1.15);
    if (key === 'sheep' && tag === 'base') c.copy(C.face);
  }, 48);
  addMesh(head, headGeo, 'head');
  // nose ring for the bull
  if (S.ring) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(S.head.snoutR * 0.45, S.head.snoutR * 0.08, 10, 28), new THREE.MeshPhysicalMaterial({ color: '#e7b53c', metalness: 1, roughness: 0.25 }));
    ring.position.copy(new THREE.Vector3(...add(sTip, [0, -S.head.snoutR * 0.25, S.head.snoutR * 0.75]))); ring.castShadow = true; ring.userData.shared = true; head.add(ring);
  }

  // ---- eyes: white sclera, big dark iris, two catch-lights. Wrapped in a group for blinking.
  // Natural eyes: a smallish dark glossy eyeball set deep into the face, a darker pupil, a rim of fur
  // around it like a socket, a soft upper lid and one small catch-light. Almost no white shows.
  const E = S.eye; const er = hr * 0.22 * E.s;
  const ballGeo = new THREE.SphereGeometry(er, 20, 14), pupilGeo = new THREE.SphereGeometry(er * 0.62, 16, 10), glintGeo = new THREE.SphereGeometry(er * 0.15, 10, 6);
  const rimGeo = new THREE.TorusGeometry(er * 0.98, er * 0.24, 8, 28), lidGeo = new THREE.SphereGeometry(er * 1.05, 20, 8, 0, Math.PI * 2, 0, Math.PI * 0.26);
  const ballM = new THREE.MeshPhysicalMaterial({ color: E.iris || '#3a2414', roughness: 0.15, clearcoat: 1 });
  const pupilM = new THREE.MeshPhysicalMaterial({ color: '#0a0706', roughness: 0.15, clearcoat: 1 });
  const glintM = new THREE.MeshBasicMaterial({ color: '#ffffff' });
  const furAround = (S.colors.face ? C.face : C.base).clone().multiplyScalar(0.92);
  const rimM = material.clone(); rimM.vertexColors = false; rimM.color.copy(furAround);
  for (const sx of [-1, 1]) {
    const side = E.side * 0.85; const dir = norm([sx * side, 0.28, Math.sqrt(Math.max(0.05, 1 - side * side))]);
    // find the real head surface along this direction (the shapes are blended), then sink the eye into it
    let t = hr * 3; const headSdf = (q) => { let d = Infinity; for (const pr of hd) { const e = pr.d(q[0], q[1], q[2]); d = d === Infinity ? e : smin(d, e, pr.k); } return d; };
    for (let it = 0; it < 40; it++) { const dd = headSdf(add(cr, mul(dir, t))); if (Math.abs(dd) < 1e-4) break; t -= dd; }
    const p = add(cr, mul(dir, t - er * 0.55));
    const eg = mkNode(sx > 0 ? 'eyeL' : 'eyeR', head, p);
    // gaze follows the face: outward for prey animals with side-set eyes, forward for dogs and foxes
    eg.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...norm([dir[0] * 0.8, 0.04, dir[2]])));
    eg.userData.restQ = eg.quaternion.clone();
    const add3 = (geo, mat, x, y, z, rx = 0) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.rotation.x = rx; m.userData.shared = true; eg.add(m); return m; };
    add3(ballGeo, ballM, 0, 0, 0);
    add3(pupilGeo, pupilM, 0, 0, er * 0.42);
    add3(glintGeo, glintM, er * 0.32 * sx, er * 0.34, er * 0.9);
    add3(rimGeo, rimM, 0, 0, er * 0.3);
    add3(lidGeo, rimM, 0, 0, 0, -0.25);
  }

  // ---- ears (pivot at the base)
  const EA = S.ears; const ew = EA.w, el = EA.len;
  const earPrims = () => {
    switch (EA.type) {
      case 'flop': return [roundCone([0, 0, 0], [0, -el, el * 0.3], ew, ew * 0.75, 'ear', { k: 0.02, squash: [1, 1, 0.35] })];
      case 'side': return [roundCone([0, 0, 0], [0, 0.0, el], ew * 0.55, ew, 'ear', { k: 0.02, squash: [1, 0.45, 1] })];
      case 'banana': return [roundCone([0, 0, 0], [0, el * 0.6, 0], ew, ew * 0.8, 'ear', { k: 0.02, squash: [1, 1, 0.45] }), roundCone([0, el * 0.6, 0], [-ew * 0.6, el, 0.0], ew * 0.8, ew * 0.25, 'ear', { k: 0.03, squash: [1, 1, 0.45] })];
      default: return [roundCone([0, 0, 0], [0, el, 0], ew, EA.type === 'point' ? ew * 0.12 : ew * 0.45, 'ear', { k: 0.02, squash: [1, 1, EA.type === 'big' ? 0.38 : 0.45] })];
    }
  };
  const earGeo = await meshPartAsync(earPrims(), (tag, x, y, z, c) => {
    shade('ear', tag, x, y, z, x, y + headW[1], z + headW[2], c);
    const front = EA.type === 'side' ? -y : z;
    const along = EA.type === 'side' ? z : EA.type === 'flop' ? -y : y;
    if (front > ew * 0.12 && along > el * 0.08 && along < el * 0.85) c.lerp(C.earIn, 0.85);
    if (key === 'fox' && along > el * 0.55) c.lerp(C.earIn, 0.8);
  }, 28);
  for (const sx of [-1, 1]) {
    let pos, rot;
    switch (EA.type) {
      case 'flop': pos = [sx * hr * 0.62, cr[1] + hr * 0.62, cr[2] - hr * 0.05]; rot = [0.15, sx * 0.25, sx * 0.55]; break;
      case 'side': pos = [sx * hr * 0.78, cr[1] + hr * 0.3, cr[2] - hr * 0.2]; rot = [0.0, sx * (Math.PI / 2 - 0.15), 0.25 * sx]; break;
      case 'long': pos = [sx * hr * 0.4, cr[1] + hr * 0.85, cr[2] - hr * 0.15]; rot = [-0.25, sx * 0.25, -sx * 0.32]; break;
      case 'banana': pos = [sx * hr * 0.45, cr[1] + hr * 0.82, cr[2] - hr * 0.1]; rot = [-0.1, sx * 0.3, -sx * 0.2]; break;
      case 'big': pos = [sx * hr * 0.55, cr[1] + hr * 0.75, cr[2] - hr * 0.2]; rot = [-0.25, sx * 0.4, -sx * 0.7]; break;
      default: pos = [sx * hr * 0.5, cr[1] + hr * 0.8, cr[2] - hr * 0.1]; rot = [-0.12, sx * 0.25, -sx * 0.32];
    }
    const ear = mkNode(sx > 0 ? 'earL' : 'earR', head, pos, rot);
    const m = addMesh(ear, earGeo, 'ear'); if (EA.type === 'side' && sx < 0) { /* symmetric */ }
  }

  // ---- tail (pivot at the rump)
  const T = S.tail; const tailBase = [0, yc + R * 0.45, -BL / 2 - R * 0.55];
  const tail = mkNode('tail', rig, tailBase);
  let tp;
  switch (T.type) {
    case 'tuft': tp = [roundCone([0, 0, 0], [0, -T.len * 0.85, -T.len * 0.12], T.r, T.r * 0.8, 'tail', { k: 0.02 }), ellipsoid([0, -T.len * 0.92, -T.len * 0.12], [T.r * 2.2, T.r * 4, T.r * 2.2], 'tailTip', { k: 0.05, bump: (x, y, z) => 0.008 * fbm(x * 60, y * 60, z * 60) })]; break;
    case 'hair': tp = [roundCone([0, 0, 0], [0, -T.len * 0.25, -T.len * 0.25], T.r * 0.7, T.r, 'tail', { k: 0.05 }), roundCone([0, -T.len * 0.25, -T.len * 0.25], [0, -T.len, -T.len * 0.18], T.r, T.r * 0.55, 'tail', { k: 0.08, squash: [0.75, 1, 1], bump: (x, y, z) => 0.012 * fbm(x * 40, y * 12, z * 40) })]; break;
    case 'bushy': tp = [roundCone([0, 0, 0], [0, -T.len * 0.35, -T.len * 0.45], T.r * 0.5, T.r, 'tail', { k: 0.06 }), roundCone([0, -T.len * 0.35, -T.len * 0.45], [0, -T.len * 0.62, -T.len * 0.85], T.r, T.r * 0.55, 'tail', { k: 0.08 })]; break;
    case 'curl': case 'curlBushy': {
      tp = []; const n = 7, rr = T.len * 0.42;
      let prev = [0, 0, 0];
      for (let i = 1; i <= n; i++) { const a = (i / n) * Math.PI * 1.7; const p = [Math.sin(a * 0.5) * rr * 0.25, Math.sin(a) * rr + rr * 0.25 * (i / n), -Math.cos(a) * rr * 0.9 + rr * 0.9 - rr * 0.6]; tp.push(roundCone(prev, p, T.r * (T.type === 'curlBushy' ? 0.9 + 0.4 * Math.sin((i / n) * Math.PI) : 1), T.r * (T.type === 'curlBushy' ? 0.9 + 0.4 * Math.sin(((i + 1) / n) * Math.PI) : 0.9), i === n ? 'tailTip' : 'tail', { k: 0.04 })); prev = p; }
      break;
    }
    default: tp = [ellipsoid([0, -T.len * 0.25, -T.len * 0.25], [T.r * 0.9, T.len * 0.6, T.r], 'tail', { k: 0.05, bump: S.wool ? (x, y, z) => 0.02 * fbm(x * 20, y * 20, z * 20) : null })];
  }
  const tailGeo = await meshPartAsync(tp, (tag, x, y, z, c) => {
    shade('tail', tag, x, y, z, x + tailBase[0], y + tailBase[1], z + tailBase[2], c);
    if ((T.type === 'bushy') && (-z) > T.len * 0.7) c.lerp(C.tailTip, smooth(T.len * 0.7, T.len * 0.78, -z));
  }, 38);
  addMesh(tail, tailGeo, 'tail');
  tail.rotation.x = T.type === 'tuft' || T.type === 'hair' ? 0.15 : 0; tail.userData.restQ = tail.quaternion.clone();

  // ---- bake every rigid part into a few skinned meshes driven by bones (2-3 draw calls per animal)
  bakeSkinned(root, groups, material);

  // ---- measurements for animation
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  const info = { S, L, yc, shoulderY: shoulder[1], neckTip, nA, hr, headW, box };
  // grazing pose: swing the neck forward-down (never past ~123 deg from vertical, or the head would
  // tuck under the chest), then tip the head down until the muzzle reaches the grass.
  const X = new THREE.Vector3(1, 0, 0);
  const maxNeck = Math.max(0.3, 2.15 - nA);
  const muzzleAt = (na, ha) => { const m = new THREE.Vector3(...sTip).applyAxisAngle(X, ha).add(new THREE.Vector3(...neckTip)); return m.applyAxisAngle(X, na); };
  const groundY = -shoulder[1] + 0.04 * L + S.head.snoutR;
  let eat = maxNeck, eatHead = 0;
  for (let a = 0.1; a <= maxNeck; a += 0.02) { if (muzzleAt(a, 0).y < groundY) { eat = a; break; } }
  if (muzzleAt(eat, 0).y > groundY) for (let h = 0; h < 0.7; h += 0.02) { eatHead = h; if (muzzleAt(eat, h).y < groundY) break; }
  // with the head mostly level the muzzle points down-forward, which reads as nibbling grass
  if (eat > 1.7 - nA + 0.4) eatHead = Math.max(eatHead, 0.25);
  info.eatHead = eatHead;
  info.eat = eat;
  return { root, groups, info };
}

// ---------------------------------------------------------------------------------------------
// Rigid parts -> skinned meshes. Each node Group becomes a Bone with the same name (so the clips
// still bind), each part's vertices are weighted 100% to its bone, and parts are merged by material
// class: fur (sheen, vertex colours), eyes (glossy) and catch-lights (unlit).
// ---------------------------------------------------------------------------------------------
const _eyeMat = () => new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.18, clearcoat: 1, clearcoatRoughness: 0.04, metalness: 0 });
const _glintMat = () => new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false });
function bakeSkinned(root, groups, furMat) {
  root.updateMatrixWorld(true);
  const names = Object.keys(groups);
  const bones = {}, list = [];
  for (const n of names) { const g = groups[n]; const b = new THREE.Bone(); b.name = n; b.position.copy(g.position); b.quaternion.copy(g.quaternion); b.scale.copy(g.scale); b.userData = { ...g.userData }; bones[n] = b; list.push(b); }
  for (const n of names) { const g = groups[n]; const pb = g.parent && bones[g.parent.name] && groups[g.parent.name] === g.parent ? bones[g.parent.name] : null; (pb || root).add(bones[n]); }
  const rigBone = bones.rig || null;
  const buckets = { fur: [], eye: [], glint: [] };
  const meshes = []; root.traverse((o) => { if (o.isMesh) meshes.push(o); });
  for (const m of meshes) {
    let owner = m.parent; while (owner && !groups[owner.name]) owner = owner.parent;
    const bi = owner ? list.indexOf(bones[owner.name]) : list.indexOf(rigBone);
    const geo = m.geometry.clone().applyMatrix4(m.matrixWorld);
    for (const k of Object.keys(geo.attributes)) if (!['position', 'normal', 'color'].includes(k)) geo.deleteAttribute(k);
    const n = geo.attributes.position.count;
    if (!geo.attributes.color) { const c = m.material.color || new THREE.Color(1, 1, 1); const a = new Float32Array(n * 3); for (let i = 0; i < n; i++) a.set([c.r, c.g, c.b], i * 3); geo.setAttribute('color', new THREE.BufferAttribute(a, 3)); }
    if (geo.index && geo.index.array instanceof Uint16Array) geo.setIndex(new THREE.BufferAttribute(new Uint32Array(geo.index.array), 1));
    if (!geo.index) { const ix = new Uint32Array(n); for (let i = 0; i < n; i++) ix[i] = i; geo.setIndex(new THREE.BufferAttribute(ix, 1)); }
    const si = new Uint16Array(n * 4), sw = new Float32Array(n * 4); for (let i = 0; i < n; i++) { si[i * 4] = Math.max(0, bi); sw[i * 4] = 1; }
    geo.setAttribute('skinIndex', new THREE.BufferAttribute(si, 4)); geo.setAttribute('skinWeight', new THREE.BufferAttribute(sw, 4));
    const kind = m.material.isMeshBasicMaterial ? 'glint' : (m.material === furMat || m.material.sheen > 0) ? 'fur' : 'eye';
    buckets[kind].push(geo);
    m.geometry.dispose(); m.removeFromParent();
  }
  for (const n of names) groups[n].removeFromParent();
  root.updateMatrixWorld(true);
  const skeleton = new THREE.Skeleton(list);
  const mats = { fur: furMat, eye: _eyeMat(), glint: _glintMat() };
  for (const [kind, geos] of Object.entries(buckets)) {
    if (!geos.length) continue;
    const g = mergeGeometries(geos); geos.forEach((x) => x.dispose()); g.computeBoundingSphere();
    const sm = new THREE.SkinnedMesh(g, mats[kind]); sm.name = 'toy_' + kind; sm.castShadow = kind !== 'glint'; sm.receiveShadow = kind === 'fur';
    sm.userData.shared = true; root.add(sm); sm.bind(skeleton, sm.matrixWorld);
  }
  for (const n of names) groups[n] = bones[n];
}

// ---------------------------------------------------------------------------------------------
// Animation clips
// ---------------------------------------------------------------------------------------------
const _e = new THREE.Euler(), _q = new THREE.Quaternion();
function makeClip(name, dur, groups, fn, fps = 30) {
  const n = Math.max(2, Math.round(dur * fps) + 1);
  const times = new Float32Array(n);
  const rot = {}, pos = {}, scl = {};
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * dur; times[i] = t;
    const pose = fn(i / (n - 1), t);
    for (const [node, v] of Object.entries(pose)) {
      const g = groups[node]; if (!g) continue;
      if (v.r) { (rot[node] ||= []); _e.set(v.r[0], v.r[1], v.r[2]); _q.copy(g.userData.restQ).multiply(new THREE.Quaternion().setFromEuler(_e)); rot[node].push(_q.x, _q.y, _q.z, _q.w); }
      if (v.p) { (pos[node] ||= []); const rp = g.userData.restP; pos[node].push(rp.x + v.p[0], rp.y + v.p[1], rp.z + v.p[2]); }
      if (v.s) { (scl[node] ||= []); scl[node].push(...v.s); }
    }
  }
  const tracks = [];
  for (const [k, v] of Object.entries(rot)) if (v.length === n * 4) tracks.push(new THREE.QuaternionKeyframeTrack(k + '.quaternion', times, v));
  for (const [k, v] of Object.entries(pos)) if (v.length === n * 3) tracks.push(new THREE.VectorKeyframeTrack(k + '.position', times, v));
  for (const [k, v] of Object.entries(scl)) if (v.length === n * 3) tracks.push(new THREE.VectorKeyframeTrack(k + '.scale', times, v));
  return new THREE.AnimationClip(name, dur, tracks);
}
const TAU = Math.PI * 2;
const blink = (phase, at) => { const d = Math.abs(phase - at); return d < 0.025 ? 0.12 + 0.88 * (d / 0.025) : 1; };

function buildClips(T, worldScale, baseSpeed) {
  const { groups: G, info } = T; const S = info.S;
  const legs = ['FL', 'FR', 'BL', 'BR'];
  const Lw = info.L * worldScale;
  const pose = () => { const p = { rig: { r: [0, 0, 0], p: [0, 0, 0] }, neck: { r: [0, 0, 0] }, head: { r: [0, 0, 0] }, tail: { r: [0, 0, 0] }, earL: { r: [0, 0, 0] }, earR: { r: [0, 0, 0] }, eyeL: { s: [1, 1, 1] }, eyeR: { s: [1, 1, 1] } }; for (const l of legs) { p['leg' + l] = { r: [0, 0, 0] }; p['knee' + l] = { r: [0, 0, 0] }; } return p; };
  const bigTail = S.tail.type === 'bushy' || S.tail.type === 'hair' || S.tail.type === 'tuft';
  const clips = {};
  const idle = (name, dur, look, headLow, extra) => makeClip(name, dur, G, (ph) => {
    const p = pose(); const s = Math.sin(ph * TAU);
    p.rig.p[1] = Math.sin(ph * TAU * 2) * info.L * 0.006;                         // breathing
    p.neck.r[0] = headLow * info.eat * 0.55 + Math.sin(ph * TAU) * 0.04; p.head.r[0] = headLow * info.eatHead * 0.5;
    p.head.r[1] = look * Math.sin(ph * TAU) + (extra ? 0.15 * Math.sin(ph * TAU * 2) : 0);
    p.head.r[2] = look * 0.3 * Math.sin(ph * TAU + 1);
    p.tail.r[1] = Math.sin(ph * TAU * 3) * (bigTail ? 0.25 : 0.12); p.tail.r[2] = Math.sin(ph * TAU * 3 + 1) * 0.08;
    const tw = Math.max(0, Math.sin(ph * TAU * 4 - 1)) ** 8; p.earL.r[2] = tw * 0.3; p.earR.r[2] = -(Math.max(0, Math.sin(ph * TAU * 3 + 2)) ** 8) * 0.3;
    const b = blink(ph, 0.37) * blink(ph, 0.83); p.eyeL.s = [1, b, 1]; p.eyeR.s = [1, b, 1];
    return p;
  });
  clips.Idle = idle('Idle', 4.2, 0.18, 0, false);
  clips.Idle_2 = idle('Idle_2', 5.0, 0.45, 0.05, true);
  clips.Idle_Headlow = idle('Idle_Headlow', 4.6, 0.25, 0.55, false);
  clips.Eating = makeClip('Eating', 3.0, G, (ph) => {
    const p = pose();
    p.neck.r[0] = info.eat + Math.sin(ph * TAU * 3) * 0.03;
    p.head.r[0] = info.eatHead + 0.1 + Math.sin(ph * TAU * 6) * 0.06; p.head.r[1] = Math.sin(ph * TAU) * 0.15;
    p.tail.r[1] = Math.sin(ph * TAU * 2) * (bigTail ? 0.3 : 0.15);
    p.earL.r[2] = Math.sin(ph * TAU) * 0.08; p.earR.r[2] = -Math.sin(ph * TAU) * 0.08;
    p.legFL.r[0] = -0.06; p.legFR.r[0] = 0.04;
    const b = blink(ph, 0.6); p.eyeL.s = [1, b, 1]; p.eyeR.s = [1, b, 1];
    return p;
  });
  // walk: lateral sequence (BL, FL, BR, FR a quarter apart). Period matches the farm's walking speed.
  const walkA = 0.42, walkPeriod = clamp((1.3 * Lw) / Math.max(0.2, baseSpeed), 0.6, 1.6);
  const offs = { BL: 0, FL: 0.25, BR: 0.5, FR: 0.75 };
  clips.Walk = makeClip('Walk', walkPeriod, G, (ph) => {
    const p = pose();
    for (const l of legs) {
      const a = (ph + offs[l]) * TAU; const front = l[0] === 'F';
      p['leg' + l].r[0] = -walkA * Math.sin(a) * (front ? 1 : 0.9);
      p['knee' + l].r[0] = Math.max(0, Math.cos(a)) * (front ? 0.85 : 0.55);
    }
    p.rig.p[1] = Math.abs(Math.sin(ph * TAU * 2)) * info.L * 0.035 - info.L * 0.02;
    p.rig.r[2] = Math.sin(ph * TAU) * 0.025;
    p.neck.r[0] = Math.sin(ph * TAU * 2) * 0.05; p.head.r[0] = -Math.sin(ph * TAU * 2) * 0.04;
    p.tail.r[1] = Math.sin(ph * TAU) * 0.25; p.tail.r[0] = 0.1;
    p.earL.r[0] = Math.sin(ph * TAU * 2) * 0.06; p.earR.r[0] = Math.sin(ph * TAU * 2 + 1) * 0.06;
    return p;
  });
  // gallop: rotary gallop with a rocking spine
  const galPeriod = clamp((2.6 * Lw) / Math.max(0.3, baseSpeed * 3.2), 0.35, 0.9);
  const goff = { BL: 0, BR: 0.1, FL: 0.45, FR: 0.55 };
  clips.Gallop = makeClip('Gallop', galPeriod, G, (ph) => {
    const p = pose();
    for (const l of legs) {
      const a = (ph + goff[l]) * TAU; const front = l[0] === 'F';
      p['leg' + l].r[0] = -0.75 * Math.sin(a);
      p['knee' + l].r[0] = Math.max(0, Math.cos(a)) * (front ? 1.2 : 0.8);
    }
    p.rig.r[0] = Math.sin(ph * TAU + 0.6) * 0.1;
    p.rig.p[1] = Math.max(0, Math.sin(ph * TAU + 1.2)) * info.L * 0.12;
    p.neck.r[0] = Math.sin(ph * TAU + 2.2) * 0.15 + 0.1;
    p.tail.r[0] = -0.5 + Math.sin(ph * TAU) * 0.15;
    p.earL.r[0] = -0.3; p.earR.r[0] = -0.3;
    return p;
  });
  // happy hop: crouch, spring, tuck, land
  clips.Jump_toIdle = makeClip('Jump_toIdle', 1.0, G, (ph) => {
    const p = pose();
    const crouch = smooth(0, 0.18, ph) * (1 - smooth(0.18, 0.3, ph)) + smooth(0.75, 0.85, ph) * (1 - smooth(0.85, 1, ph));
    const air = ph > 0.25 && ph < 0.8 ? Math.sin(((ph - 0.25) / 0.55) * Math.PI) : 0;
    p.rig.p[1] = -crouch * info.L * 0.12 + air * info.L * 0.5;
    p.rig.r[0] = -air * 0.12 + crouch * 0.05;
    for (const l of legs) { const front = l[0] === 'F'; p['leg' + l].r[0] = (front ? -0.5 : 0.5) * air + crouch * (front ? 0.2 : -0.25); p['knee' + l].r[0] = air * (front ? 1.1 : 0.6) + crouch * 0.4; }
    p.neck.r[0] = -air * 0.25 + crouch * 0.15; p.tail.r[0] = -air * 0.6; p.tail.r[1] = Math.sin(ph * TAU * 3) * 0.4;
    p.earL.r[2] = air * 0.5; p.earR.r[2] = -air * 0.5;
    const b = ph > 0.3 && ph < 0.7 ? 0.25 : 1; p.eyeL.s = [1, b, 1]; p.eyeR.s = [1, b, 1]; // happy squint
    return p;
  });
  const react = (name, side) => makeClip(name, 0.8, G, (ph) => {
    const p = pose(); const e = Math.sin(ph * Math.PI);
    p.rig.r[2] = side * e * 0.12; p.head.r[1] = Math.sin(ph * TAU * 2.5) * 0.35 * e; p.head.r[2] = side * e * 0.2;
    p.earL.r[2] = e * 0.5; p.earR.r[2] = -e * 0.5; p.tail.r[1] = Math.sin(ph * TAU * 4) * 0.5 * e;
    p.rig.p[1] = e * info.L * 0.08;
    return p;
  });
  clips.Idle_HitReact_Left = react('Idle_HitReact_Left', 1);
  clips.Idle_HitReact_Right = react('Idle_HitReact_Right', -1);
  return clips;
}

// ---------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------
const _templates = new Map();
export async function buildToyAnimal(key, { height = 1, speed = 1 } = {}) {
  if (!SPECIES[key]) throw new Error('unknown toy animal ' + key);
  const ck = key + ':' + height.toFixed(3) + ':' + speed.toFixed(3);
  if (!_templates.has(key)) _templates.set(key, buildTemplate(key)); // a promise, so parallel callers share one build
  const T = await _templates.get(key);
  if (!_templates.has(ck)) {
    const h = T.info.box.max.y - T.info.box.min.y;
    _templates.set(ck, { T, clips: buildClips(T, height / h, speed) });
  }
  const { clips } = _templates.get(ck);
  const model = skeletonClone(T.root);
  // restQ/restP live in userData (copied by clone) — clips only need names
  return { model, clips, box: T.info.box.clone() };
}
export function purgeToyAnimals() {
  for (const v of _templates.values()) if (v && v.then) v.then((T) => T.root.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); }));
  _templates.clear();
}
