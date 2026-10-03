// world3d.js — the 3D alpine farm for "Letters Are Alive".
// ES module. Needs an importmap that maps "three" and "three/addons/" (see tools/world3d-demo.html).
// Animals are procedural (toyanimals.js). Nature assets (CC0, Quaternius) load from ./assets/ next to this file
// (override with World3D.setAssetBase()); the old animal GLBs there are a fallback.
//
// API (also on window.World3D):
//   supported()                                   -> boolean (WebGL2 available)
//   mountFarm(container, {animals, items, onAnimalTap, arriving}) -> {update(opts), destroy()}
//   renderBackdrop(width, height, variant)        -> Promise<string>  (JPEG data URL; variant "map" | "meadow" | "pond")
//   WORLD_ANIMALS                                 -> [{index, file, displayName}]

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { clone as skeletonClone } from 'three/addons/utils/SkeletonUtils.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { buildToyAnimal, purgeToyAnimals } from './toyanimals.js';

// ---------------------------------------------------------------------------------------------
// Roster
// ---------------------------------------------------------------------------------------------
// h = target height in metres on the farm (toy-scaled: small animals a bit bigger than life).
// zone = where it likes to hang out.
export const WORLD_ANIMALS = [
  { index: 0, toy: 'sheep', file: 'A_Sheep', displayName: 'Gotland sheep', h: 1.0, zone: 'pasture' },
  { index: 1, toy: 'cow', file: 'U_Cow', displayName: 'Fjällko cow', h: 1.5, zone: 'pasture' },
  { index: 2, toy: 'pig', file: 'A_Pig', displayName: 'Linderöd pig', h: 0.9, zone: 'yard' },
  { index: 3, toy: 'pony', file: 'U_Horse', displayName: 'Gotland pony', h: 1.35, zone: 'pasture' },
  { index: 4, toy: 'donkey', file: 'U_Donkey', displayName: 'Donkey', h: 1.25, zone: 'pasture' },
  { index: 5, toy: 'shiba', file: 'U_ShibaInu', displayName: 'Farm dog', h: 0.7, zone: 'yard' },
  { index: 6, toy: 'llama', file: 'A_Llama', displayName: 'Llama', h: 1.75, zone: 'pasture' },
  { index: 7, toy: 'alpaca', file: 'U_Alpaca', displayName: 'Alpaca', h: 1.5, zone: 'pasture' },
  { index: 8, toy: 'deer', file: 'U_Deer', displayName: 'Roe deer', h: 1.15, zone: 'edge' },
  { index: 9, toy: 'fox', file: 'U_Fox', displayName: 'Red fox', h: 0.66, zone: 'edge' },
  { index: 10, toy: 'pug', file: 'A_Pug', displayName: 'Pug', h: 0.56, zone: 'yard' },
  { index: 11, toy: 'bull', file: 'U_Bull', displayName: 'Bull', h: 1.6, zone: 'pasture' },
  { index: 12, toy: 'husky', file: 'U_Husky', displayName: 'Husky', h: 0.8, zone: 'yard' },
  { index: 13, toy: 'horse', file: 'U_Horse_White', displayName: 'White horse', h: 1.7, zone: 'pasture' },
  { index: 14, toy: 'stag', file: 'U_Stag', displayName: 'Stag', h: 1.8, zone: 'edge' },
  { index: 15, toy: 'wolf', file: 'U_Wolf', displayName: 'Wolf', h: 0.85, zone: 'edge' },
  { index: 16, toy: 'zebra', file: 'A_Zebra', displayName: 'Zebra', h: 1.6, zone: 'pasture' },
];

const ANIMAL_SCALE = 1.35; // toy exaggeration so a 4-year-old can see (and tap) them
let ASSET_BASE = new URL('./assets/', import.meta.url).href;
export function setAssetBase(url) { ASSET_BASE = url.endsWith('/') ? url : url + '/'; }

export function supported() {
  try {
    const c = document.createElement('canvas');
    return !!(window.WebGL2RenderingContext && c.getContext('webgl2'));
  } catch (e) { return false; }
}

const REDUCED_MOTION = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------------------------------------------------------------------------------------------
// Small math helpers + noise
// ---------------------------------------------------------------------------------------------
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, t) => a + (b - a) * t;
function smoothstep(a, b, x) { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
function angleLerp(a, b, t) { let d = b - a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return a + d * t; }

const GRAD = [[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]];
function makeNoise(seed) {
  const r = mulberry32(seed);
  const perm = [...Array(256).keys()];
  for (let i = 255; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
  const p = new Uint8Array(512); for (let i = 0; i < 512; i++) p[i] = perm[i & 255];
  const F2 = 0.5 * (Math.sqrt(3) - 1), G2 = (3 - Math.sqrt(3)) / 6;
  return function (x, y) {
    let n0 = 0, n1 = 0, n2 = 0;
    const s = (x + y) * F2; const i = Math.floor(x + s), j = Math.floor(y + s);
    const t = (i + j) * G2; const x0 = x - (i - t), y0 = y - (j - t);
    const i1 = x0 > y0 ? 1 : 0, j1 = x0 > y0 ? 0 : 1;
    const x1 = x0 - i1 + G2, y1 = y0 - j1 + G2, x2 = x0 - 1 + 2 * G2, y2 = y0 - 1 + 2 * G2;
    const ii = i & 255, jj = j & 255;
    let t0 = 0.5 - x0 * x0 - y0 * y0; if (t0 > 0) { const g = GRAD[p[ii + p[jj]] & 7]; t0 *= t0; n0 = t0 * t0 * (g[0] * x0 + g[1] * y0); }
    let t1 = 0.5 - x1 * x1 - y1 * y1; if (t1 > 0) { const g = GRAD[p[ii + i1 + p[jj + j1]] & 7]; t1 *= t1; n1 = t1 * t1 * (g[0] * x1 + g[1] * y1); }
    let t2 = 0.5 - x2 * x2 - y2 * y2; if (t2 > 0) { const g = GRAD[p[ii + 1 + p[jj + 1]] & 7]; t2 *= t2; n2 = t2 * t2 * (g[0] * x2 + g[1] * y2); }
    return 70 * (n0 + n1 + n2);
  };
}
const N1 = makeNoise(11), N2 = makeNoise(23), N3 = makeNoise(37), N4 = makeNoise(51);
function fbm(n, x, y, o = 4) { let a = 0.5, f = 1, s = 0; for (let i = 0; i < o; i++) { s += a * n(x * f, y * f); f *= 2.03; a *= 0.5; } return s; }
function ridged(x, y, o = 6) {
  let s = 0, a = 0.6, f = 1, w = 1;
  for (let i = 0; i < o; i++) {
    let n = 1 - Math.abs(N3(x * f, y * f)); n *= n; n *= w; w = clamp(n * 1.7, 0, 1);
    s += n * a; f *= 2.13; a *= 0.47;
  }
  return s;
}

// ---------------------------------------------------------------------------------------------
// World layout (metres). Camera looks from +z toward -z; the Alps rise in the north (-z).
// ---------------------------------------------------------------------------------------------
const LAYOUT = {
  meadow: { x: 1, z: -1, rx: 15.5, rz: 10.5 },           // where animals may roam
  barn: { x: -12.5, z: -8.5, rot: 0.55, w: 7.2, d: 10 },   // rot = yaw; door faces local +z
  pond: { x: 9.5, z: 3.2, r: 3.1 },
  cottage: { x: 46, z: -46, rot: -0.35 },
  pathPts: [[-9.2, -3.6], [-6, 0.5], [-1, 5], [5, 9.2], [14, 10.8], [24, 12.8], [42, 12]],
  entry: { x: 30, z: 12.8 },
};
const SUN_DIR = new THREE.Vector3(0.68, 0.58, 0.3).normalize();
const SKY = { zenith: '#2f74cf', mid: '#8fc0ee', horizon: '#cfe0f2', fog: '#b4cdea' };

function distToPath(x, z) {
  let best = 1e9; const P = LAYOUT.pathPts;
  for (let i = 0; i < P.length - 1; i++) {
    const [ax, az] = P[i], [bx, bz] = P[i + 1];
    const dx = bx - ax, dz = bz - az; const t = clamp(((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz), 0, 1);
    const d = Math.hypot(x - (ax + dx * t), z - (az + dz * t)); if (d < best) best = d;
  }
  return best;
}
function farmRR(x, z) { return Math.hypot((x - 1) / 1.3, (z + 2) / 1.0); }
function mountainMask(x, z) {
  const d = -z + Math.max(0, Math.abs(x) - 70) * 0.65 + 18 * N1(x * 0.004, 5.1);
  return smoothstep(80, 185, d);
}
export function heightAt(x, z) {
  const rr = farmRR(x, z);
  let h = fbm(N1, x * 0.03, z * 0.03, 3) * 1.3 * smoothstep(8, 24, rr) + fbm(N2, x * 0.09, z * 0.09, 2) * 0.22;
  const hill = smoothstep(19, 90, rr) * (z < 0 ? lerp(1, 0.55, smoothstep(0, -60, z)) : 1);
  h += hill * hill * (4 + 8 * (fbm(N2, x * 0.011 + 3, z * 0.011 - 2, 4) + 0.5));
  const P = LAYOUT.pond; const pd = Math.hypot(x - P.x, z - P.z);
  h -= 0.75 * (1 - smoothstep(P.r * 0.55, P.r + 0.9, pd));
  const m = mountainMask(x, z);
  if (m > 0) {
    const wx = x + 40 * N4(x * 0.003, z * 0.003), wz = z + 40 * N4(x * 0.003 + 9, z * 0.003);
    const r = ridged(wx * 0.0031, wz * 0.0031, 7);
    const massif = 0.6 + 0.4 * smoothstep(-0.4, 0.5, N2(x * 0.0017 + 4, z * 0.0017));
    const far = smoothstep(250, 520, -z);
    h += m * (8 + Math.pow(r, 1.45) * (135 + 70 * far) * massif + (fbm(N1, x * 0.0022, z * 0.0022, 2) + 0.3) * 24);
  }
  return h;
}
function snowLine(x, z) { return 66 + 12 * N2(x * 0.01, z * 0.01); }

// ---------------------------------------------------------------------------------------------
// Asset loading (cached across mounts; call World3D.purgeCache() to free)
// ---------------------------------------------------------------------------------------------
let _loader = null;
const _cache = new Map();
function loader() {
  if (!_loader) { _loader = new GLTFLoader(); _loader.setMeshoptDecoder(MeshoptDecoder); }
  return _loader;
}
function loadGLTF(rel) {
  const url = ASSET_BASE + rel;
  if (!_cache.has(url)) {
    const p = loader().loadAsync(url).then((g) => {
      g.scene.traverse((o) => { o.userData.shared = true; });
      return g;
    });
    p.catch(() => _cache.delete(url));
    _cache.set(url, p);
  }
  return _cache.get(url);
}
export function purgeCache() {
  for (const p of _cache.values()) p.then((g) => g.scene.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) [].concat(o.material).forEach((m) => { for (const k in m) if (m[k] && m[k].isTexture) m[k].dispose(); m.dispose(); });
  })).catch(() => {});
  _cache.clear();
  for (const g of _terrainCache.values()) g.dispose();
  _terrainCache.clear(); _nature = null;
  purgeToyAnimals();
}

let _nature = null;  // {name: [{geometry, material, matrix}]}  with bottom at y=0
async function natureParts() {
  if (_nature) return _nature;
  const g = await loadGLTF('nature/nature.glb');
  const out = {};
  g.scene.updateMatrixWorld(true);
  for (const root of g.scene.children) {
    const box = new THREE.Box3().setFromObject(root);
    const inv = new THREE.Matrix4().makeTranslation(-(box.min.x + box.max.x) / 2, -box.min.y, -(box.min.z + box.max.z) / 2);
    const parts = [];
    root.traverse((o) => {
      if (!o.isMesh) return;
      const m = o.material;
      m.flatShading = false;
      m.vertexColors = false; // COLOR_0 in these models is a wind/AO mask, not a tint (it turned the bushes dark red)
      if (m.map) { m.map.anisotropy = 4; }
      m.side = THREE.DoubleSide;
      parts.push({ geometry: o.geometry, material: m, matrix: inv.clone().multiply(o.matrixWorld), height: box.max.y - box.min.y });
    });
    out[root.name] = parts;
  }
  _nature = out;
  return out;
}

// ---------------------------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------------------------
function toyMat(color, o = {}) {
  return new THREE.MeshPhysicalMaterial({
    color, roughness: o.roughness ?? 0.62, metalness: 0, flatShading: o.flat ?? false,
    clearcoat: o.clearcoat ?? 0.5, clearcoatRoughness: 0.4, vertexColors: !!o.vc,
    side: o.side ?? THREE.FrontSide, emissive: o.emissive ?? 0x000000, emissiveIntensity: o.ei ?? 1,
  });
}
function vcMat(o = {}) {
  return new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: o.flat ?? false, roughness: o.roughness ?? 0.9, metalness: 0, side: o.side ?? THREE.FrontSide });
}

// Colour a geometry with one sRGB colour (as vertex colours) and make it non-indexed
function paint(geo, hex, jitter = 0, rnd = Math.random) {
  const g = geo.index ? geo.toNonIndexed() : geo;
  const n = g.attributes.position.count; const c = new Float32Array(n * 3); const col = new THREE.Color(hex);
  for (let i = 0; i < n; i += 6) { // one shade per quad (two triangles), so box faces don't show a diagonal
    const k = 1 + (rnd() - 0.5) * jitter;
    for (let j = 0; j < Math.min(6, n - i); j++) { c[(i + j) * 3] = col.r * k; c[(i + j) * 3 + 1] = col.g * k; c[(i + j) * 3 + 2] = col.b * k; }
  }
  g.setAttribute('color', new THREE.BufferAttribute(c, 3));
  if (g.attributes.uv) g.deleteAttribute('uv');
  return g;
}
function place(geo, x, y, z, rx = 0, ry = 0, rz = 0, sx = 1, sy = sx, sz = sx) {
  const m = new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)), new THREE.Vector3(sx, sy, sz));
  geo.applyMatrix4(m); return geo;
}
function merge(list) { const g = mergeGeometries(list.map((x) => { if (x.attributes.uv) x.deleteAttribute('uv'); return x; })); list.forEach((x) => x.dispose()); return g; }

// ---------------------------------------------------------------------------------------------
// Terrain
// ---------------------------------------------------------------------------------------------
const C = (h) => new THREE.Color(h);
const COL = {
  meadow: [C('#7cc443'), C('#8dcc47'), C('#6dba3e'), C('#94d04c')],
  hill: [C('#5f9f3a'), C('#6aa83f'), C('#548f36')],
  alp: [C('#4f8a45'), C('#5c9450'), C('#447a40')],
  forestFloor: C('#3b6230'),
  rock: [C('#3a4d8a'), C('#31437a'), C('#46599a'), C('#3b4782')],
  snow: [C('#f6f9ff'), C('#eef3ff'), C('#ffffff')],
  path: [C('#cfb07c'), C('#c4a270'), C('#d6ba88')],
  mud: C('#7c7650'),
};

const _terrainCache = new Map(); // deterministic, so share geometry between farm + backdrops
function terrainGeometry(key, params) {
  if (!_terrainCache.has(key)) _terrainCache.set(key, buildTerrainMesh(params));
  return _terrainCache.get(key);
}
function buildTerrainMesh({ x0, x1, z0, z1, nx, nz, warp = null, far = false }) {
  // Smooth indexed grid. Vertex colours carry the soft ground palette (grass, path, mud);
  // rock and snow are painted per pixel in terrainMaterial from height, slope and noise.
  const gx = (i) => { const u = i / nx; return warp ? warp.x(u) : lerp(x0, x1, u); };
  const gz = (j) => { const v = j / nz; return warp ? warp.z(v) : lerp(z0, z1, v); };
  const nv = (nx + 1) * (nz + 1);
  const pos = new Float32Array(nv * 3), cols = new Float32Array(nv * 3), meadowA = new Float32Array(nv), mtnA = new Float32Array(nv);
  for (let j = 0, k = 0; j <= nz; j++) for (let i = 0; i <= nx; i++, k++) {
    const x = gx(i), z = gz(j);
    let h = heightAt(x, z);
    if (far) { // sink under the near mesh
      const inside = Math.min(smoothstep(-150, -120, x) * smoothstep(150, 120, x), smoothstep(-100, -80, z) * smoothstep(100, 80, z));
      h -= inside * 3;
    } else if (!warp) { // near mesh: tuck the rim under the far mesh so the two never zig-zag through each other
      const rim = 1 - Math.min(smoothstep(x0, x0 + 18, x) * smoothstep(x1, x1 - 18, x), smoothstep(z0, z0 + 18, z) * smoothstep(z1, z1 - 18, z));
      h -= rim * 3;
    }
    pos[k * 3] = x; pos[k * 3 + 1] = h; pos[k * 3 + 2] = z;
    const md = groundColor(x, h, z, _tc);
    cols[k * 3] = _tc.r; cols[k * 3 + 1] = _tc.g; cols[k * 3 + 2] = _tc.b;
    meadowA[k] = md; mtnA[k] = mountainMask(x, z);
  }
  const flip = gz(1) < gz(0);
  const idx = [];
  const tri = (p, q, r) => { if (flip) idx.push(p, r, q); else idx.push(p, q, r); };
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const i00 = j * (nx + 1) + i, i10 = i00 + 1, i01 = i00 + nx + 1, i11 = i01 + 1;
    if ((i + j) % 2 === 0) { tri(i00, i01, i10); tri(i10, i01, i11); }
    else { tri(i00, i01, i11); tri(i00, i11, i10); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  g.setAttribute('meadow', new THREE.BufferAttribute(meadowA, 1));
  g.setAttribute('mtn', new THREE.BufferAttribute(mtnA, 1));
  g.setIndex(nv > 65535 ? new THREE.Uint32BufferAttribute(idx, 1) : new THREE.Uint16BufferAttribute(idx, 1));
  g.computeVertexNormals();
  return g;
}
const _tc = new THREE.Color(), _tc2 = new THREE.Color();
function pick(arr, r) { return arr[Math.floor(r * arr.length) % arr.length]; }
// Soft, noise-driven ground colour (no per-triangle randomness). Returns the meadow-flower weight.
function groundColor(x, y, z, out) {
  const m = mountainMask(x, z);
  const n = N4(x * 0.045, z * 0.045), n2 = N3(x * 0.012, z * 0.012);
  // two greens blended by noise, plus sunny yellow-green patches
  out.copy(COL.meadow[2]).lerp(COL.meadow[3], 0.5 + 0.5 * n);
  out.lerp(_tc2.set('#a9cf55'), smoothstep(0.15, 0.6, n2) * 0.35);
  const rr = farmRR(x, z);
  out.lerp(_tc2.copy(COL.hill[1]), smoothstep(14, 40, rr) * 0.55);
  out.lerp(_tc2.copy(COL.alp[0]), smoothstep(0.1, 0.6, m) * 0.7);
  out.lerp(COL.forestFloor, smoothstep(0.2, 0.7, N2(x * 0.018, z * 0.018)) * smoothstep(30, 60, rr) * 0.35);
  // pond bed + path verge
  const P = LAYOUT.pond; const pd = Math.hypot(x - P.x, z - P.z);
  out.lerp(COL.mud, 1 - smoothstep(P.r, P.r + 0.8, pd));
  const dp = distToPath(x, z);
  out.lerp(COL.path[0], (1 - smoothstep(0.8, 1.8, dp)) * 0.3);
  let md = 0;
  if (rr < 24) md = 1 - smoothstep(20, 24, rr);
  else md = smoothstep(60, 30, rr) * 0.9 * (y < 22 ? 1 : 0);
  md *= smoothstep(1.0, 1.6, dp) * smoothstep(P.r + 0.3, P.r + 1.0, pd) * (1 - smoothstep(0.1, 0.25, m));
  return md;
}

// GLSL value noise shared by the custom shaders
const GLSL_NOISE = `
float wHash3(vec3 p){ p = fract(p*0.3183099 + 0.1); p *= 17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
float wNoise3(vec3 x){ vec3 i = floor(x); vec3 f = fract(x); f = f*f*(3.0-2.0*f);
  return mix(mix(mix(wHash3(i), wHash3(i+vec3(1,0,0)), f.x), mix(wHash3(i+vec3(0,1,0)), wHash3(i+vec3(1,1,0)), f.x), f.y),
             mix(mix(wHash3(i+vec3(0,0,1)), wHash3(i+vec3(1,0,1)), f.x), mix(wHash3(i+vec3(0,1,1)), wHash3(i+vec3(1,1,1)), f.x), f.y), f.z); }
float wFbm3(vec3 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++){ s += a*wNoise3(p); p = p*2.03 + 17.1; a *= 0.5; } return s; }
vec3 wLin(vec3 c){ return pow(c, vec3(2.2)); }
`;

function terrainMaterial(uniforms) {
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: false, roughness: 0.92, metalness: 0 });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = uniforms.uTime;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float meadow;\nattribute float mtn;\nvarying float vMeadow;\nvarying float vMtn;\nvarying vec3 vWPos;\nvarying vec3 vWN;')
      .replace('#include <fog_vertex>', '#include <fog_vertex>\nvMeadow = meadow; vMtn = mtn;\nvWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;\nvWN = normalize(mat3(modelMatrix) * objectNormal);');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
varying float vMeadow; varying float vMtn; varying vec3 vWPos; varying vec3 vWN;
${GLSL_NOISE}
float h21(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
vec3 flowerLayer(vec2 wp, float freq, float rad, float density, vec3 base){
  vec2 p = wp*freq; vec2 cell = floor(p); vec2 f = fract(p);
  float h = h21(cell); vec2 off = vec2(h21(cell+13.1), h21(cell+7.7))*0.6+0.2;
  float fw = length(fwidth(p));
  float d = length(f-off);
  float present = step(1.0-density, h21(cell+3.3));
  vec3 col = h < 0.78 ? vec3(1.0,0.78,0.02) : (h < 0.9 ? vec3(0.95,0.95,0.92) : vec3(0.55,0.33,0.85));
  float dotm = (1.0 - smoothstep(rad-fw*0.7, rad+fw*0.7, d))*present;
  float cov = 3.1416*rad*rad*density;
  float far = smoothstep(0.25, 0.9, fw);
  vec3 avg = vec3(1.0,0.82,0.1);
  return mix(mix(base, col, dotm), mix(base, avg, cov*0.6), far);
}`)
      .replace('#include <color_fragment>', `#include <color_fragment>
float tFw = length(fwidth(vWPos));                                     // ~metres per pixel
float tLod = clamp(tFw * 0.4, 0.0, 1.0);                               // fades fine detail with distance
float tEdge = 1.0 - smoothstep(4.0, 10.0, tFw / max(1.0, -vViewPosition.z * 0.004)); // 0 on silhouettes (derivatives explode there)
float tN1 = wFbm3(vWPos * 0.012);
float tM = smoothstep(0.08, 0.3, vMtn);
float tSnow = 0.0, tRock = 0.0;
diffuseColor.rgb *= 0.9 + 0.2 * tN1;                                   // grass: large soft mottling
{
  vec3 c = flowerLayer(vWPos.xz, 1.6, 0.13, 0.55, diffuseColor.rgb);
  c = flowerLayer(vWPos.xz + 17.3, 3.1, 0.12, 0.35, c);
  diffuseColor.rgb = mix(diffuseColor.rgb, c, max(vMeadow, 0.0) * tEdge);
}`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
{
  // Alpine rock (no branch: derivatives must be computed in uniform control flow): vertical gullies (noise stretched along y) + broad buttresses, as a bump on the smooth mesh.
  vec3 gp = vWPos * vec3(0.045, 0.011, 0.045);
  float gully = 1.0 - abs(wNoise3(gp) * 2.0 - 1.0);
  float hB = (gully * gully * 6.0 + wFbm3(vWPos * 0.03) * 7.0 + wNoise3(vWPos * 0.25) * 0.6 * (1.0 - tLod)) * max(tM, smoothstep(50.0, 60.0, vWPos.y));
  vec3 sX = dFdx(-vViewPosition), sY = dFdy(-vViewPosition);
  vec3 R1 = cross(sY, normal), R2 = cross(normal, sX);
  float fDet = dot(sX, R1);
  vec2 dH = vec2(dFdx(hB), dFdy(hB));
  dH *= 1.0 / max(1.0, length(dH) / (0.6 * length(vec2(length(sX), length(sY)))));  // no spikes at silhouettes
  normal = normalize(mix(normal, normalize(abs(fDet) * normal - sign(fDet) * (dH.x * R1 + dH.y * R2)), tEdge));
  vec3 wN = normalize((vec4(normal, 0.0) * viewMatrix).xyz);
  float up = mix(wN.y, vWN.y, 0.35);
  // rock where steep or above the tree line, crisp but not aliased
  float tN3 = wNoise3(vWPos * 0.06);
  float rk = max(smoothstep(0.86, 0.66, up + (tN3 - 0.5) * 0.15) * smoothstep(22.0, 40.0, vWPos.y + (tN1 - 0.5) * 16.0), smoothstep(34.0, 52.0, vWPos.y + (tN1 - 0.5) * 24.0 + (tN3 - 0.5) * 8.0)) * tM;
  tRock = max(rk, smoothstep(52.0, 60.0, vWPos.y));
  // snow settles on ledges and gentle slopes above a ragged snow line
  float sl = 62.0 + (tN1 - 0.5) * 28.0;
  float ledge = smoothstep(0.38, 0.58, mix(up, vWN.y, 0.5));
  tSnow = smoothstep(sl - 4.0, sl + 4.0, vWPos.y) * ledge;
  tSnow = max(tSnow, smoothstep(sl + 14.0, sl + 30.0, vWPos.y) * smoothstep(0.15, 0.32, mix(up, vWN.y, 0.5)));
    tSnow *= tM;
  vec3 rockC = mix(wLin(vec3(0.46, 0.47, 0.52)), wLin(vec3(0.58, 0.56, 0.53)), smoothstep(0.35, 0.65, wFbm3(vWPos * vec3(0.02, 0.08, 0.02))));
  rockC *= mix(0.62, 1.06, smoothstep(0.1, 0.9, gully));                // dark gullies, bright ribs
  rockC = mix(rockC, diffuseColor.rgb * 0.8, (1.0 - smoothstep(0.55, 0.85, rk)) * 0.5);   // grassy scree at the rock's edge
  rockC = mix(rockC, wLin(vec3(0.33, 0.43, 0.27)), smoothstep(0.80, 0.9, up) * (1.0 - smoothstep(30.0, 52.0, vWPos.y)) * 0.6); // moss on lower ledges
  diffuseColor.rgb = mix(diffuseColor.rgb, rockC, tRock);
  diffuseColor.rgb = mix(diffuseColor.rgb, wLin(vec3(0.95, 0.97, 1.0)), tSnow);
  roughnessFactor = mix(roughnessFactor, 0.6, tSnow);
}`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
totalEmissiveRadiance += diffuseColor.rgb * vec3(0.08, 0.09, 0.12) * tSnow;`);
  };
  mat.customProgramCacheKey = () => 'terrain3';
  return mat;
}

// ---------------------------------------------------------------------------------------------
// Procedural props
// ---------------------------------------------------------------------------------------------
// One soft spruce tier: apex, a shoulder ring and a scalloped, drooping skirt, with a shallow
// underside. Smooth normals are bent outward from the trunk so the foliage shades like a soft mass.
function spruceTier(r, h, y, seg, rnd, top, bottom) {
  const phase = rnd() * 6.28, lobes = 5 + Math.floor(rnd() * 3);
  const P = [0, y + h, 0], col = [top.r, top.g, top.b];
  const ringR = [0.48, 1], ringY = [0.5, 0];
  for (let k = 0; k < 2; k++) for (let i = 0; i < seg; i++) {
    const a = (i / seg) * Math.PI * 2;
    const sc = k === 1 ? 1 + 0.13 * Math.sin(a * lobes + phase) : 1 + 0.05 * Math.sin(a * 3 + phase);
    const rr = r * ringR[k] * sc, droop = k === 1 ? -h * 0.06 * (0.5 + 0.5 * Math.sin(a * lobes + phase)) : 0;
    P.push(Math.cos(a) * rr, y + h * ringY[k] + droop, Math.sin(a) * rr);
    const c = k === 0 ? top.clone().lerp(bottom, 0.45) : bottom; col.push(c.r, c.g, c.b);
  }
  P.push(0, y + h * 0.18, 0); const dk = bottom.clone().multiplyScalar(0.45); col.push(dk.r, dk.g, dk.b);
  const idx = [], ring = (k, i) => 1 + k * seg + (i % seg), under = 1 + 2 * seg;
  for (let i = 0; i < seg; i++) {
    idx.push(0, ring(0, i + 1), ring(0, i));
    idx.push(ring(0, i), ring(0, i + 1), ring(1, i)); idx.push(ring(0, i + 1), ring(1, i + 1), ring(1, i));
    idx.push(under, ring(1, i), ring(1, i + 1));
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx); g.computeVertexNormals();
  const n = g.attributes.normal, p = g.attributes.position, v = new THREE.Vector3(), o = new THREE.Vector3();
  for (let i = 0; i < n.count; i++) {
    o.set(p.getX(i), (p.getY(i) - (y + h * 0.3)) * 0.8, p.getZ(i)).normalize();
    v.set(n.getX(i), n.getY(i), n.getZ(i)).lerp(o, 0.55).normalize();
    n.setXYZ(i, v.x, v.y, v.z);
  }
  return g.toNonIndexed();
}
function spruceGeometry(detail = 1, rnd = Math.random) {
  const parts = [];
  const trunk = new THREE.CylinderGeometry(0.03, 0.055, 0.3, 7); trunk.translate(0, 0.15, 0);
  parts.push(paint(trunk, '#5a3b26'));
  const tiers = detail ? [[0.37, 0.4, 0.12], [0.31, 0.36, 0.3], [0.24, 0.32, 0.47], [0.16, 0.3, 0.63]] : [[0.37, 0.58, 0.12], [0.23, 0.5, 0.45]];
  const tops = ['#3f7d47', '#3b7a44', '#438a4b', '#4a9150'].map((c) => new THREE.Color(c));
  const bots = ['#1b4127', '#1d452a', '#204a2c', '#244f30'].map((c) => new THREE.Color(c));
  tiers.forEach(([r, h, y], i) => parts.push(spruceTier(r, h, y, detail ? 12 : 8, rnd, tops[i], bots[i])));
  return mergeGeometries(parts);
}
function buttercupGeometry() {
  const cup = new THREE.ConeGeometry(0.075, 0.05, 5, 1, true); cup.rotateX(Math.PI); cup.translate(0, 0.3, 0);
  const center = new THREE.CircleGeometry(0.025, 5); center.rotateX(-Math.PI / 2); center.translate(0, 0.29, 0);
  const stem = new THREE.CylinderGeometry(0.006, 0.008, 0.28, 3); stem.translate(0, 0.14, 0);
  return merge([paint(cup, '#ffd21a'), paint(center, '#f0a000'), paint(stem, '#4d8f2c')]);
}
function daisyGeometry(color = '#ffffff') {
  const petals = new THREE.CircleGeometry(0.07, 8); petals.rotateX(-Math.PI / 2 + 0.25); petals.translate(0, 0.24, 0);
  const center = new THREE.CircleGeometry(0.024, 6); center.rotateX(-Math.PI / 2 + 0.25); center.translate(0, 0.245, 0.002);
  const stem = new THREE.CylinderGeometry(0.006, 0.008, 0.24, 3); stem.translate(0, 0.12, 0);
  return merge([paint(petals, color), paint(center, '#ffc21a'), paint(stem, '#4d8f2c')]);
}
function bellGeometry() {
  const parts = [];
  for (let k = 0; k < 3; k++) {
    const b = new THREE.ConeGeometry(0.04, 0.07, 5, 1, true); b.rotateX(Math.PI * 0.85); b.translate(Math.sin(k * 2.1) * 0.05, 0.28 + k * 0.05, Math.cos(k * 2.1) * 0.05);
    parts.push(paint(b, '#8a5ad8'));
  }
  const stem = new THREE.CylinderGeometry(0.006, 0.008, 0.36, 3); stem.translate(0, 0.18, 0);
  parts.push(paint(stem, '#4d8f2c'));
  return merge(parts);
}
function grassGeometry(rnd) {
  const pos = [], col = [];
  const dark = new THREE.Color('#3f7f2a'), light = new THREE.Color('#9ad155');
  for (let k = 0; k < 7; k++) {
    const a = rnd() * Math.PI * 2, r = rnd() * 0.1, h = 0.1 + rnd() * 0.16, w = 0.03;
    const x = Math.cos(a) * r, z = Math.sin(a) * r, lean = (rnd() - 0.5) * 0.25;
    const ca = Math.cos(a + 1.57) * w, sa = Math.sin(a + 1.57) * w;
    pos.push(x - ca, 0, z - sa, x + ca, 0, z + sa, x + lean * Math.cos(a), h, z + lean * Math.sin(a));
    col.push(dark.r, dark.g, dark.b, dark.r, dark.g, dark.b, light.r, light.g, light.b);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.computeVertexNormals();
  // point normals up for soft lighting
  const n = g.attributes.normal; for (let i = 0; i < n.count; i++) n.setXYZ(i, 0, 1, 0);
  return g;
}
function addWind(mat, uniforms, strength = 0.08) {
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = (sh, r) => {
    if (prev) prev(sh, r);
    sh.uniforms.uTime = uniforms.uTime;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float uTime;')
      .replace('#include <begin_vertex>', `#include <begin_vertex>
#ifdef USE_INSTANCING
  vec3 ip = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);
#else
  vec3 ip = vec3(0.0);
#endif
  float sway = sin(uTime*1.7 + ip.x*0.35 + ip.z*0.21) * 0.6 + sin(uTime*2.9 + ip.x*1.3) * 0.25;
  transformed.x += sway * ${strength.toFixed(3)} * max(transformed.y, 0.0) * 3.0;
  transformed.z += sway * ${(strength * 0.5).toFixed(3)} * max(transformed.y, 0.0) * 3.0;`);
  };
  mat.customProgramCacheKey = () => 'wind' + strength;
}

// --- Falu red barn ---------------------------------------------------------------------------
const FALU = '#9e2b25', TRIM = '#f4f1ea', ROOF = '#3a3a42', STONE = '#8d8f94';
function gablePrism(w, d, h) { // triangle prism, base on y=0, ridge along z
  const s = new THREE.Shape(); s.moveTo(-w / 2, 0); s.lineTo(w / 2, 0); s.lineTo(0, h); s.lineTo(-w / 2, 0);
  const g = new THREE.ExtrudeGeometry(s, { depth: d, bevelEnabled: false }); g.translate(0, 0, -d / 2); return g;
}
function xDoor(w, h, parts, x, y, z, rotY = 0) {
  // red leaf with white frame + white X, facing +z (before rotY)
  const add = (geo, col) => { geo.rotateY(rotY); geo.translate(x, y, z); parts.push(paint(geo, col)); };
  add(place(new THREE.BoxGeometry(w, h, 0.08), 0, h / 2, 0), '#8a241f');
  const t = 0.12;
  add(place(new THREE.BoxGeometry(w, t, 0.1), 0, t / 2, 0.02), TRIM);
  add(place(new THREE.BoxGeometry(w, t, 0.1), 0, h - t / 2, 0.02), TRIM);
  add(place(new THREE.BoxGeometry(t, h, 0.1), -w / 2 + t / 2, h / 2, 0.02), TRIM);
  add(place(new THREE.BoxGeometry(t, h, 0.1), w / 2 - t / 2, h / 2, 0.02), TRIM);
  const diag = Math.hypot(w - 2 * t, h - 2 * t), ang = Math.atan2(h - 2 * t, w - 2 * t);
  add(place(new THREE.BoxGeometry(diag, t * 0.9, 0.1), 0, h / 2, 0.03, 0, 0, ang), TRIM);
  add(place(new THREE.BoxGeometry(diag, t * 0.9, 0.1), 0, h / 2, 0.03, 0, 0, -ang), TRIM);
}
function windowGeo(w, h, parts, x, y, z, rotY = 0) {
  const add = (geo, col) => { geo.rotateY(rotY); geo.translate(x, y, z); parts.push(paint(geo, col)); };
  add(place(new THREE.BoxGeometry(w, h, 0.06), 0, 0, 0), '#2b3440');
  const t = 0.09;
  add(place(new THREE.BoxGeometry(w + t, t, 0.1), 0, h / 2, 0.02), TRIM);
  add(place(new THREE.BoxGeometry(w + t, t, 0.1), 0, -h / 2, 0.02), TRIM);
  add(place(new THREE.BoxGeometry(t, h + t, 0.1), -w / 2, 0, 0.02), TRIM);
  add(place(new THREE.BoxGeometry(t, h + t, 0.1), w / 2, 0, 0.02), TRIM);
  add(place(new THREE.BoxGeometry(t * 0.6, h, 0.1), 0, 0, 0.02), TRIM);
  add(place(new THREE.BoxGeometry(w, t * 0.6, 0.1), 0, 0, 0.02), TRIM);
}
function boardWalls(w, d, h, parts, color) {
  // walls with vertical board ridges
  parts.push(paint(place(new THREE.BoxGeometry(w, h, d), 0, h / 2, 0), color, 0.04));
  const rnd = mulberry32(5);
  for (let x = -w / 2 + 0.3; x < w / 2 - 0.2; x += 0.34) {
    parts.push(paint(place(new THREE.BoxGeometry(0.05, h - 0.05, 0.04), x, h / 2, d / 2 + 0.01), color, 0.1, rnd));
    parts.push(paint(place(new THREE.BoxGeometry(0.05, h - 0.05, 0.04), x, h / 2, -d / 2 - 0.01), color, 0.1, rnd));
  }
  for (let z = -d / 2 + 0.3; z < d / 2 - 0.2; z += 0.34) {
    parts.push(paint(place(new THREE.BoxGeometry(0.04, h - 0.05, 0.05), w / 2 + 0.01, h / 2, z), color, 0.1, rnd));
    parts.push(paint(place(new THREE.BoxGeometry(0.04, h - 0.05, 0.05), -w / 2 - 0.01, h / 2, z), color, 0.1, rnd));
  }
  // white corner boards
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) parts.push(paint(place(new THREE.BoxGeometry(0.2, h, 0.2), sx * (w / 2), h / 2, sz * (d / 2)), TRIM));
}
function buildBarn() {
  const { w, d } = LAYOUT.barn; const h = 3.6, rh = 2.9;
  const walls = [], roof = [];
  walls.push(paint(place(new THREE.BoxGeometry(w + 0.3, 0.7, d + 0.3), 0, 0.05, 0), STONE, 0.12));
  const wall = []; boardWalls(w, d, h, wall, FALU); wall.forEach((g) => g.translate(0, 0.4, 0)); walls.push(...wall);
  // gables
  const gab = gablePrism(w, d, rh); gab.translate(0, h + 0.4, 0); walls.push(paint(gab, FALU));
  // roof slabs
  const slope = Math.hypot(w / 2 + 0.45, rh + 0.35), ang = Math.atan2(rh + 0.35, w / 2 + 0.45);
  for (const s of [-1, 1]) {
    const slab = new THREE.BoxGeometry(slope, 0.18, d + 0.9);
    place(slab, s * (w / 4 + 0.05), h + 0.4 + rh / 2 + 0.02, 0, 0, 0, -s * ang);
    roof.push(paint(slab, ROOF, 0.05));
  }
  // ridge cap
  roof.push(paint(place(new THREE.BoxGeometry(0.3, 0.18, d + 0.95), 0, h + 0.4 + rh + 0.13, 0, 0, 0, Math.PI / 4), '#2c2c33'));
  // white fascia boards along the gable edges (front and back)
  const fl = Math.hypot(w / 2 + 0.3, rh + 0.22);
  for (const zf of [d / 2 + 0.46, -d / 2 - 0.46]) for (const s of [-1, 1]) {
    walls.push(paint(place(new THREE.BoxGeometry(fl, 0.2, 0.08), s * (w / 4 + 0.07), h + 0.4 + rh / 2 + 0.02, zf, 0, 0, -s * ang), TRIM));
  }
  // front: big X doors + loft door + windows
  xDoor(1.5, 2.7, walls, -0.76, 0.45, d / 2 + 0.03);
  xDoor(1.5, 2.7, walls, 0.76, 0.45, d / 2 + 0.03);
  walls.push(paint(place(new THREE.BoxGeometry(3.3, 0.18, 0.12), 0, 3.25, d / 2 + 0.06), TRIM));
  xDoor(1.3, 1.3, walls, 0, 4.4, d / 2 + 0.03);
  windowGeo(0.8, 0.8, walls, -2.6, 2.2, d / 2 + 0.03);
  windowGeo(0.8, 0.8, walls, 2.6, 2.2, d / 2 + 0.03);
  for (const zz of [-3, 0, 3]) {
    windowGeo(0.9, 0.7, walls, w / 2 + 0.03, 2.4, zz, Math.PI / 2);
    windowGeo(0.9, 0.7, walls, -w / 2 - 0.03, 2.4, zz, -Math.PI / 2);
  }
  xDoor(1.4, 2.2, walls, -w / 2 - 0.03, 0.45, -2.6 + 1.5, -Math.PI / 2);
  // ramp (loge) at back
  walls.push(paint(place(new THREE.BoxGeometry(3.2, 0.25, 4), 0, 1.1, -d / 2 - 1.9, -0.45), '#7c6a55', 0.1));
  // cupola
  walls.push(paint(place(new THREE.BoxGeometry(0.9, 0.8, 0.9), 0, h + 0.4 + rh + 0.2, 1.2), FALU));
  const cr = gablePrism(1.25, 1.25, 0.6); cr.translate(0, h + 0.4 + rh + 0.6, 1.2); roof.push(paint(cr, ROOF));
  for (const s of [-1, 1]) walls.push(paint(place(new THREE.BoxGeometry(0.06, 0.6, 0.94), s * 0.46, h + 0.4 + rh + 0.2, 1.2), TRIM));
  const gw = merge(walls), gr = merge(roof);
  const grp = new THREE.Group();
  const mw = new THREE.Mesh(gw, toyMat(0xffffff, { vc: true, roughness: 0.7, clearcoat: 0.35 }));
  const mr = new THREE.Mesh(gr, toyMat(0xffffff, { vc: true, roughness: 0.55, clearcoat: 0.6 }));
  for (const m of [mw, mr]) { m.castShadow = true; m.receiveShadow = true; grp.add(m); }
  return grp;
}
function buildCottage(scale = 1) {
  const w = 4.2, d = 5.6, h = 2.5, rh = 2.0;
  const walls = [], roof = [];
  walls.push(paint(place(new THREE.BoxGeometry(w + 0.2, 0.5, d + 0.2), 0, 0, 0), STONE, 0.1));
  const wall = []; boardWalls(w, d, h, wall, '#a8322b'); wall.forEach((g) => g.translate(0, 0.25, 0)); walls.push(...wall);
  const gab = gablePrism(d, w, rh); gab.rotateY(Math.PI / 2); gab.translate(0, h + 0.25, 0); walls.push(paint(gab, '#a8322b'));
  const slope = Math.hypot(d / 2 + 0.4, rh + 0.3), ang = Math.atan2(rh + 0.3, d / 2 + 0.4);
  for (const s of [-1, 1]) {
    const slab = new THREE.BoxGeometry(w + 0.7, 0.16, slope);
    place(slab, 0, h + 0.25 + rh / 2 + 0.02, s * (d / 4 + 0.05), s * ang, 0, 0);
    roof.push(paint(slab, '#40404a', 0.05));
  }
  // chimney
  walls.push(paint(place(new THREE.BoxGeometry(0.45, 1.3, 0.45), 1.1, h + rh + 0.3, -0.7), '#e9e4dc'));
  // front (long side +z): door + windows
  const zf = d / 2 + 0.03;
  walls.push(paint(place(new THREE.BoxGeometry(0.9, 1.8, 0.08), 0, 0.25 + 0.9, zf), '#2f5d8a'));
  walls.push(paint(place(new THREE.BoxGeometry(1.1, 0.12, 0.12), 0, 0.25 + 1.86, zf + 0.02), TRIM));
  for (const s of [-1, 1]) walls.push(paint(place(new THREE.BoxGeometry(0.12, 1.9, 0.12), s * 0.5, 0.25 + 0.95, zf + 0.02), TRIM));
  windowGeo(0.8, 0.8, walls, -1.35, 1.6, zf); windowGeo(0.8, 0.8, walls, 1.35, 1.6, zf);
  windowGeo(0.7, 0.7, walls, w / 2 + 0.03, 1.6, 0, Math.PI / 2); windowGeo(0.7, 0.7, walls, -w / 2 - 0.03, 1.6, 0, -Math.PI / 2);
  windowGeo(0.6, 0.6, walls, 0, h + 0.9, d / 2 + 0.03 - 0.0); // not visible (under roof) harmless
  // white gable trim on side ends
  const grp = new THREE.Group();
  const mw = new THREE.Mesh(merge(walls), toyMat(0xffffff, { vc: true, roughness: 0.7, clearcoat: 0.35 }));
  const mr = new THREE.Mesh(merge(roof), toyMat(0xffffff, { vc: true, roughness: 0.55, clearcoat: 0.6 }));
  for (const m of [mw, mr]) { m.castShadow = true; m.receiveShadow = true; grp.add(m); }
  grp.scale.setScalar(scale);
  return grp;
}

// --- Gärdesgård (Swedish roundpole fence) ------------------------------------------------------
function buildFence(points, rnd) {
  const pole = new THREE.CylinderGeometry(0.045, 0.05, 1, 8);
  const posts = [], slants = [];
  const up = new THREE.Vector3(0, 1, 0);
  for (let s = 0; s < points.length - 1; s++) {
    const [ax, az] = points[s], [bx, bz] = points[s + 1];
    const len = Math.hypot(bx - ax, bz - az); const n = Math.max(1, Math.round(len / 1.1));
    const dir = new THREE.Vector3((bx - ax) / len, 0, (bz - az) / len);
    const side = new THREE.Vector3(-dir.z, 0, dir.x);
    for (let i = 0; i <= n; i++) {
      if (i === n && s < points.length - 2) continue;
      const x = lerp(ax, bx, i / n), z = lerp(az, bz, i / n), y = heightAt(x, z);
      for (const o of [-0.09, 0.09]) posts.push(new THREE.Matrix4().compose(new THREE.Vector3(x + side.x * o, y + 0.6, z + side.z * o), new THREE.Quaternion().setFromAxisAngle(dir, o * 0.6), new THREE.Vector3(1, 1.3, 1)));
      if (i < n) {
        const x2 = lerp(ax, bx, (i + 0.5) / n), z2 = lerp(az, bz, (i + 0.5) / n), y2 = heightAt(x2, z2);
        const q = new THREE.Quaternion().setFromAxisAngle(side, -0.85 + (rnd() - 0.5) * 0.1);
        for (let k = 0; k < 3; k++) {
          const off = (k - 1) * 0.28;
          slants.push(new THREE.Matrix4().compose(new THREE.Vector3(x2 + dir.x * off, y2 + 0.52 + k * 0.03, z2 + dir.z * off), q, new THREE.Vector3(1, 1.75, 1)));
        }
      }
    }
  }
  const mat = toyMat('#9a8469', { roughness: 0.85, clearcoat: 0.1 });
  const g = new THREE.Group();
  for (const list of [posts, slants]) {
    const im = new THREE.InstancedMesh(pole, mat, list.length);
    list.forEach((m, i) => { im.setColorAt(i, new THREE.Color().setHSL(0.08, 0.18, 0.45 + rnd() * 0.12)); im.setMatrixAt(i, m); });
    im.castShadow = true; im.receiveShadow = true; g.add(im);
  }
  return g;
}

// --- Pond -------------------------------------------------------------------------------------
function buildPond(uniforms, scene) {
  const P = LAYOUT.pond; const g = new THREE.Group();
  const geo = new THREE.CircleGeometry(P.r + 0.35, 64); geo.rotateX(-Math.PI / 2);
  // wobble the rim
  const pa = geo.attributes.position;
  for (let i = 1; i < pa.count; i++) { const x = pa.getX(i), z = pa.getZ(i); const a = Math.atan2(z, x); const k = 1 + 0.08 * Math.sin(a * 3 + 1) + 0.05 * Math.sin(a * 5); pa.setXYZ(i, x * k, 0, z * k); }
  const mat = new THREE.ShaderMaterial({
    transparent: true, fog: true, depthWrite: false,
    uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, { uTime: { value: 0 }, uR: { value: P.r + 0.35 } }]),
    vertexShader: `varying vec3 vW; varying vec2 vL;
#include <fog_pars_vertex>
void main(){ vL = position.xz; vec4 w = modelMatrix*vec4(position,1.0); vW = w.xyz; vec4 mvPosition = viewMatrix*w; gl_Position = projectionMatrix*mvPosition;
#include <fog_vertex>
}`,
    fragmentShader: `uniform float uTime; uniform float uR; varying vec3 vW; varying vec2 vL;
#include <common>
#include <fog_pars_fragment>
void main(){
  vec3 V = normalize(cameraPosition - vW);
  float r = length(vL)/uR;
  vec2 p = vL*2.2;
  float w1 = sin(p.x*1.7 + uTime*1.1 + sin(p.y*1.3+uTime*0.7)*1.5);
  float w2 = sin(p.y*2.3 - uTime*0.9 + sin(p.x*1.1)*1.2);
  vec3 N = normalize(vec3(w1*0.06, 1.0, w2*0.06));
  float fres = pow(1.0 - max(dot(N, V), 0.0), 3.0);
  vec3 deep = vec3(0.05, 0.22, 0.32), shallow = vec3(0.22, 0.52, 0.5), sky = vec3(0.72, 0.85, 1.0);
  vec3 col = mix(deep, shallow, smoothstep(0.5, 1.0, r));
  col = mix(col, sky, 0.25 + 0.6*fres);
  vec3 L = normalize(vec3(0.680, 0.580, 0.300)); vec3 H = normalize(L+V);
  col += vec3(1.0,0.95,0.8) * pow(max(dot(N,H),0.0), 180.0) * 2.5;
  float ring = smoothstep(0.86, 0.97, r) * (0.5+0.5*sin(r*60.0 - uTime*2.0));
  col = mix(col, vec3(0.9,0.97,1.0), ring*0.25);
  gl_FragColor = vec4(col, 0.9 * (1.0 - smoothstep(0.97, 1.0, r)*0.6));
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
}`,
  });
  mat.uniforms.uTime = uniforms.uTime;
  const water = new THREE.Mesh(geo, mat); water.position.set(P.x, -0.22, P.z); water.renderOrder = 2;
  g.add(water);
  // lily pads (the frog's home!)
  const padGeo = new THREE.CircleGeometry(0.32, 10, 0.3, Math.PI * 2 - 0.6); padGeo.rotateX(-Math.PI / 2);
  const padMat = toyMat('#3f9a3a', { roughness: 0.5, side: THREE.DoubleSide });
  const rnd = mulberry32(99);
  const pads = [[-1.2, 0.6], [-0.4, -1.3], [0.9, -0.6], [1.5, 0.9], [-1.9, -0.5], [0.2, 1.4]];
  for (const [x, z] of pads) {
    const m = new THREE.Mesh(padGeo, padMat); m.position.set(P.x + x, -0.2, P.z + z); m.rotation.y = rnd() * 6; m.scale.setScalar(0.8 + rnd() * 0.5); m.receiveShadow = true; g.add(m);
  }
  const lily = merge([paint(place(new THREE.ConeGeometry(0.14, 0.14, 6, 1, true), 0, 0.05, 0, Math.PI), '#ffb6d0'), paint(place(new THREE.CircleGeometry(0.05, 6), 0, 0.02, 0, -Math.PI / 2), '#ffd84a')]);
  for (const [x, z] of [[-1.2, 0.6], [1.5, 0.9]]) { const m = new THREE.Mesh(lily, toyMat(0xffffff, { vc: true })); m.position.set(P.x + x + 0.05, -0.2, P.z + z); g.add(m); }
  // reeds + cattails
  const reed = paint(place(new THREE.CylinderGeometry(0.012, 0.02, 1, 3), 0, 0.5, 0), '#5c8f36');
  const cat = paint(place(new THREE.CylinderGeometry(0.035, 0.035, 0.16, 5), 0, 0.92, 0), '#6b4426');
  const reedGeo = merge([reed, cat]);
  const reedM = [];
  for (let i = 0; i < 46; i++) {
    const a = 3.6 + rnd() * 2.1, rr = P.r * (0.92 + rnd() * 0.25);
    const x = P.x + Math.cos(a) * rr, z = P.z + Math.sin(a) * rr;
    reedM.push(new THREE.Matrix4().compose(new THREE.Vector3(x, heightAt(x, z) - 0.05, z), new THREE.Quaternion().setFromEuler(new THREE.Euler((rnd() - 0.5) * 0.25, 0, (rnd() - 0.5) * 0.25)), new THREE.Vector3(1, 0.7 + rnd() * 0.6, 1)));
  }
  const rm = new THREE.InstancedMesh(reedGeo, vcMat({ flat: true, roughness: 0.8 }), reedM.length);
  reedM.forEach((m, i) => rm.setMatrixAt(i, m)); rm.castShadow = true; g.add(rm);
  addWind(rm.material, uniforms, 0.05);
  return g;
}

// --- Dirt path (ribbon following the terrain) ---------------------------------------------------
function buildPath(rnd) {
  const curve = new THREE.CatmullRomCurve3(LAYOUT.pathPts.map(([x, z]) => new THREE.Vector3(x, 0, z)), false, 'centripetal');
  const n = 160; const pos = [], col = []; const c = new THREE.Color();
  const pts = curve.getSpacedPoints(n);
  const L = [], R = [];
  for (let i = 0; i <= n; i++) {
    const p = pts[i], q = pts[Math.min(n, i + 1)], o = pts[Math.max(0, i - 1)];
    const dx = q.x - o.x, dz = q.z - o.z, len = Math.hypot(dx, dz) || 1;
    const w = 0.75 + 0.18 * Math.sin(i * 0.7) + 0.1 * rnd() + (i > n * 0.6 ? 0.2 : 0);
    const nx = -dz / len * w, nz = dx / len * w;
    L.push([p.x + nx, p.z + nz]); R.push([p.x - nx, p.z - nz]);
  }
  const push = (x, z) => { pos.push(x, heightAt(x, z) + 0.05, z); };
  for (let i = 0; i < n; i++) {
    const quad = [L[i], R[i], L[i + 1], R[i + 1]];
    const tri = [[0, 2, 1], [1, 2, 3]];
    for (const t of tri) {
      c.set(COL.path[Math.floor(rnd() * 3)].getHex()).multiplyScalar(0.95 + rnd() * 0.1);
      for (const k of t) { push(...quad[k]); col.push(c.r, c.g, c.b); }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, flatShading: false, polygonOffset: true, polygonOffsetFactor: -1 }));
  m.receiveShadow = true;
  return m;
}

// --- Clouds -----------------------------------------------------------------------------------
function cloudGeometry(rnd, puffs = 9) {
  const parts = [];
  for (let i = 0; i < puffs; i++) {
    const r = 1 + rnd() * 1.4; const x = (i / (puffs - 1) - 0.5) * 7 + (rnd() - 0.5) * 1.2;
    const y = r * 0.45 + rnd() * 0.5 - Math.abs(x) * 0.12, z = (rnd() - 0.5) * 2.2;
    const g = new THREE.IcosahedronGeometry(r, 3); g.scale(1, 0.8, 1); g.translate(x, y, z); g.deleteAttribute('uv'); parts.push(g);
  }
  return mergeGeometries(parts); // keeps each puff's smooth normals
}
function cloudMaterial() {
  return new THREE.ShaderMaterial({
    fog: true,
    uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, { sunDir: { value: SUN_DIR.clone() } }]),
    vertexShader: `varying vec3 vN; varying vec3 vW;
#include <fog_pars_vertex>
void main(){ vN = normalize(mat3(modelMatrix) * normal); vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz;
  vec4 mvPosition = viewMatrix * w; gl_Position = projectionMatrix * mvPosition;
#include <fog_vertex>
}`,
    fragmentShader: `uniform vec3 sunDir; varying vec3 vN; varying vec3 vW;
#include <common>
#include <fog_pars_fragment>
void main(){
  vec3 N = normalize(vN); vec3 V = normalize(cameraPosition - vW);
  float lit = smoothstep(-0.35, 0.9, dot(N, sunDir));                 // soft wrap lighting
  vec3 shade = vec3(0.62, 0.70, 0.86), light = vec3(1.18, 1.14, 1.06);
  vec3 c = mix(shade, light, lit);
  c = mix(c, shade * 0.92, smoothstep(0.0, -0.8, N.y) * 0.6);          // flat grey-blue bellies
  c += vec3(1.0, 0.95, 0.85) * pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.35 * (0.4 + lit); // silver lining
  gl_FragColor = vec4(c, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
}`,
  });
}
function buildClouds(rnd, count = 9) {
  const g = new THREE.Group();
  const mat = cloudMaterial();
  const clouds = [];
  for (let i = 0; i < count; i++) {
    const m = new THREE.Mesh(cloudGeometry(rnd, 7 + Math.floor(rnd() * 5)), mat);
    const far = i < count * 0.7;
    const s = far ? 6 + rnd() * 5 : 2.6 + rnd() * 2;
    m.scale.set(s, s * (0.8 + rnd() * 0.3), s);
    const x = (rnd() - 0.5) * (far ? 800 : 300), z = far ? -300 - rnd() * 300 : -130 - rnd() * 80, y = far ? 120 + rnd() * 110 : 62 + rnd() * 30;
    m.position.set(x, y, z); m.rotation.y = (rnd() - 0.5) * 0.4;
    m.userData.speed = (far ? 1.2 : 0.7) * (0.6 + rnd() * 0.6);
    m.userData.span = far ? 420 : 170;
    clouds.push(m); g.add(m);
  }
  g.userData.clouds = clouds;
  return g;
}

// --- Sky --------------------------------------------------------------------------------------
function buildSky() {
  const geo = new THREE.SphereGeometry(2400, 48, 24);
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { top: { value: new THREE.Color(SKY.zenith) }, mid: { value: new THREE.Color(SKY.mid) }, bottom: { value: new THREE.Color(SKY.horizon) }, sunDir: { value: SUN_DIR.clone() } },
    vertexShader: 'varying vec3 vD; void main(){ vD = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
    fragmentShader: `uniform vec3 top; uniform vec3 mid; uniform vec3 bottom; uniform vec3 sunDir; varying vec3 vD;
void main(){
  vec3 d = normalize(vD); float y = max(d.y, 0.0);
  vec3 c = mix(bottom, mid, smoothstep(0.0, 0.22, y));
  c = mix(c, top, smoothstep(0.2, 0.85, y));
  float s = max(dot(d, sunDir), 0.0);
  c += vec3(1.0, 0.86, 0.62) * (pow(s, 6.0) * 0.18 + pow(s, 60.0) * 0.45);   // warm glow around the sun
  c += vec3(1.0, 0.97, 0.88) * smoothstep(0.9993, 0.9997, s) * 4.0;           // sun disc (blooms)
  c = mix(c, bottom * 1.05, (1.0 - smoothstep(-0.02, 0.06, d.y)) * 0.6);       // milky haze band at the horizon
  gl_FragColor = vec4(c, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`,
  });
  const m = new THREE.Mesh(geo, mat); m.renderOrder = -10; m.frustumCulled = false;
  return m;
}

// ---------------------------------------------------------------------------------------------
// Scatter helpers
// ---------------------------------------------------------------------------------------------
function instancedFromParts(parts, matrices, opts = {}) {
  // parts: [{geometry, material, matrix}] from natureParts ; returns group of InstancedMeshes
  const g = new THREE.Group();
  for (const p of parts) {
    const im = new THREE.InstancedMesh(p.geometry, p.material, matrices.length);
    const tmp = new THREE.Matrix4();
    matrices.forEach((m, i) => { tmp.multiplyMatrices(m, p.matrix); im.setMatrixAt(i, tmp); });
    im.castShadow = !!opts.cast; im.receiveShadow = opts.receive ?? true;
    im.userData.sharedGeo = true;
    im.computeBoundingSphere();
    g.add(im);
  }
  return g;
}
function mtx(x, y, z, ry, s, sy = s) { return new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(0, ry, 0)), new THREE.Vector3(s, sy, s)); }

function blockedStatic(x, z, pad = 0) {
  const B = LAYOUT.barn; const lx = x - B.x, lz = z - B.z; const c = Math.cos(-B.rot), s = Math.sin(-B.rot);
  const bx = lx * c - lz * s, bz = lx * s + lz * c;
  if (Math.abs(bx) < B.w / 2 + 1 + pad && bz > -B.d / 2 - 4 - pad && bz < B.d / 2 + 1.2 + pad) return true;
  const P = LAYOUT.pond; if (Math.hypot(x - P.x, z - P.z) < P.r + 0.9 + pad) return true;
  return false;
}

// ---------------------------------------------------------------------------------------------
// World builder (shared by the farm and the backdrops)
// ---------------------------------------------------------------------------------------------
async function buildWorld({ quality = 'high', backdrop = false } = {}) {
  const nature = await natureParts().catch((e) => { console.warn('World3D: nature assets missing', e); return {}; });
  const scene = new THREE.Scene();
  const uniforms = { uTime: { value: 0 } };
  const rnd = mulberry32(1234);
  const hi = quality !== 'low';

  scene.background = new THREE.Color('#bcd6f0');
  scene.fog = new THREE.FogExp2(new THREE.Color(SKY.fog), 0.00125);  // aerial perspective: far ridges go soft blue

  // Lights: warm late-morning key from the right (rakes across the mountains), cool sky fill
  const hemi = new THREE.HemisphereLight(0xaac6f2, 0x5d7a3e, 0.85);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffe2b8, 4.0);
  sun.position.copy(SUN_DIR).multiplyScalar(70);
  sun.target.position.set(0, 0, -3);
  sun.castShadow = true;
  const ss = hi ? 2048 : 1024;
  sun.shadow.mapSize.set(ss, ss);
  const sc = sun.shadow.camera; sc.left = -36; sc.right = 36; sc.top = 30; sc.bottom = -30; sc.near = 10; sc.far = 160;
  sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.03; sun.shadow.radius = 3;
  scene.add(sun, sun.target);

  scene.add(buildSky());

  // Terrain
  const tmat = terrainMaterial(uniforms);
  const near = new THREE.Mesh(terrainGeometry('near' + hi, { x0: -150, x1: 150, z0: -110, z1: 100, nx: hi ? 150 : 100, nz: hi ? 105 : 70 }), tmat);
  near.receiveShadow = true;
  const warp = {
    x: (u) => { const s = u * 2 - 1; return Math.sign(s) * Math.pow(Math.abs(s), 1.5) * 1100; },
    z: (v) => lerp(-95, -900, Math.pow(v, 1.35)) + 0,
  };
  const farGeo = terrainGeometry('far' + hi, { nx: hi ? 420 : 200, nz: hi ? 220 : 110, warp, far: true });
  const far = new THREE.Mesh(farGeo, tmat);
  far.receiveShadow = false;
  // side wings so orbiting never shows the edge of the world
  const sideGeo = terrainGeometry('side', { x0: -700, x1: 700, z0: -100, z1: 400, nx: 70, nz: 25, far: true });
  const side = new THREE.Mesh(sideGeo, tmat); side.position.y = -0.2;
  for (const m of [near, far, side]) m.userData.sharedGeo = true;
  scene.add(near, far, side);

  // Spruce forests
  const spruceHi = spruceGeometry(1, rnd), spruceLo = spruceGeometry(0, rnd);
  const spruceMat = vcMat({ roughness: 0.8, flat: false });
  const nearT = [], farT = [];
  const addTree = (x, z, s) => {
    const y = heightAt(x, z) - 0.1;
    const m = mtx(x, y, z, rnd() * 6.28, s * (0.8 + rnd() * 0.3), s);
    (Math.hypot(x, z - 10) < 90 ? nearT : farT).push(m);
  };
  // clusters at meadow edge (framing), behind the fence line
  const clusters = [[-24, -18, 7, 14], [-30, -4, 6, 10], [-19, -26, 6, 10], [22, -22, 8, 14], [31, -10, 6, 9], [5, -30, 9, 14], [-8, -30, 7, 10], [36, 4, 5, 6], [-34, 10, 6, 8], [16, -32, 6, 8]];
  for (const [cx, cz, r, n] of clusters) {
    for (let i = 0; i < n; i++) {
      const a = rnd() * 6.28, d = Math.sqrt(rnd()) * r; const x = cx + Math.cos(a) * d, z = cz + Math.sin(a) * d;
      if (farmRR(x, z) < 19 || blockedStatic(x, z, 2)) continue;
      addTree(x, z, 3.2 + rnd() * 3.2);
    }
  }
  // forest belts on hills + mountain lower slopes
  for (let i = 0; i < (hi ? 9000 : 4500); i++) {
    const x = (rnd() - 0.5) * 900, z = -30 - rnd() * 420;
    const rr = farmRR(x, z); if (rr < 42) continue;
    const y = heightAt(x, z); const m = mountainMask(x, z);
    const f = N2(x * 0.018, z * 0.018) + 0.35 * N1(x * 0.06, z * 0.06);
    if (f < 0.22 + m * 0.1) continue;
    if (y > 44 + 8 * N4(x * 0.02, z * 0.02)) continue;
    // slope check
    const sl = Math.abs(heightAt(x + 2, z) - y) + Math.abs(heightAt(x, z + 2) - y);
    if (sl > 3.2) continue;
    addTree(x, z, 4 + rnd() * 4 + m * 6);
  }
  // a few more near hills in front/side so the meadow is framed when orbiting
  for (let i = 0; i < 400; i++) {
    const x = (rnd() - 0.5) * 200, z = -10 + rnd() * 90; const rr = farmRR(x, z);
    if (rr < 30 || N2(x * 0.03, z * 0.03) < 0.15) continue;
    addTree(x, z, 3.5 + rnd() * 3.5);
  }
  const mkForest = (geo, list, cast) => {
    const im = new THREE.InstancedMesh(geo, spruceMat, list.length);
    list.forEach((m, i) => im.setMatrixAt(i, m)); im.castShadow = cast; im.receiveShadow = true; im.computeBoundingSphere(); scene.add(im); return im;
  };
  mkForest(spruceHi, nearT, true); mkForest(spruceLo, farT, false);

  // Barn
  const barn = buildBarn();
  const B = LAYOUT.barn; barn.position.set(B.x, heightAt(B.x, B.z) - 0.2, B.z); barn.rotation.y = B.rot;
  scene.add(barn);
  // Distant cottage + a second one far on the hill
  const cot = buildCottage(1);
  cot.position.set(LAYOUT.cottage.x, heightAt(LAYOUT.cottage.x, LAYOUT.cottage.z) - 0.1, LAYOUT.cottage.z); cot.rotation.y = LAYOUT.cottage.rot;
  scene.add(cot);
  const cot2 = buildCottage(1.1);
  cot2.position.set(-62, heightAt(-62, -58) - 0.1, -58); cot2.rotation.y = 0.5; scene.add(cot2);

  // Fence around the back of the pasture
  const fencePts = [];
  const M = LAYOUT.meadow;
  for (let t = 3.55; t <= 6.2; t += 0.12) fencePts.push([M.x + Math.cos(t) * (M.rx + 3.2), M.z + Math.sin(t) * (M.rz + 3.2)]);
  scene.add(buildFence(fencePts, rnd));
  const fence2 = []; for (let t = 0.02; t <= 0.46; t += 0.11) fence2.push([M.x + Math.cos(t) * (M.rx + 3.8), M.z + Math.sin(t) * (M.rz + 3.8)]);
  scene.add(buildFence(fence2, rnd));

  // Pond + path
  scene.add(buildPond(uniforms, scene));
  scene.add(buildPath(rnd));

  // Meadow flowers & grass (instanced)
  const flowerCount = backdrop ? 9000 : (hi ? 9000 : 4500);
  const bcM = [], daM = [], beM = [], grM = [];
  const inMeadow = (x, z) => farmRR(x, z) < 30 && !blockedStatic(x, z, -0.6) && distToPath(x, z) > 1.1;
  let tries = 0;
  while (bcM.length + daM.length + beM.length < flowerCount && tries++ < flowerCount * 4) {
    const x = (rnd() - 0.5) * 80, z = (rnd() - 0.5) * 64 + 2;
    if (!inMeadow(x, z)) continue;
    const clump = N1(x * 0.15, z * 0.15) * 0.5 + 0.5; if (rnd() > 0.35 + clump * 0.8) continue;
    const y = heightAt(x, z); const s = 0.85 + rnd() * 0.6; const r = rnd();
    const m = mtx(x, y, z, rnd() * 6.28, s * 1.25, s);
    if (r < 0.74) bcM.push(m); else if (r < 0.9) daM.push(m); else beM.push(m);
  }
  const grassCount = hi ? 4500 : 2000;
  for (let i = 0; i < grassCount; i++) {
    const x = (rnd() - 0.5) * 80, z = (rnd() - 0.5) * 64 + 2;
    if (farmRR(x, z) > 34 || blockedStatic(x, z, -0.8)) continue;
    grM.push(mtx(x, heightAt(x, z) - 0.02, z, rnd() * 6.28, 0.8 + rnd() * 0.9, 0.7 + rnd() * 0.8));
  }
  const flowerMat = vcMat({ flat: false, roughness: 0.6, side: THREE.DoubleSide });
  flowerMat.emissive = new THREE.Color('#2a2200'); addWind(flowerMat, uniforms, 0.06);
  const grassMat = vcMat({ flat: false, roughness: 0.9, side: THREE.DoubleSide }); addWind(grassMat, uniforms, 0.1);
  const mkI = (geo, mat, list, cast = false) => { const im = new THREE.InstancedMesh(geo, mat, list.length); list.forEach((m, i) => im.setMatrixAt(i, m)); im.castShadow = cast; im.receiveShadow = true; im.computeBoundingSphere(); scene.add(im); return im; };
  mkI(buttercupGeometry(), flowerMat, bcM); mkI(daisyGeometry(), flowerMat, daM); mkI(bellGeometry(), flowerMat, beM);
  mkI(grassGeometry(rnd), grassMat, grM);

  // Nature accents from Quaternius (rocks, bushes, ferns, a few deciduous trees near the barn)
  if (nature.Rock_Medium_1) {
    const rocks = { Rock_Medium_1: [], Rock_Medium_2: [], Rock_Medium_3: [] };
    const rk = Object.keys(rocks);
    for (let i = 0; i < 40; i++) {
      const a = rnd() * 6.28; const P = LAYOUT.pond;
      let x, z, s;
      if (i < 9) { x = P.x + Math.cos(a) * (P.r + 0.5); z = P.z + Math.sin(a) * (P.r + 0.4); s = 0.12 + rnd() * 0.12; if (z > P.z + 1 && Math.abs(x - P.x) < 2) continue; }
      else { x = (rnd() - 0.5) * 90; z = -20 - rnd() * 30; s = 0.3 + rnd() * 0.6; if (farmRR(x, z) < 22) continue; }
      rocks[rk[i % 3]].push(mtx(x, heightAt(x, z) - s * 0.3, z, rnd() * 6.28, s));
    }
    for (const k of rk) scene.add(instancedFromParts(nature[k], rocks[k], { cast: true }));
    // soft round shrubs (the Quaternius bush texture renders as dark maroon blobs)
    for (const [x, z] of [[-7.2, -13], [-17, -1.5], [-16.5, -3.2], [-5.5, -14.5], [13.5, -13], [15, -11.5], [-9, 3.5]]) {
      const b = flowerBushGroup(rnd, rnd() < 0.5); b.position.set(x, heightAt(x, z) - 0.05, z); b.rotation.y = rnd() * 6; b.scale.setScalar(1.1 + rnd() * 0.5); scene.add(b);
    }
    const ferns = [];
    for (let i = 0; i < 30; i++) { const c = clusters[i % clusters.length]; const x = c[0] + (rnd() - 0.5) * c[2] * 1.6, z = c[1] + (rnd() - 0.5) * c[2] * 1.6; if (farmRR(x, z) < 19) continue; ferns.push(mtx(x, heightAt(x, z), z, rnd() * 6, 0.6 + rnd() * 0.4)); }
    scene.add(instancedFromParts(nature.Fern_1, ferns));
    const birch = [];
    for (const [x, z, s] of [[-19, -12, 0.75], [-4, -17.5, 0.62], [19, -3, 0.7], [-21, 3, 0.66]]) birch.push(mtx(x, heightAt(x, z) - 0.1, z, rnd() * 6, s));
    scene.add(instancedFromParts(nature.CommonTree_1, birch, { cast: true }));
  }

  // Clouds
  const clouds = buildClouds(rnd, hi ? 12 : 8);
  scene.add(clouds);

  // Environment (for clearcoat reflections): a tiny sky gradient scene
  const update = (t, dt) => {
    uniforms.uTime.value = t;
    for (const c of clouds.userData.clouds) {
      c.position.x += c.userData.speed * dt * (REDUCED_MOTION ? 0.3 : 1);
      if (c.position.x > c.userData.span) c.position.x -= c.userData.span * 2;
    }
  };
  return { scene, sun, hemi, uniforms, update };
}

function makeEnvironment(renderer) {
  const pm = new THREE.PMREMGenerator(renderer);
  const s = new THREE.Scene();
  const sky = buildSky(); sky.scale.setScalar(0.01); s.add(sky);
  const ground = new THREE.Mesh(new THREE.CircleGeometry(20, 16), new THREE.MeshBasicMaterial({ color: '#6aa048' })); ground.rotation.x = -Math.PI / 2; ground.position.y = -2; s.add(ground);
  const rt = pm.fromScene(s, 0.02);
  pm.dispose(); sky.geometry.dispose(); sky.material.dispose(); ground.geometry.dispose(); ground.material.dispose();
  return rt;
}

function makeRenderer(canvas, { width, height, pixelRatio, preserve = false }) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: preserve, alpha: false });
  renderer.setPixelRatio(pixelRatio);
  renderer.setSize(width, height, false);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.shadowMap.autoUpdate = false;
  renderer.shadowMap.needsUpdate = true;
  return renderer;
}

const TiltShiftShader = {
  uniforms: { tDiffuse: { value: null }, uFocus: { value: 0.45 }, uRange: { value: 0.18 }, uAmount: { value: 2.2 }, uTop: { value: 0.3 }, uRes: { value: new THREE.Vector2(1, 1) }, uDir: { value: new THREE.Vector2(1, 0) } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
  fragmentShader: `uniform sampler2D tDiffuse; uniform float uFocus; uniform float uRange; uniform float uAmount; uniform float uTop; uniform vec2 uRes; uniform vec2 uDir; varying vec2 vUv;
void main(){ float dy = vUv.y - uFocus; float d = dy < 0.0 ? max(-dy - uRange, 0.0) : max(dy - uRange, 0.0) * uTop;
 float spread = d * uAmount * uRes.y * 0.05; float st = spread / 8.0; vec4 s = vec4(0.0); float tw = 0.0;
 for (int i=-10;i<=10;i++){ float fi = float(i); float w = exp(-fi*fi/50.0); s += texture2D(tDiffuse, vUv + uDir*fi*st/uRes) * w; tw += w; }
 gl_FragColor = s/tw; }`,
};

// Final grade (display space, after tone mapping): gentle S-curve, a little extra colour, warm highlights,
// cool shadows and a soft vignette.
const GradeShader = {
  uniforms: { tDiffuse: { value: null }, uVignette: { value: 0.18 } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
  fragmentShader: `uniform sampler2D tDiffuse; uniform float uVignette; varying vec2 vUv;
void main(){
  vec3 c = texture2D(tDiffuse, vUv).rgb;
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = mix(vec3(l), c, 1.12);
  c = mix(c, c * c * (3.0 - 2.0 * c), 0.22);
  c *= mix(vec3(0.97, 0.99, 1.04), vec3(1.03, 1.0, 0.95), smoothstep(0.2, 0.8, l));
  vec2 q = (vUv - 0.5) * vec2(1.0, 0.85);
  c *= 1.0 - uVignette * smoothstep(0.25, 0.75, length(q));
  gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`,
};

function makeComposer(renderer, scene, camera, w, h, { ao = true, bloom = true, tilt = null, grade = true } = {}) {
  // Multisampled HDR target: without it the composer renders with no antialiasing at all (jaggy edges).
  const prx = renderer.getPixelRatio();
  const rt = new THREE.WebGLRenderTarget(Math.round(w * prx), Math.round(h * prx), { type: THREE.HalfFloatType, samples: 4 });
  const composer = new EffectComposer(renderer, rt);
  composer.setPixelRatio(renderer.getPixelRatio());
  composer.setSize(w, h);
  composer.addPass(new RenderPass(scene, camera));
  let aoPass = null, bloomPass = null; const tiltPasses = [];
  if (ao) {
    aoPass = new GTAOPass(scene, camera, w, h);
    aoPass.output = GTAOPass.OUTPUT.Default;
    aoPass.blendIntensity = 0.85;
    aoPass.updateGtaoMaterial({ radius: 0.9, distanceExponent: 1.5, thickness: 1.2, scale: 1.1, samples: 12, distanceFallOff: 1, screenSpaceRadius: false });
    aoPass.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 6, rings: 2, samples: 12 });
    composer.addPass(aoPass);
  }
  if (tilt) {
    for (const dir of [[1, 0], [0, 1]]) {
      const p = new ShaderPass(TiltShiftShader);
      p.uniforms.uFocus.value = tilt.focus; p.uniforms.uRange.value = tilt.range; p.uniforms.uAmount.value = tilt.amount; p.uniforms.uTop.value = tilt.top ?? 0.3;
      p.uniforms.uRes.value.set(w * renderer.getPixelRatio(), h * renderer.getPixelRatio()); p.uniforms.uDir.value.set(...dir);
      composer.addPass(p); tiltPasses.push(p);
    }
  }
  if (bloom) {
    bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 0.22, 0.6, 0.9);
    composer.addPass(bloomPass);
  }
  composer.addPass(new OutputPass());
  if (grade) composer.addPass(new ShaderPass(GradeShader));
  return { composer, aoPass, bloomPass, tiltPasses };
}

// ---------------------------------------------------------------------------------------------
// Items (shop decorations)
// ---------------------------------------------------------------------------------------------
const ITEM_SLOTS = {
  flowers: [[-6.8, -2.2], [-5, -12.6], [4, -14.2], [13.2, -9.5], [-15.2, 1.2], [15.5, 0.5], [-10, 5.5], [2.5, 11.5], [-1, -15], [10, -13]],
  tulips: [[-14.4, 4.2], [-3.2, 9.8], [14.8, 4.5], [-9.5, -14.5], [7.5, -15.2], [18, -6]],
  fir: [[-18.5, -9], [18.5, -14], [-21, 6.5], [22, 3], [-13, -18.5], [11, -19], [-25, -3], [25, -6]],
  hay: [[-5.9, -9.6], [-19.2, -4.8], [-4.2, -11.2], [-20, -7.2], [-7.4, -11.8], [-17.8, -2]],
  wood: [[-17.6, -12.2], [-8.3, -15.5], [-19, -13.8]],
  sunflower: [[-8.3, -1.4], [-16.9, 3.5], [16.5, -3], [4.5, -15.7], [-2, 12.5], [11.5, -14.5]],
  apple: [[-20, 1], [20, -9], [-12, 12], [15, 11.5], [-27, -12], [27, -1]],
  tractor: [[-5.6, -4.4], [-18.5, 0.5], [17.5, 7.5]],
  balloon: [[-26, 20, -48], [30, 24, -60], [-6, 30, -90], [52, 22, -70]],
  kite: [[9, 16, -6], [-6, 18, -10], [17, 20, -12]],
  house: [[21, -17, -0.6], [-24, -15, 0.8], [26, 9, -1.4], [-27, 8, 1.2]],
  rainbow: [[60, -200], [-90, -220]],
};
const ITEM_OBSTACLE = { flowers: 0.8, tulips: 1.2, fir: 1.4, hay: 1.2, wood: 1.4, sunflower: 0.8, apple: 1.6, tractor: 2.2, house: 4, kite: 0.3 };

function tractorGroup() {
  const parts = [];
  parts.push(paint(place(new THREE.BoxGeometry(1.0, 0.7, 2.0), 0, 0.95, 0.2), '#d8261c'));  // body
  parts.push(paint(place(new THREE.BoxGeometry(0.8, 0.55, 1.1), 0, 1.1, 1.3), '#e02a1f')); // hood
  parts.push(paint(place(new THREE.BoxGeometry(0.82, 0.12, 0.2), 0, 0.95, 1.9), '#c8c8c8')); // grill
  parts.push(paint(place(new THREE.CylinderGeometry(0.06, 0.06, 0.8, 6), 0.25, 1.6, 1.25), '#2b2b2b')); // exhaust
  // cab frame
  for (const sx of [-0.45, 0.45]) for (const sz of [-0.55, 0.3]) parts.push(paint(place(new THREE.BoxGeometry(0.08, 1.2, 0.08), sx, 1.9, sz), '#f1f1f1'));
  parts.push(paint(place(new THREE.BoxGeometry(1.1, 0.1, 1.1), 0, 2.5, -0.12), '#f1f1f1'));
  parts.push(paint(place(new THREE.BoxGeometry(0.5, 0.15, 0.5), 0, 1.35, -0.3), '#333333')); // seat
  const wheel = (r, w, x, z) => {
    parts.push(paint(place(new THREE.CylinderGeometry(r, r, w, 14), x, r, z, 0, 0, Math.PI / 2), '#1f1f22'));
    parts.push(paint(place(new THREE.CylinderGeometry(r * 0.55, r * 0.55, w + 0.04, 10), x, r, z, 0, 0, Math.PI / 2), '#f3c21b'));
  };
  wheel(0.72, 0.38, -0.72, -0.35); wheel(0.72, 0.38, 0.72, -0.35); wheel(0.4, 0.24, -0.55, 1.4); wheel(0.4, 0.24, 0.55, 1.4);
  const m = new THREE.Mesh(merge(parts), toyMat(0xffffff, { vc: true, roughness: 0.45, clearcoat: 0.8 })); m.castShadow = true; m.receiveShadow = true;
  const g = new THREE.Group(); g.add(m); g.scale.setScalar(0.95); return g;
}
function hayGroup(rnd) {
  const parts = [];
  const bale = (x, y, z, ry) => {
    parts.push(paint(place(new THREE.CylinderGeometry(0.62, 0.62, 1.05, 28), x, y, z, 0, ry, Math.PI / 2), '#e2c066'));
    for (const s of [-1, 1]) parts.push(paint(place(new THREE.CylinderGeometry(0.4, 0.4, 0.02, 12), x + Math.cos(ry) * s * 0.53, y, z - Math.sin(ry) * s * 0.53, 0, ry, Math.PI / 2), '#c9a24a'));
  };
  bale(0, 0.62, 0, 0.2); bale(0.3, 0.62, 1.3, -0.1); bale(0.15, 1.72, 0.65, 0.05);
  const m = new THREE.Mesh(merge(parts), toyMat(0xffffff, { vc: true, roughness: 0.9, clearcoat: 0.1 })); m.castShadow = true; m.receiveShadow = true;
  const g = new THREE.Group(); g.add(m); return g;
}
function woodGroup(rnd) {
  const parts = [];
  let k = 0;
  for (let row = 0; row < 4; row++) for (let i = 0; i < 5 - row; i++) {
    const x = (i - (4 - row) / 2) * 0.34, y = 0.17 + row * 0.29;
    parts.push(paint(place(new THREE.CylinderGeometry(0.16, 0.16, 1.3, 14), x, y, 0, Math.PI / 2, 0, 0), '#7a5033', 0.15, rnd));
    for (const s of [-1, 1]) parts.push(paint(place(new THREE.CircleGeometry(0.15, 8), x, y, s * 0.651, 0, s < 0 ? Math.PI : 0, 0), '#dcb07a', 0.1, rnd));
    k++;
  }
  // chopping block + axe-free (kid friendly) basket
  parts.push(paint(place(new THREE.CylinderGeometry(0.3, 0.32, 0.5, 10), 1.3, 0.25, 0.3), '#8a6040'));
  const m = new THREE.Mesh(merge(parts), toyMat(0xffffff, { vc: true, roughness: 0.85, clearcoat: 0.1 })); m.castShadow = true; m.receiveShadow = true;
  const g = new THREE.Group(); g.add(m); return g;
}
function sunflowerGroup(rnd) {
  const parts = [];
  for (let i = 0; i < 3; i++) {
    const x = (i - 1) * 0.55 + (rnd() - 0.5) * 0.2, z = (rnd() - 0.5) * 0.4, h = 1.5 + rnd() * 0.6;
    parts.push(paint(place(new THREE.CylinderGeometry(0.035, 0.05, h, 5), x, h / 2, z), '#4f8a2a'));
    for (const s of [-1, 1]) parts.push(paint(place(new THREE.SphereGeometry(0.18, 12, 8), x + s * 0.15, h * 0.45, z, 0, 0, s * 0.6, 1, 0.25, 0.6), '#4f9a2e'));
    const head = new THREE.Group();
    parts.push(paint(place(new THREE.CylinderGeometry(0.36, 0.3, 0.06, 14), x, h, z + 0.05, 1.25, 0, 0), '#ffc814'));
    parts.push(paint(place(new THREE.ConeGeometry(0.44, 0.06, 14, 1, true), x, h - 0.01, z + 0.03, 1.25 - Math.PI, 0, 0), '#ffd21f'));
    parts.push(paint(place(new THREE.CylinderGeometry(0.2, 0.2, 0.08, 12), x, h + 0.02, z + 0.09, 1.25, 0, 0), '#6b3f1d'));
  }
  const m = new THREE.Mesh(merge(parts), toyMat(0xffffff, { vc: true, roughness: 0.6, side: THREE.DoubleSide })); m.castShadow = true;
  const g = new THREE.Group(); g.add(m); return g;
}
// A soft, rounded flowering shrub dotted with blossoms (the shop's "flowers").
function flowerBushGroup(rnd, blossoms = true) {
  const parts = [];
  const greens = ['#3f8f3a', '#4a9c40', '#367f34'];
  const blobs = [[0, 0.42, 0, 0.5], [0.42, 0.32, 0.1, 0.38], [-0.38, 0.3, -0.05, 0.4], [0.05, 0.3, 0.38, 0.36], [-0.1, 0.28, -0.4, 0.34]];
  blobs.forEach(([x, y, z, r], i) => parts.push(paint(place(new THREE.IcosahedronGeometry(r, 3), x, y, z, 0, 0, 0, 1, 0.85, 1), greens[i % 3])));
  const palettes = [['#ff6fa8', '#ffd1e3'], ['#ffd21a', '#ffffff'], ['#b25ae0', '#f3e6ff'], ['#ff8a5c', '#ffe0a0']];
  const pal = palettes[Math.floor(rnd() * palettes.length)];
  for (let i = 0; i < (blossoms ? 26 : 0); i++) {
    const [bx, by, bz, br] = blobs[i % blobs.length];
    const a = rnd() * 6.28, e = rnd() * 1.2 + 0.1;
    const nx = Math.cos(a) * Math.cos(e), ny = Math.sin(e), nz = Math.sin(a) * Math.cos(e);
    const x = bx + nx * br * 0.98, y = by + ny * br * 0.85, z = bz + nz * br * 0.98;
    parts.push(paint(place(new THREE.SphereGeometry(0.075, 10, 6, 0, Math.PI * 2, 0, Math.PI * 0.5), x, y, z, 0, 0, 0, 1, 0.45, 1), pal[0]));
    parts.push(paint(place(new THREE.SphereGeometry(0.03, 8, 6), x, y + 0.03, z), pal[1]));
  }
  const m = new THREE.Mesh(merge(parts), toyMat(0xffffff, { vc: true, roughness: 0.65, clearcoat: 0.2 })); m.castShadow = true; m.receiveShadow = true;
  const g = new THREE.Group(); g.add(m); g.scale.setScalar(1.1); return g;
}
function tulipGroup(rnd) {
  const parts = [];
  parts.push(paint(place(new THREE.BoxGeometry(2.4, 0.16, 1.2), 0, 0.02, 0), '#6a4a32'));
  const cols = ['#e8232c', '#ff6fa8', '#ffd21a', '#ff8a1a', '#b25ae0', '#ffffff'];
  for (let i = 0; i < 14; i++) {
    const x = ((i % 7) - 3) * 0.32 + (rnd() - 0.5) * 0.06, z = (Math.floor(i / 7) - 0.5) * 0.5, h = 0.4 + rnd() * 0.12;
    parts.push(paint(place(new THREE.CylinderGeometry(0.012, 0.015, h, 4), x, h / 2 + 0.08, z), '#3f8a2a'));
    parts.push(paint(place(new THREE.SphereGeometry(0.07, 12, 8, 0, Math.PI * 2, 0, Math.PI * 0.6), x, h + 0.13, z, Math.PI, 0, 0, 1, 1.5, 1), cols[(i * 7 + Math.floor(rnd() * 3)) % cols.length]));
    parts.push(paint(place(new THREE.SphereGeometry(0.06, 4, 3), x + 0.04, 0.2, z, 0, rnd() * 3, 0.3, 0.4, 2, 0.8), '#4c9a32'));
  }
  const m = new THREE.Mesh(merge(parts), toyMat(0xffffff, { vc: true, roughness: 0.5, side: THREE.DoubleSide })); m.castShadow = true; m.receiveShadow = true;
  const g = new THREE.Group(); g.add(m); return g;
}
function appleTreeGroup(rnd) {
  const parts = [];
  parts.push(paint(place(new THREE.CylinderGeometry(0.16, 0.24, 1.8, 12), 0, 0.9, 0), '#6d4a30'));
  for (let i = 0; i < 5; i++) {
    const a = i * 1.3, r = i ? 0.75 : 0, y = i ? 2.2 + rnd() * 0.5 : 2.8;
    parts.push(paint(place(new THREE.IcosahedronGeometry(i ? 0.95 : 1.2, 4), Math.cos(a) * r, y, Math.sin(a) * r), i % 2 ? '#4f9e3a' : '#5aad40'));
  }
  for (let i = 0; i < 16; i++) {
    const a = rnd() * 6.28, e = rnd() * 1.2 - 0.2, R = 1.45;
    parts.push(paint(place(new THREE.IcosahedronGeometry(0.12, 2), Math.cos(a) * Math.cos(e) * R, 2.5 + Math.sin(e) * R, Math.sin(a) * Math.cos(e) * R), '#e3262b'));
  }
  const m = new THREE.Mesh(merge(parts), toyMat(0xffffff, { vc: true, roughness: 0.6 })); m.castShadow = true; m.receiveShadow = true;
  const g = new THREE.Group(); g.add(m); return g;
}
function balloonGroup(rnd, idx) {
  const pts = [];
  for (let i = 0; i <= 28; i++) { const t = i / 28; const a = t * Math.PI; pts.push(new THREE.Vector2(Math.sin(a) * (1.9 - 0.9 * (1 - t) * (1 - t)) * (t < 0.15 ? 0.6 + t * 2.6 : 1), -Math.cos(a) * 2.2)); }
  const lathe = new THREE.LatheGeometry(pts, 48); lathe.computeVertexNormals(); const geo = lathe.toNonIndexed(); lathe.dispose();
  const palettes = [['#e8232c', '#ffd21a'], ['#2f7de0', '#ffffff'], ['#ff6fa8', '#7c4ce0'], ['#2eb85c', '#ffd21a']];
  const pal = palettes[idx % palettes.length];
  const pos = geo.attributes.position; const cols = new Float32Array(pos.count * 3); const c = new THREE.Color();
  for (let i = 0; i < pos.count; i += 3) {
    const cx = (pos.getX(i) + pos.getX(i + 1) + pos.getX(i + 2)) / 3, cz = (pos.getZ(i) + pos.getZ(i + 1) + pos.getZ(i + 2)) / 3;
    const seg = Math.floor(((Math.atan2(cz, cx) + Math.PI) / (Math.PI * 2)) * 16 + 1e-3);
    c.set(pal[seg % 2]); for (let j = 0; j < 3; j++) { cols[(i + j) * 3] = c.r; cols[(i + j) * 3 + 1] = c.g; cols[(i + j) * 3 + 2] = c.b; }
  }
  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3)); geo.deleteAttribute('uv'); geo.translate(0, 2.2, 0);
  const parts = [geo];
  parts.push(paint(place(new THREE.BoxGeometry(0.7, 0.55, 0.7), 0, -1.2, 0), '#8a5a30'));
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) parts.push(paint(place(new THREE.CylinderGeometry(0.015, 0.015, 1.3, 3), sx * 0.3, -0.35, sz * 0.3, sz * 0.25, 0, -sx * 0.25), '#5a4030'));
  const m = new THREE.Mesh(merge(parts), toyMat(0xffffff, { vc: true, roughness: 0.5, clearcoat: 0.7, side: THREE.DoubleSide }));
  const g = new THREE.Group(); g.add(m); g.scale.setScalar(2.2); return g;
}
function kiteGroup() {
  const parts = [];
  const tri = (a, b, c, col) => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute([...a, ...b, ...c], 3)); g.computeVertexNormals(); return paint(g, col); };
  const T = [0, 0.9, 0], R = [0.55, 0.15, 0], Bm = [0, -0.9, 0], L = [-0.55, 0.15, 0], Cc = [0, 0.15, 0];
  parts.push(tri(T, L, Cc, '#e8232c'), tri(T, Cc, R, '#ffd21a'), tri(Cc, L, Bm, '#2f7de0'), tri(Cc, Bm, R, '#2eb85c'));
  const kite = new THREE.Mesh(merge(parts), toyMat(0xffffff, { vc: true, side: THREE.DoubleSide, roughness: 0.5 }));
  const g = new THREE.Group(); g.add(kite); kite.scale.setScalar(1.3);
  // tail bows
  const bows = [];
  for (let i = 0; i < 5; i++) { const b = new THREE.Mesh(new THREE.OctahedronGeometry(0.12, 0), toyMat(['#ff6fa8', '#ffd21a', '#2f7de0', '#e8232c', '#2eb85c'][i])); b.scale.set(1.4, 0.6, 0.4); g.add(b); bows.push(b); }
  g.userData.bows = bows; g.userData.kite = kite;
  return g;
}
function rainbowMesh() {
  const geo = new THREE.RingGeometry(95, 118, 96, 1, 0, Math.PI);
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, fog: false,
    uniforms: { uIn: { value: 95 }, uOut: { value: 118 } },
    vertexShader: 'varying vec2 vP; void main(){ vP = position.xy; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
    fragmentShader: `uniform float uIn; uniform float uOut; varying vec2 vP;
vec3 hue(float h){ return clamp(abs(mod(h*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0); }
void main(){ float t = (length(vP)-uIn)/(uOut-uIn); vec3 c = hue(0.8*(1.0-t)); float a = smoothstep(0.0,0.12,t)*smoothstep(1.0,0.88,t)*0.42*smoothstep(0.0, 60.0, vP.y);
 gl_FragColor = vec4(c, a); }`,
  });
  const m = new THREE.Mesh(geo, mat); m.renderOrder = 1; return m;
}

async function buildItems(items, rnd, obstacles, uniforms) {
  const nature = await natureParts().catch(() => ({}));
  const group = new THREE.Group();
  const counts = {};
  const animated = [];
  for (const id of items || []) {
    const k = String(id).toLowerCase();
    const slots = ITEM_SLOTS[k]; if (!slots) continue;
    const n = counts[k] = (counts[k] || 0) + 1; if (n > slots.length) continue;
    const s = slots[n - 1];
    let obj = null;
    const x = s[0], z = k === 'balloon' || k === 'kite' ? s[2] : s[1];
    const y = heightAt(x, z);
    const faceCam = Math.atan2(0 - x, 30 - z);
    switch (k) {
      case 'flowers': obj = flowerBushGroup(rnd); obj.rotation.y = rnd() * 6; break;
      case 'tulips': obj = tulipGroup(rnd); obj.rotation.y = faceCam; break;
      case 'fir': {
        const geo = spruceGeometry(1, rnd); obj = new THREE.Mesh(geo, vcMat({ roughness: 0.85 })); obj.scale.set(6.5, 8 + rnd() * 2, 6.5); obj.castShadow = true; obj.receiveShadow = true; break;
      }
      case 'hay': obj = hayGroup(rnd); obj.rotation.y = rnd() * 3; break;
      case 'wood': obj = woodGroup(rnd); obj.rotation.y = LAYOUT.barn.rot + Math.PI / 2; break;
      case 'sunflower': obj = sunflowerGroup(rnd); obj.rotation.y = faceCam; break;
      case 'apple': obj = appleTreeGroup(rnd); obj.scale.setScalar(1.2); obj.rotation.y = rnd() * 6; break;
      case 'tractor': obj = tractorGroup(); obj.rotation.y = faceCam + 0.9; break;
      case 'house': obj = buildCottage(0.85); obj.rotation.y = s[2]; break;
      case 'balloon': {
        const bl = balloonGroup(rnd, n - 1); bl.position.set(s[0], s[1], s[2]);
        const base = bl.position.clone(); const ph = rnd() * 6;
        animated.push((t) => { bl.position.set(base.x + Math.sin(t * 0.05 + ph) * 6, base.y + Math.sin(t * 0.4 + ph) * 1.2, base.z); bl.rotation.y = t * 0.05; });
        group.add(bl); break;
      }
      case 'kite': {
        const kg = kiteGroup(); const anchor = new THREE.Vector3(s[0], heightAt(s[0], s[2]) + 0.3, s[2]);
        const top = new THREE.Vector3(s[0] + 4, s[1], s[2] - 8);
        const lineGeo = new THREE.BufferGeometry().setFromPoints(new Array(12).fill(0).map(() => new THREE.Vector3()));
        const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: 0xffffff }));
        group.add(kg, line);
        const ph = rnd() * 6;
        const upd = (t) => {
          const kp = top.clone().add(new THREE.Vector3(Math.sin(t * 0.7 + ph) * 1.5, Math.sin(t * 1.1 + ph) * 0.8, 0));
          kg.position.copy(kp); kg.userData.kite.rotation.set(-0.35, 0.2 + Math.sin(t * 0.9) * 0.2, Math.sin(t * 1.3 + ph) * 0.25);
          const p = lineGeo.attributes.position;
          for (let i = 0; i < 12; i++) { const f = i / 11; const q = new THREE.Vector3().lerpVectors(anchor, kp, f); q.y -= Math.sin(f * Math.PI) * 1.2; p.setXYZ(i, q.x, q.y, q.z); }
          p.needsUpdate = true;
          kg.userData.bows.forEach((b, i) => { b.position.set(Math.sin(t * 3 + i) * 0.25 * (i + 1) * 0.3, -1.3 - i * 0.45, 0.1 * i); b.rotation.z = t * 2 + i; });
        };
        upd(0); animated.push(upd);
        obstacles.push({ x: s[0], z: s[2], r: 0.4 });
        break;
      }
      case 'rainbow': obj = rainbowMesh(); obj.position.set(s[0], -25, s[1]); obj.lookAt(0, -25, 40); group.add(obj); obj = null; break;
    }
    if (obj) {
      obj.position.set(x, y - 0.02, z);
      group.add(obj);
    }
    if (ITEM_OBSTACLE[k] && k !== 'kite') obstacles.push({ x, z, r: ITEM_OBSTACLE[k] });
  }
  group.userData.animated = animated;
  return group;
}

// ---------------------------------------------------------------------------------------------
// Animals
// ---------------------------------------------------------------------------------------------
const ZONES = {
  pasture: { x: 1.5, z: -3.5, rx: 11, rz: 6 },
  yard: { x: -4.5, z: 2.5, rx: 5, rz: 3.8 },
  edge: { x: -1, z: -9, rx: 12, rz: 3 },
};
function clipName(n) { const p = n.split('|'); return p[p.length - 1]; }
async function loadAnimal(def) {
  if (def.toy && !(typeof window !== 'undefined' && window.W3D_GLB_ANIMALS)) {
    try { return await loadToyAnimal(def); } catch (e) { console.warn('World3D: toy animal failed, using the model file', def.toy, e); }
  }
  const g = await loadGLTF('animals/' + def.file + '.glb');
  if (!g.userData._prepared) {
    g.userData._prepared = true;
    // measure bind-pose height
    g.scene.updateMatrixWorld(true);
    const box = new THREE.Box3();
    g.scene.traverse((o) => { if (o.isMesh) box.expandByObject(o, true); });
    g.userData.box = box;
    g.scene.traverse((o) => {
      if (o.isMesh) {
        o.castShadow = true; o.receiveShadow = true; o.frustumCulled = false;
        // The models ship flat-shaded (faceted). Weld vertices that share position + uv + skinning and
        // recompute normals: bodies round off, while edges between different colours stay crisp.
        smoothNormals(o.geometry);
        for (const m of [].concat(o.material)) { m.roughness = Math.min(m.roughness ?? 1, 0.75); m.metalness = 0; if (m.map) m.map.anisotropy = 4; }
      }
    });
  }
  const model = skeletonClone(g.scene);
  const box = g.userData.box;
  const s = def.h * ANIMAL_SCALE / (box.max.y - box.min.y);
  model.scale.setScalar(s);
  model.position.y = -box.min.y * s;
  const clips = {};
  for (const c of g.animations) clips[clipName(c.name)] = c;
  return { model, clips, size: new THREE.Vector3().subVectors(box.max, box.min).multiplyScalar(s) };
}

// Smooth normals for (possibly quantized/interleaved) glTF geometry, without re-indexing: area-weighted
// face normals are summed per (position, uv) key, so a body rounds off but colour seams stay crisp.
function smoothNormals(geo) {
  const pos = geo.attributes.position, uv = geo.attributes.uv, n = pos.count;
  const index = geo.index ? geo.index.array : null; const triCount = index ? index.length / 3 : n / 3;
  const box = new THREE.Box3().setFromBufferAttribute(pos); const q = 2e4 / Math.max(1e-6, box.getSize(new THREE.Vector3()).length());
  const keyOf = new Int32Array(n), keys = new Map();
  for (let i = 0; i < n; i++) {
    let k = Math.round(pos.getX(i) * q) + ',' + Math.round(pos.getY(i) * q) + ',' + Math.round(pos.getZ(i) * q);
    if (uv) k += '|' + Math.round(uv.getX(i) * 512) + ',' + Math.round(uv.getY(i) * 512);
    let id = keys.get(k); if (id === undefined) { id = keys.size; keys.set(k, id); } keyOf[i] = id;
  }
  const acc = new Float32Array(keys.size * 3);
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  for (let t = 0; t < triCount; t++) {
    const i0 = index ? index[t * 3] : t * 3, i1 = index ? index[t * 3 + 1] : t * 3 + 1, i2 = index ? index[t * 3 + 2] : t * 3 + 2;
    a.fromBufferAttribute(pos, i0); b.fromBufferAttribute(pos, i1); c.fromBufferAttribute(pos, i2);
    c.sub(b); a.sub(b); c.cross(a); // un-normalised = area weighted
    for (const i of [i0, i1, i2]) { const k = keyOf[i] * 3; acc[k] += c.x; acc[k + 1] += c.y; acc[k + 2] += c.z; }
  }
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const k = keyOf[i] * 3; a.set(acc[k], acc[k + 1], acc[k + 2]).normalize();
    out[i * 3] = a.x; out[i * 3 + 1] = a.y; out[i * 3 + 2] = a.z;
  }
  geo.setAttribute('normal', new THREE.BufferAttribute(out, 3));
}

// Procedural toy animal (toyanimals.js), scaled to the roster height.
async function loadToyAnimal(def) {
  const H = def.h * ANIMAL_SCALE;
  const { model, clips, box } = await buildToyAnimal(def.toy, { height: H, speed: 0.5 + def.h * 0.45 });
  const s = H / (box.max.y - box.min.y);
  model.scale.setScalar(s); model.position.y = -box.min.y * s;
  model.traverse((o) => { if (o.isMesh) o.frustumCulled = false; });
  return { model, clips, size: new THREE.Vector3().subVectors(box.max, box.min).multiplyScalar(s) };
}

class Animal {
  constructor(def, loaded, farm) {
    this.def = def; this.farm = farm;
    this.root = new THREE.Group();
    this.root.add(loaded.model);
    this.model = loaded.model;
    this.size = loaded.size;
    this.radius = Math.max(this.size.x, this.size.z) * 0.42;
    this.mixer = new THREE.AnimationMixer(loaded.model);
    this.actions = {};
    for (const [n, c] of Object.entries(loaded.clips)) {
      if (/attack|death/i.test(n)) continue;
      this.actions[n] = this.mixer.clipAction(c);
    }
    this.find = (...names) => names.map((n) => this.actions[n]).filter(Boolean);
    this.idleClips = this.find('Idle', 'Idle_2', 'Idle_Headlow', 'Idle_2_HeadLow');
    this.eatClip = this.find('Eating')[0] || null;
    this.walkClip = this.find('Walk', 'WalkSlow')[0] || null;
    this.runClip = this.find('Gallop', 'Run')[0] || this.walkClip;
    this.jumpClip = this.find('Gallop_Jump', 'Jump_toIdle', 'Jump_ToIdle', 'Jump')[0] || null;
    this.reactClips = this.find('Idle_HitReact_Left', 'Idle_HitReact_Right');
    this.hopper = !this.walkClip; // A_ animals without a walk cycle hop instead
    this.current = null;
    this.state = 'idle'; this.timer = 1 + Math.random() * 3;
    this.pos = new THREE.Vector2(); this.yaw = Math.random() * 6.28; this.target = null; this.speed = 0;
    this.baseSpeed = 0.5 + def.h * 0.45;
    this.hopT = 0; this.happy = 0; this.oneShot = null;
    // tap proxy
    const proxy = new THREE.Mesh(new THREE.BoxGeometry(this.size.x * 1.5 + 0.4, this.size.y * 1.2 + 0.3, this.size.z * 1.15 + 0.3), new THREE.MeshBasicMaterial({ visible: false }));
    proxy.position.y = this.size.y * 0.55; proxy.userData.animal = this;
    this.root.add(proxy); this.proxy = proxy;
    // blob contact shadow
    const blob = new THREE.Mesh(farm.blobGeo, farm.blobMat); blob.scale.set(this.size.x * 1.3 + 0.3, 1, this.size.z * 0.9 + 0.3); blob.position.y = 0.03; blob.renderOrder = 1;
    this.root.add(blob);
    this.mixer.addEventListener('finished', (e) => { if (e.action === this.oneShot) { this.oneShot = null; this.play(this.pickIdle(), 0.3); } });
  }
  pickIdle() { return this.idleClips[Math.floor(Math.random() * this.idleClips.length)] || null; }
  play(action, fade = 0.35, timeScale = 1) {
    if (!action) return;
    if (this.current === action) { action.timeScale = timeScale; return; }
    action.reset(); action.setLoop(THREE.LoopRepeat, Infinity); action.timeScale = timeScale; action.enabled = true; action.setEffectiveWeight(1);
    action.play();
    if (this.current) this.current.crossFadeTo(action, fade, false);
    this.current = action;
  }
  playOnce(action, fade = 0.2, timeScale = 1) {
    if (!action) return false;
    action.reset(); action.setLoop(THREE.LoopOnce, 1); action.clampWhenFinished = true; action.timeScale = timeScale; action.enabled = true; action.setEffectiveWeight(1); action.play();
    if (this.current && this.current !== action) this.current.crossFadeTo(action, fade, false);
    this.current = action; this.oneShot = action; return true;
  }
  setPos(x, z) { this.pos.set(x, z); this.root.position.set(x, heightAt(x, z), z); }
  chooseTarget() {
    const Z = ZONES[this.def.zone] || ZONES.pasture;
    for (let i = 0; i < 30; i++) {
      const a = Math.random() * 6.28, r = Math.sqrt(Math.random());
      const x = Z.x + Math.cos(a) * Z.rx * r, z = Z.z + Math.sin(a) * Z.rz * r;
      if (this.farm.blocked(x, z, this.radius + 0.4, this)) continue;
      if (Math.hypot(x - this.pos.x, z - this.pos.y) < 2.5) continue;
      return new THREE.Vector2(x, z);
    }
    return null;
  }
  react() {
    this.happy = 1;
    if (this.state === 'walk' || this.state === 'arrive') { /* keep going, just wiggle */ return; }
    const r = Math.random();
    if (this.jumpClip && (r < 0.6 || !this.reactClips.length)) this.playOnce(this.jumpClip, 0.15, 1);
    else if (this.reactClips.length) this.playOnce(this.reactClips[Math.floor(Math.random() * this.reactClips.length)], 0.15, 1);
    this.state = 'idle'; this.timer = 3;
  }
  update(dt, t) {
    this.mixer.update(dt);
    const calm = REDUCED_MOTION ? 3 : 1;
    if (this.oneShot) { this.applyTransform(dt, t, 0); return; }
    if (this.state === 'idle' || this.state === 'eat') {
      this.timer -= dt;
      if (this.timer <= 0) {
        const r = Math.random();
        if (r < 0.45 / calm && !this.farm.arrivingAnimal) {
          const tg = this.chooseTarget();
          if (tg) { this.target = tg; this.state = 'walk'; this.timer = 14; }
        } else if (r < 0.75 && this.eatClip) { this.state = 'eat'; this.play(this.eatClip, 0.5); this.timer = 4 + Math.random() * 5; }
        else { this.state = 'idle'; this.play(this.pickIdle(), 0.5); this.timer = (3 + Math.random() * 5) * calm; }
      }
      if (this.state === 'walk') this.play(this.hopper ? this.pickIdle() : this.walkClip, 0.35, 1);
      this.speed = Math.max(0, this.speed - dt * 3);
    }
    if (this.state === 'walk' || this.state === 'arrive') {
      const run = this.state === 'arrive' && this.arrivePhase === 'run';
      const tgt = this.target; const dx = tgt.x - this.pos.x, dz = tgt.y - this.pos.y; const dist = Math.hypot(dx, dz);
      let want = Math.atan2(dx, dz);
      // obstacle + neighbour avoidance
      const fx = Math.sin(this.yaw), fz = Math.cos(this.yaw);
      let ax = 0, az = 0;
      for (const o of this.farm.obstaclesFor(this)) {
        const ox = o.x - this.pos.x, oz = o.z - this.pos.y; const d = Math.hypot(ox, oz); const rr = o.r + this.radius + 0.6;
        if (d < rr * 1.8 && d > 0.01 && ox * fx + oz * fz > -0.3) { const k = (rr * 1.8 - d) / (rr * 1.8); ax -= (ox / d) * k * 2.2; az -= (oz / d) * k * 2.2; }
      }
      if (ax || az) { const dxn = dx / (dist || 1) + ax, dzn = dz / (dist || 1) + az; want = Math.atan2(dxn, dzn); }
      const turnRate = run ? 3.2 : 2.2;
      let d = want - this.yaw; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
      this.yaw += clamp(d, -turnRate * dt, turnRate * dt);
      const targetSpeed = (run ? this.baseSpeed * 3.2 : this.baseSpeed) * (Math.abs(d) > 1.4 ? 0.3 : 1) * (dist < 1 ? Math.max(0.3, dist) : 1);
      this.speed = lerp(this.speed, targetSpeed, 1 - Math.exp(-dt * 4));
      const nx = this.pos.x + Math.sin(this.yaw) * this.speed * dt, nz = this.pos.y + Math.cos(this.yaw) * this.speed * dt;
      if (this.state === 'arrive' || !this.farm.blockedHard(nx, nz, this)) this.setPos(nx, nz);
      else this.yaw += dt * 2;
      if (!this.hopper && this.current && (this.current === this.walkClip || this.current === this.runClip)) {
        const nominal = run ? this.baseSpeed * 3.2 : this.baseSpeed;
        this.current.timeScale = clamp(this.speed / nominal, 0.35, 1.3);
      }
      this.timer -= dt;
      if (dist < 0.5 || (this.state === 'walk' && this.timer <= 0)) {
        if (this.state === 'arrive') { this.farm.onArrived(this); }
        this.state = 'idle'; this.target = null; this.timer = (2 + Math.random() * 4) * calm;
        if (this.eatClip && Math.random() < 0.5) { this.state = 'eat'; this.play(this.eatClip, 0.5); } else this.play(this.pickIdle(), 0.5);
      }
    }
    this.applyTransform(dt, t, this.speed);
  }
  applyTransform(dt, t, speed) {
    this.root.rotation.y = this.yaw;
    let bob = 0, roll = 0;
    if (this.hopper && speed > 0.05) {
      this.hopT += dt * (6 + speed * 3);
      bob = Math.abs(Math.sin(this.hopT)) * 0.12 * this.size.y * Math.min(1, speed / this.baseSpeed);
      roll = Math.sin(this.hopT) * 0.06;
    }
    if (this.happy > 0) {
      this.happy = Math.max(0, this.happy - dt * 1.2);
      const h = this.happy; bob += Math.abs(Math.sin(h * 14)) * 0.25 * h * this.size.y; roll += Math.sin(h * 22) * 0.12 * h;
    }
    this.model.rotation.z = roll;
    this.root.position.y = heightAt(this.pos.x, this.pos.y) + bob;
  }
  dispose() { this.mixer.stopAllAction(); this.mixer.uncacheRoot(this.model); this.proxy.geometry.dispose(); this.proxy.material.dispose(); }
}

// ---------------------------------------------------------------------------------------------
// Farm (interactive)
// ---------------------------------------------------------------------------------------------
function farmFov(aspect) { // keep ~52deg horizontal view; portrait gets a taller view
  return clamp(2 * Math.atan(Math.tan((52 * Math.PI) / 360) / aspect) * 180 / Math.PI, 40, 75);
}
const CAM = { azMin: -0.75, azMax: 0.75, elMin: 0.2, elMax: 0.78, distMin: 9, distMax: 44 };
function pickQuality() {
  const small = Math.min(screen.width, screen.height) < 700;
  const touch = navigator.maxTouchPoints > 0;
  return small && touch ? 'medium' : 'high';
}

export function mountFarm(container, opts = {}) {
  let destroyed = false;
  let quality = opts.quality || pickQuality();
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'display:block;width:100%;height:100%;touch-action:none;outline:none;';
  container.appendChild(canvas);
  const size = () => ({ w: Math.max(1, container.clientWidth), h: Math.max(1, container.clientHeight) });
  let { w, h } = size();
  const pr = () => Math.min(window.devicePixelRatio || 1, quality === 'low' ? 1 : quality === 'medium' ? 1.5 : 2);
  const renderer = makeRenderer(canvas, { width: w, height: h, pixelRatio: pr() });
  renderer.info.autoReset = false;
  const camera = new THREE.PerspectiveCamera(farmFov(w / h), w / h, 0.5, 5000);

  const state = {
    animals: new Map(), items: null, itemsKey: '', obstacles: [], itemObstacles: [],
    onAnimalTap: opts.onAnimalTap || null, world: null, post: null, env: null, running: true,
    cam: { az: 0.1, el: 0.36, dist: 27, target: new THREE.Vector3(0, 1, -1) },
    camGoal: { az: 0.1, el: 0.36, dist: 27, target: new THREE.Vector3(0, 1, -1) },
    focus: null, focusUntil: 0, frame: 0, fpsSamples: [], lastT: performance.now() / 1000, t: 0,
    pending: Promise.resolve(),
  };
  const farm = {
    blobGeo: (() => { const g = new THREE.PlaneGeometry(1, 1); g.rotateX(-Math.PI / 2); return g; })(),
    blobMat: (() => {
      const c = document.createElement('canvas'); c.width = c.height = 64; const x = c.getContext('2d');
      const gr = x.createRadialGradient(32, 32, 0, 32, 32, 32); gr.addColorStop(0, 'rgba(20,40,10,0.55)'); gr.addColorStop(1, 'rgba(20,40,10,0)');
      x.fillStyle = gr; x.fillRect(0, 0, 64, 64);
      const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
      return new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2 });
    })(),
    arrivingAnimal: null,
    blocked(x, z, r, self) {
      const M = LAYOUT.meadow; if (((x - M.x) / M.rx) ** 2 + ((z - M.z) / M.rz) ** 2 > 1) return true;
      if (blockedStatic(x, z, r * 0.5)) return true;
      for (const o of state.itemObstacles) if (Math.hypot(x - o.x, z - o.z) < o.r + r) return true;
      for (const a of state.animals.values()) {
        if (a === self) continue;
        if (Math.hypot(x - a.pos.x, z - a.pos.y) < a.radius + r + 0.5) return true;
        if (a.target && Math.hypot(x - a.target.x, z - a.target.y) < a.radius + r + 0.5) return true;
      }
      return false;
    },
    blockedHard(x, z, self) {
      if (blockedStatic(x, z, -0.3)) return true;
      for (const o of state.itemObstacles) if (Math.hypot(x - o.x, z - o.z) < o.r * 0.8) return true;
      for (const a of state.animals.values()) { if (a === self) continue; const d = Math.hypot(x - a.pos.x, z - a.pos.y), d0 = Math.hypot(self.pos.x - a.pos.x, self.pos.y - a.pos.y); if (d < (a.radius + self.radius) * 0.8 && d < d0) return true; }
      return false;
    },
    obstaclesFor(self) {
      const list = [...state.obstacles, ...state.itemObstacles];
      for (const a of state.animals.values()) if (a !== self) list.push({ x: a.pos.x, z: a.pos.y, r: a.radius });
      return list;
    },
    onArrived(a) {
      farm.arrivingAnimal = null;
      if (a.jumpClip) a.playOnce(a.jumpClip, 0.2);
      a.happy = 1;
      state.focusUntil = state.t + 3.5;
    },
  };
  // static obstacles for steering (barn as three circles, pond)
  {
    const B = LAYOUT.barn; const c = Math.cos(B.rot), s = Math.sin(B.rot);
    for (const lz of [-3.5, 0, 3.5]) state.obstacles.push({ x: B.x + lz * s, z: B.z + lz * c, r: B.w / 2 + 0.6 });
    state.obstacles.push({ x: LAYOUT.pond.x, z: LAYOUT.pond.z, r: LAYOUT.pond.r + 0.5 });
  }

  const ready = (async () => {
    const world = await buildWorld({ quality });
    if (destroyed) return;
    state.world = world;
    state.env = makeEnvironment(renderer);
    world.scene.environment = state.env.texture;
    world.scene.environmentIntensity = 0.45;
    setupPost();
    await applyOpts(opts, true);
    renderer.shadowMap.needsUpdate = true;
    loop();
  })();
  ready.catch((e) => console.error('World3D mountFarm failed', e));

  function setupPost() {
    if (state.post) { state.post.composer.dispose(); state.post.aoPass && state.post.aoPass.dispose(); state.post.bloomPass && state.post.bloomPass.dispose(); }
    state.post = quality === 'low' ? null : makeComposer(renderer, state.world.scene, camera, w, h, { ao: quality === 'high', bloom: true });
    renderer.shadowMap.type = quality === 'low' ? THREE.PCFShadowMap : THREE.PCFSoftShadowMap;
  }

  async function applyOpts(o, first = false) {
    if (!state.world) return;
    if ('onAnimalTap' in o) state.onAnimalTap = o.onAnimalTap;
    const scene = state.world.scene;
    // items
    if (o.items) {
      const key = JSON.stringify(o.items);
      if (key !== state.itemsKey) {
        state.itemsKey = key;
        const obst = [];
        const g = await buildItems(o.items, mulberry32(4321), obst, state.world.uniforms);
        if (destroyed) return;
        if (state.items) { scene.remove(state.items); disposeTree(state.items); }
        state.items = g; state.itemObstacles = obst; scene.add(g);
        renderer.shadowMap.needsUpdate = true;
      }
    }
    // animals
    if (o.animals) {
      const want = new Set(o.animals.filter((i) => WORLD_ANIMALS[i]));
      for (const [i, a] of state.animals) if (!want.has(i)) { scene.remove(a.root); a.dispose(); state.animals.delete(i); }
      const arriving = typeof o.arriving === 'number' ? o.arriving : null;
      const loads = [...want].filter((i) => !state.animals.has(i)).map(async (i) => {
        const def = WORLD_ANIMALS[i];
        try {
          const loaded = await loadAnimal(def);
          if (destroyed || state.animals.has(i)) return;
          // compile its shaders off the critical path so the first frame with it doesn't hitch
          try { await renderer.compileAsync(loaded.model, camera, scene); } catch (e) { /* older browsers: compile on first draw */ }
          if (destroyed || state.animals.has(i)) return;
          const a = new Animal(def, loaded, farm);
          state.animals.set(i, a);
          scene.add(a.root);
          if (i === arriving) startArrival(a);
          else {
            let p = null;
            for (let k = 0; k < 40 && !p; k++) { const t = a.chooseTarget(); if (t) p = t; }
            p = p || new THREE.Vector2(ZONES.pasture.x, ZONES.pasture.z);
            a.setPos(p.x, p.y); a.yaw = Math.random() * 6.28 - 3.14;
            a.play(a.pickIdle(), 0);
            a.mixer.setTime(Math.random() * 3);
          }
        } catch (e) { console.warn('World3D: could not load animal', def.file, e); }
      });
      await Promise.all(loads);
      if (arriving !== null && state.animals.has(arriving) && !want.has(arriving)) { /* noop */ }
    } else if (typeof o.arriving === 'number' && state.animals.has(o.arriving)) {
      startArrival(state.animals.get(o.arriving));
    }
  }

  function startArrival(a) {
    const E = LAYOUT.entry;
    a.setPos(E.x, E.z); a.yaw = -Math.PI / 2 - 0.25;
    let tg = null;
    for (let k = 0; k < 40 && !tg; k++) { const x = 2 + (Math.random() - 0.5) * 8, z = 2 + (Math.random() - 0.5) * 4; if (!farm.blocked(x, z, a.radius + 0.3, a)) tg = new THREE.Vector2(x, z); }
    a.target = tg || new THREE.Vector2(3, 1);
    a.state = 'arrive'; a.arrivePhase = 'run'; a.speed = a.baseSpeed * 2;
    a.play(a.hopper ? a.pickIdle() : a.runClip, 0.1);
    farm.arrivingAnimal = a;
    state.focus = a; state.focusUntil = Infinity;
  }

  // ---- input ----
  const pointers = new Map(); let downInfo = null; let pinch0 = null;
  const onDown = (e) => {
    canvas.setPointerCapture?.(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 1) downInfo = { x: e.clientX, y: e.clientY, t: performance.now(), moved: false };
    else { downInfo = null; const [p, q] = [...pointers.values()]; pinch0 = { d: Math.hypot(p.x - q.x, p.y - q.y), dist: state.camGoal.dist }; }
  };
  const onMove = (e) => {
    const p = pointers.get(e.pointerId); if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY;
    if (pointers.size === 1) {
      if (downInfo && Math.hypot(e.clientX - downInfo.x, e.clientY - downInfo.y) > 8) downInfo.moved = true;
      if (downInfo && !downInfo.moved) return;
      state.camGoal.az = clamp(state.camGoal.az - dx * 0.0045, CAM.azMin, CAM.azMax);
      state.camGoal.el = clamp(state.camGoal.el + dy * 0.0035, CAM.elMin, CAM.elMax);
      if (state.focus && state.focusUntil !== Infinity) state.focusUntil = 0;
    } else if (pointers.size === 2 && pinch0) {
      const [a, b] = [...pointers.values()]; const d = Math.hypot(a.x - b.x, a.y - b.y);
      state.camGoal.dist = clamp(pinch0.dist * pinch0.d / Math.max(20, d), CAM.distMin, CAM.distMax);
    }
  };
  const onUp = (e) => {
    const was = pointers.size;
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch0 = null;
    if (was === 1 && downInfo && !downInfo.moved && performance.now() - downInfo.t < 700) tap(e.clientX, e.clientY);
    if (pointers.size === 0) downInfo = null;
  };
  const onWheel = (e) => { e.preventDefault(); state.camGoal.dist = clamp(state.camGoal.dist * Math.exp(e.deltaY * 0.001), CAM.distMin, CAM.distMax); };
  canvas.addEventListener('pointerdown', onDown); canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp); canvas.addEventListener('pointercancel', onUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  const ray = new THREE.Raycaster(); const ndc = new THREE.Vector2();
  function tap(cx, cy) {
    const r = canvas.getBoundingClientRect(); ndc.set(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const proxies = [...state.animals.values()].map((a) => a.proxy);
    const hit = ray.intersectObjects(proxies, false)[0];
    if (hit) {
      const a = hit.object.userData.animal; a.react();
      try { state.onAnimalTap && state.onAnimalTap(a.def.index); } catch (err) { console.error(err); }
    }
  }

  // ---- resize / visibility ----
  const ro = new ResizeObserver(() => {
    const s = size(); if (s.w === w && s.h === h) return; w = s.w; h = s.h;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.fov = farmFov(w / h); camera.updateProjectionMatrix();
    if (state.post) state.post.composer.setSize(w, h);
  });
  ro.observe(container);
  let raf = 0;
  const onVis = () => { if (document.hidden) { cancelAnimationFrame(raf); raf = 0; } else if (!raf && !destroyed && state.world) { state.lastT = performance.now() / 1000; loop(); } };
  document.addEventListener('visibilitychange', onVis);

  // ---- camera ----
  const tmpV = new THREE.Vector3();
  function updateCamera(dt) {
    const g = state.camGoal, c = state.cam;
    if (state.focus) {
      if (state.t > state.focusUntil || !state.animals.has(state.focus.def.index)) { state.focus = null; g.target.set(0, 1, -1); g.dist = 27; g.el = 0.36; }
      else { g.target.set(state.focus.pos.x, 0.8, state.focus.pos.y); g.dist = 13 + state.focus.def.h * 2; g.el = 0.3; g.az = clamp(Math.atan2(state.focus.pos.x - 0, 30) * 0.8, CAM.azMin, CAM.azMax); }
    }
    const k = 1 - Math.exp(-dt * (REDUCED_MOTION ? 12 : 3));
    c.az = lerp(c.az, g.az, k); c.el = lerp(c.el, g.el, k); c.dist = lerp(c.dist, g.dist, k); c.target.lerp(g.target, k);
    const ce = Math.cos(c.el);
    camera.position.set(c.target.x + Math.sin(c.az) * ce * c.dist, c.target.y + Math.sin(c.el) * c.dist, c.target.z + Math.cos(c.az) * ce * c.dist);
    const minY = heightAt(camera.position.x, camera.position.z) + 1.5; if (camera.position.y < minY) camera.position.y = minY;
    // aim above the target so the Alps stay in frame
    tmpV.copy(c.target); tmpV.y += c.dist * 0.36 * clamp(camera.aspect, 0.45, 1) * (1 - (c.el - CAM.elMin) / (CAM.elMax - CAM.elMin) * 0.6);
    camera.lookAt(tmpV);
  }

  // ---- loop ----
  function loop() {
    if (destroyed) return;
    raf = requestAnimationFrame(loop);
    const now = performance.now() / 1000; let dt = now - state.lastT; state.lastT = now; dt = Math.min(dt, 0.05);
    state.t += dt;
    const W = state.world; if (!W) return;
    try {
      W.update(state.t, dt);
      if (state.items) for (const f of state.items.userData.animated) f(state.t);
      for (const a of state.animals.values()) a.update(dt, state.t);
      updateCamera(dt);
      const every = quality === 'high' ? 1 : quality === 'medium' ? 2 : 4;
      if (state.frame % every === 0) renderer.shadowMap.needsUpdate = true;
      state.frame++;
      renderer.info.reset();
      if (state.post) state.post.composer.render(dt); else renderer.render(W.scene, camera);
      state.info = { calls: renderer.info.render.calls, triangles: renderer.info.render.triangles, geometries: renderer.info.memory.geometries, textures: renderer.info.memory.textures };
    } catch (e) { if (!state.loggedErr) { state.loggedErr = true; console.error('World3D frame error', e); } }
    adapt(dt);
  }
  function adapt(dt) {
    if (opts.quality) return;
    state.fpsSamples.push(dt);
    if (state.fpsSamples.length < 90) return;
    const avg = state.fpsSamples.reduce((a, b) => a + b, 0) / state.fpsSamples.length; state.fpsSamples = [];
    if (avg > 1 / 40 && quality !== 'low') {
      quality = quality === 'high' ? 'medium' : 'low';
      renderer.setPixelRatio(pr()); renderer.setSize(w, h, false);
      setupPost();
    }
  }

  function destroy() {
    if (destroyed) return; destroyed = true;
    cancelAnimationFrame(raf); ro.disconnect(); document.removeEventListener('visibilitychange', onVis);
    canvas.removeEventListener('pointerdown', onDown); canvas.removeEventListener('pointermove', onMove);
    canvas.removeEventListener('pointerup', onUp); canvas.removeEventListener('pointercancel', onUp); canvas.removeEventListener('wheel', onWheel);
    for (const a of state.animals.values()) a.dispose();
    state.animals.clear();
    if (state.world) disposeTree(state.world.scene);
    if (state.post) { state.post.composer.dispose(); state.post.aoPass && state.post.aoPass.dispose(); state.post.bloomPass && state.post.bloomPass.dispose(); }
    state.env && state.env.dispose();
    farm.blobGeo.dispose(); farm.blobMat.map.dispose(); farm.blobMat.dispose();
    renderer.dispose(); renderer.forceContextLoss();
    canvas.remove();
  }

  return {
    ready,
    update(o = {}) { state.pending = state.pending.then(() => ready).then(() => applyOpts(o)); return state.pending; },
    destroy,
    get renderer() { return renderer; },
    get camera() { return camera; },
    get scene() { return state.world && state.world.scene; },
    get quality() { return quality; },
    get info() { return state.info; },
    _debug: {
      state, farm,
      // step the simulation without rendering (for tests/screenshots on slow software GL)
      advance(sec, dt = 1 / 30) {
        for (let t = 0; t < sec; t += dt) {
          state.t += dt; state.world.update(state.t, dt);
          if (state.items) for (const f of state.items.userData.animated) f(state.t);
          for (const a of state.animals.values()) a.update(dt, state.t);
          updateCamera(dt);
        }
        renderer.shadowMap.needsUpdate = true;
      },
    },
  };
}

function disposeTree(root) {
  const seen = new Set();
  root.traverse((o) => {
    if (o.userData.shared) return;
    if (o.geometry && !o.userData.sharedGeo && !o.geometry.userData?.shared && !seen.has(o.geometry)) { seen.add(o.geometry); o.geometry.dispose(); }
    if (o.material) for (const m of [].concat(o.material)) {
      if (seen.has(m) || isSharedMaterial(m)) continue; seen.add(m);
      for (const k in m) if (m[k] && m[k].isTexture && !seen.has(m[k])) { seen.add(m[k]); m[k].dispose(); }
      m.dispose();
    }
    if (o.isInstancedMesh) o.dispose();
  });
}
function isSharedMaterial(m) {
  if (!_nature) return false;
  for (const parts of Object.values(_nature)) for (const p of parts) if (p.material === m) return true;
  return false;
}

// ---------------------------------------------------------------------------------------------
// Backdrops (offscreen stills)
// ---------------------------------------------------------------------------------------------
const BACKDROPS = {
  // pos, look-at, horizontal fov, tilt-shift
  map: { pos: [2, 26, 44], look: [0, 12, -150], hfov: 72, tilt: null },
  meadow: { pos: [7, 1.5, 16], look: [-8, 12, -150], hfov: 66, tilt: { focus: 0.55, range: 0.08, amount: 1.4, top: 0.15 } },
  pond: { pos: [10.5, 3.4, 11.8], look: [6, 4, -150], hfov: 64, tilt: { focus: 0.6, range: 0.1, amount: 1.0, top: 0.15 } },
};
export async function renderBackdrop(width, height, variant = 'map') {
  const V = BACKDROPS[variant] || BACKDROPS.map;
  const W = Math.max(16, Math.round(width)), H = Math.max(16, Math.round(height));
  // Render at the screen's real pixel density (retina iPad = 2x) so the picture isn't stretched and soft.
  const dpr = clamp(window.devicePixelRatio || 1, 1, 2);
  const canvas = document.createElement('canvas');
  const renderer = makeRenderer(canvas, { width: W, height: H, pixelRatio: dpr, preserve: true });
  let world = null, env = null, post = null;
  try {
    world = await buildWorld({ quality: 'high', backdrop: true });
    env = makeEnvironment(renderer);
    world.scene.environment = env.texture; world.scene.environmentIntensity = 0.45;
    world.update(0, 0);
    const aspect = W / H;
    const vfov = clamp(2 * Math.atan(Math.tan((V.hfov * Math.PI) / 360) / aspect) * 180 / Math.PI, 36, 78);
    const camera = new THREE.PerspectiveCamera(vfov, aspect, 0.5, 5000);
    camera.position.set(...V.pos);
    const look = new THREE.Vector3(...V.look);
    if (aspect < 1) { look.y -= (1 - aspect) * 45; }
    camera.lookAt(look);
    post = makeComposer(renderer, world.scene, camera, W, H, { ao: true, bloom: true, tilt: V.tilt });
    renderer.shadowMap.needsUpdate = true;
    post.composer.render(0);
    return canvas.toDataURL('image/jpeg', 0.92);
  } finally {
    if (post) { post.composer.dispose(); post.aoPass && post.aoPass.dispose(); post.bloomPass && post.bloomPass.dispose(); }
    if (world) disposeTree(world.scene);
    env && env.dispose();
    renderer.dispose(); renderer.forceContextLoss();
  }
}

const World3D = { supported, mountFarm, renderBackdrop, WORLD_ANIMALS, heightAt, setAssetBase, purgeCache };
if (typeof window !== 'undefined') window.World3D = World3D;
export default World3D;
