// ============================================================================
//  Zoom Vision — Probador Virtual 3D  (ajuste 100% automático + oclusión)
//  Cámara + pose facial 3D (MediaPipe) + render 3D (Three.js)
//  - Centrado, escala y rotación automáticos.
//  - Oclusor de cabeza: oculta las patillas/lentes que quedan detrás de la cara.
// ============================================================================

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import {
  FaceLandmarker,
  FilesetResolver
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18';

// ---------- DOM ----------
const video    = document.getElementById('video');
const overlay  = document.getElementById('overlay');
const stageEl  = document.getElementById('stage');
const statusEl = document.getElementById('status');
const loaderEl = document.getElementById('loader');
const startBtn = document.getElementById('startBtn');
const debugBtn = document.getElementById('debugBtn');
const debugCanvas = document.getElementById('debug');
const debugCtx = debugCanvas.getContext('2d');
let showDebug = false;

// ---------- Ajuste automático (no hace falta tocar nada) ----------
const FRAME_WIDTH_IPD = 2.12; // ancho TOTAL del marco respecto a la distancia entre pupilas (~2x)
const V_NUDGE      = 0.02; // corrimiento vertical fino (fracción de la dist. pupilas)
const D_NUDGE      = 0.20; // empuja el anteojo hacia adelante (queda por delante de la cara)
const SMOOTH       = 0.55; // suavizado (0 = sin suavizar, 1 = muy suave)

// ---------- Oclusor de cabeza (malla del contorno facial) ----------
const OCC_DEPTH = 1.0;   // profundidad del "casco" hacia atrás (x ancho de cara)

// ---------- Índices de puntos faciales de MediaPipe ----------
const R_EYE_OUTER = 33,  R_EYE_INNER = 133;  // ojo derecho del sujeto
const L_EYE_OUTER = 263, L_EYE_INNER = 362;  // ojo izquierdo del sujeto
const FACE_LEFT = 234, FACE_RIGHT = 454;     // bordes de la cara (ancho)

// Contorno del rostro (FACE_OVAL de MediaPipe), en orden, como anillo cerrado.
const OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
  378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
  162, 21, 54, 103, 67, 109
];

// ---------- Lienzo espejo para MediaPipe ----------
// Le pasamos a MediaPipe el cuadro YA volteado, así los puntos y la matriz de
// pose quedan en el MISMO espacio que ves en pantalla (mismo lado izq/der).
const mirrorCanvas = document.createElement('canvas');
const mirrorCtx = mirrorCanvas.getContext('2d', { willReadFrequently: true });

// ============================================================================
//  THREE.JS
// ============================================================================
const renderer = new THREE.WebGLRenderer({ canvas: overlay, alpha: true, antialias: true });
renderer.setClearColor(0x000000, 0);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 4 / 3, 1, 100000);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dir = new THREE.DirectionalLight(0xffffff, 1.1);
dir.position.set(0.4, 1, 1.2);
scene.add(dir);

// ============================================================================
//  OCLUSORES (escriben SOLO profundidad; tapan lo que queda detrás del rostro)
// ============================================================================
const NUM_LM = 478;
const FACE_BACK_BIAS = 0.06; // empuja la malla facial un poco hacia atrás (en IPD)

// Reconstruye los triángulos de la malla facial desde las conexiones de MediaPipe.
// (Cada arista interior pertenece a 2 triángulos -> buscamos vecinos comunes.)
function buildFaceTriangles() {
  const conns = (typeof FaceLandmarker !== 'undefined' && FaceLandmarker.FACE_LANDMARKS_TESSELATION) || [];
  const adj = new Map();
  const add = (a, b) => { if (!adj.has(a)) adj.set(a, new Set()); adj.get(a).add(b); };
  for (const c of conns) { add(c.start, c.end); add(c.end, c.start); }
  const seen = new Set(), tris = [];
  for (const c of conns) {
    const a = c.start, b = c.end, na = adj.get(a), nb = adj.get(b);
    if (!na || !nb) continue;
    const [small, big] = na.size < nb.size ? [na, nb] : [nb, na];
    for (const k of small) {
      if (big.has(k)) {
        const key = [a, b, k].sort((x, y) => x - y).join('_');
        if (!seen.has(key)) { seen.add(key); tris.push(a, b, k); }
      }
    }
  }
  return tris;
}

const occMat = new THREE.MeshBasicMaterial({
  colorWrite: false, depthWrite: true, depthTest: true, side: THREE.DoubleSide
});

// --- Oclusor 1: malla DENSA de la cara (incluye nariz, mejillas, cejas) ---
const faceTris = buildFaceTriangles();
const occFaceGeo = new THREE.BufferGeometry();
const occFacePos = new Float32Array(NUM_LM * 3);
occFaceGeo.setAttribute('position', new THREE.BufferAttribute(occFacePos, 3));
if (faceTris.length) occFaceGeo.setIndex(faceTris);
const occFace = new THREE.Mesh(occFaceGeo, occMat);
occFace.renderOrder = -1; occFace.frustumCulled = false; occFace.visible = false;
scene.add(occFace);

// --- Oclusor 2: "casco" trasero (contorno -> ápice atrás) para tapar las patillas ---
const N_OVAL = OVAL.length;
const occBackGeo = new THREE.BufferGeometry();
const occBackPos = new Float32Array((N_OVAL + 1) * 3);
occBackGeo.setAttribute('position', new THREE.BufferAttribute(occBackPos, 3));
const backIdx = [];
for (let i = 0; i < N_OVAL; i++) backIdx.push(N_OVAL, i, (i + 1) % N_OVAL);
occBackGeo.setIndex(backIdx);
const occBack = new THREE.Mesh(occBackGeo, occMat);
occBack.renderOrder = -1; occBack.frustumCulled = false; occBack.visible = false;
scene.add(occBack);

// El anteojo: se posiciona sobre las pupilas; el modelo va recentrado dentro.
const glassesPivot = new THREE.Group();
glassesPivot.visible = false;
scene.add(glassesPivot);

let modelWidthX = 1;     // ancho del marco ya orientado (en unidades del modelo)
let modelLoaded = false;

const loader = new GLTFLoader();
loader.load(
  'glasses.glb',
  (gltf) => {
    const model = gltf.scene;
    model.updateWorldMatrix(true, true);

    // --- Recolectar centro de lentes y aplicar transparencia ---
    const lensCenters = [];
    model.traverse((o) => {
      if (o.isMesh) {
        const n = (o.name || '').toLowerCase();
        if (n.includes('lens') || n.includes('glass')) {
          lensCenters.push(new THREE.Box3().setFromObject(o).getCenter(new THREE.Vector3()));
          if (o.material) {
            o.material.transparent = true;
            o.material.opacity = 0.20;
            o.material.metalness = 0.0;
            o.material.roughness = 0.08;
            o.material.depthWrite = false;
          }
        }
        if (o.material) { o.material.side = THREE.DoubleSide; o.material.needsUpdate = true; }
      }
    });

    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const lensAnchor = lensCenters.length
      ? lensCenters.reduce((a, c) => a.add(c), new THREE.Vector3()).multiplyScalar(1 / lensCenters.length)
      : center.clone();

    // --- Auto-orientación: el frente (donde están las lentes) debe mirar a +Z ---
    const fwd = lensAnchor.clone().sub(center); fwd.y = 0;
    if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, 1);
    fwd.normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const right = new THREE.Vector3().crossVectors(up, fwd).normalize();
    const basis = new THREE.Matrix4().makeBasis(right, up, fwd).transpose(); // inversa = rotación a aplicar
    const orientQuat = new THREE.Quaternion().setFromRotationMatrix(basis);

    // Rotar el modelo y recentrarlo en el ancla de las lentes (queda en el origen del pivot).
    model.quaternion.copy(orientQuat);
    const anchorOriented = lensAnchor.clone().applyQuaternion(orientQuat);
    model.position.copy(anchorOriented).multiplyScalar(-1);

    // Ancho del marco ya orientado (eje X = izquierda-derecha) para la escala.
    model.updateWorldMatrix(true, true);
    const obox = new THREE.Box3().setFromObject(model);
    modelWidthX = obox.getSize(new THREE.Vector3()).x || 1;

    glassesPivot.add(model);
    modelLoaded = true;
    console.log('Modelo cargado. Ancho del marco:', modelWidthX.toFixed(3),
                '| frente detectado hacia', fwd.toArray().map((v) => v.toFixed(2)));
  },
  undefined,
  (err) => {
    console.error('Error cargando el modelo 3D:', err);
    setStatus('No se pudo cargar el modelo 3D (revisá que glasses.glb esté en la carpeta).');
  }
);

// ============================================================================
//  CÁMARA + MEDIAPIPE
// ============================================================================
let faceLandmarker = null;
let running = false;
let lastVideoTime = -1;

async function initFaceLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFacialTransformationMatrixes: true
  });
}

async function startCamera() {
  startBtn.disabled = true;
  loaderEl.classList.remove('hidden');
  setStatus('Pidiendo permiso de cámara…');
  try {
    if (!faceLandmarker) {
      setStatus('Inicializando motor facial…');
      await initFaceLandmarker();
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false
    });
    video.srcObject = stream;
    await new Promise((res) => { video.onloadedmetadata = () => { video.play(); res(); }; });

    mirrorCanvas.width = video.videoWidth;
    mirrorCanvas.height = video.videoHeight;
    stageEl.style.aspectRatio = `${video.videoWidth / video.videoHeight}`;
    resize();

    running = true;
    loaderEl.classList.add('hidden');
    setStatus('¡Listo! Mirá a la cámara — el anteojo se ajusta solo.');
    renderLoop();
  } catch (e) {
    console.error(e);
    loaderEl.classList.add('hidden');
    startBtn.disabled = false;
    if (e && e.name === 'NotAllowedError') {
      setStatus('Permiso de cámara denegado. Habilitalo en el navegador y reintentá.');
    } else {
      setStatus('No se pudo iniciar la cámara. Abrí la página con Live Server (http://localhost).');
    }
  }
}

// ============================================================================
//  BUCLE DE RENDER
// ============================================================================
const targetPos  = new THREE.Vector3();
const targetQuat = new THREE.Quaternion();
const poseMat    = new THREE.Matrix4();
const _p = new THREE.Vector3(), _q = new THREE.Quaternion(), _s = new THREE.Vector3();
const _fwd = new THREE.Vector3(), _up = new THREE.Vector3();
let targetScale = 0, targetIpd = 0, curIpd = 0;
let inited = false;

function renderLoop() {
  if (!running) return;
  requestAnimationFrame(renderLoop);

  if (faceLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    // Dibujar el cuadro VOLTEADO en el lienzo espejo y analizar ESE cuadro.
    mirrorCtx.save();
    mirrorCtx.scale(-1, 1);
    mirrorCtx.drawImage(video, -mirrorCanvas.width, 0, mirrorCanvas.width, mirrorCanvas.height);
    mirrorCtx.restore();

    const res = faceLandmarker.detectForVideo(mirrorCanvas, performance.now());

    const hasFace = res && res.faceLandmarks && res.faceLandmarks.length > 0;
    if (hasFace && modelLoaded) {
      const matrix = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
      computeTargets(res.faceLandmarks[0], matrix);
      applySmoothed();
      updateOccluder(res.faceLandmarks[0]);
      glassesPivot.visible = true;
      occFace.visible = true;
      occBack.visible = true;
      drawDebug(hasFace ? res.faceLandmarks[0] : null);
      setStatus(showDebug ? `Cara detectada ✓  — ${res.faceLandmarks[0].length} puntos` : 'Cara detectada ✓');
    } else {
      glassesPivot.visible = false;
      occFace.visible = false;
      occBack.visible = false;
      drawDebug(null);
      if (modelLoaded) setStatus('Acercá tu rostro y mirá a la cámara…');
    }
  }
  renderer.render(scene, camera);
}

function computeTargets(lm, matrix) {
  const W = overlay.clientWidth || 1;
  const H = overlay.clientHeight || 1;

  const toWorld = (p) =>
    new THREE.Vector3((p.x - 0.5) * W, (0.5 - p.y) * H, -p.z * W);

  const pupilR = toWorld(lm[R_EYE_OUTER]).add(toWorld(lm[R_EYE_INNER])).multiplyScalar(0.5);
  const pupilL = toWorld(lm[L_EYE_OUTER]).add(toWorld(lm[L_EYE_INNER])).multiplyScalar(0.5);

  targetIpd = pupilL.distanceTo(pupilR);
  const mid = pupilR.clone().add(pupilL).multiplyScalar(0.5);

  // Rotación real de la cabeza (yaw/pitch/roll) desde la matriz de pose
  if (matrix && matrix.data) {
    poseMat.fromArray(matrix.data);
    poseMat.decompose(_p, _q, _s);
    targetQuat.copy(_q);
  } else {
    targetQuat.identity();
  }

  _up.set(0, 1, 0).applyQuaternion(targetQuat);
  _fwd.set(0, 0, 1).applyQuaternion(targetQuat);
  targetPos.copy(mid)
    .addScaledVector(_up,  V_NUDGE * targetIpd)
    .addScaledVector(_fwd, D_NUDGE * targetIpd);

  targetScale = (targetIpd * FRAME_WIDTH_IPD) / modelWidthX;
}

function applySmoothed() {
  const a = 1 - SMOOTH;
  if (!inited) {
    glassesPivot.position.copy(targetPos);
    glassesPivot.quaternion.copy(targetQuat);
    glassesPivot.scale.setScalar(targetScale);
    curIpd = targetIpd;
    inited = true;
  } else {
    glassesPivot.position.lerp(targetPos, a);
    glassesPivot.quaternion.slerp(targetQuat, a);
    const s = glassesPivot.scale.x + (targetScale - glassesPivot.scale.x) * a;
    glassesPivot.scale.setScalar(s);
    curIpd += (targetIpd - curIpd) * a;
  }
  // El oclusor se reconstruye aparte en updateOccluder().
}

// Reconstruye las dos mallas del oclusor con los puntos reales del rostro.
function updateOccluder(lm) {
  const W = overlay.clientWidth || 1;
  const H = overlay.clientHeight || 1;
  const toW = (p, o) => { o.set((p.x - 0.5) * W, (0.5 - p.y) * H, -p.z * W); };
  const v = new THREE.Vector3();

  // --- Oclusor 1: malla densa de la cara (todos los puntos, incluye nariz) ---
  const n = Math.min(lm.length, NUM_LM);
  for (let i = 0; i < n; i++) {
    toW(lm[i], v);
    occFacePos[i * 3] = v.x; occFacePos[i * 3 + 1] = v.y; occFacePos[i * 3 + 2] = v.z;
  }
  occFaceGeo.attributes.position.needsUpdate = true;
  occFaceGeo.computeBoundingSphere();
  // La empujamos un poquito hacia atrás para no tapar el frente del anteojo,
  // pero la nariz (que sobresale más) sigue ocultando lo que pasa por detrás.
  _fwd.set(0, 0, 1).applyQuaternion(glassesPivot.quaternion);
  occFace.position.copy(_fwd).multiplyScalar(-FACE_BACK_BIAS * curIpd);

  // --- Oclusor 2: casco trasero (anillo del contorno + ápice atrás) ---
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < N_OVAL; i++) {
    toW(lm[OVAL[i]], v);
    occBackPos[i * 3] = v.x; occBackPos[i * 3 + 1] = v.y; occBackPos[i * 3 + 2] = v.z;
    cx += v.x; cy += v.y; cz += v.z;
  }
  cx /= N_OVAL; cy /= N_OVAL; cz /= N_OVAL;

  const fl = new THREE.Vector3(), fr = new THREE.Vector3();
  toW(lm[FACE_LEFT], fl); toW(lm[FACE_RIGHT], fr);
  const faceW = fl.distanceTo(fr) || 1;
  const d = faceW * OCC_DEPTH;
  occBackPos[N_OVAL * 3]     = cx - _fwd.x * d;
  occBackPos[N_OVAL * 3 + 1] = cy - _fwd.y * d;
  occBackPos[N_OVAL * 3 + 2] = cz - _fwd.z * d;
  occBackGeo.attributes.position.needsUpdate = true;
  occBackGeo.computeBoundingSphere();
}

// ---- Visor de puntos faciales (para verificar el tracking) ----
const EYE_PTS = [R_EYE_OUTER, R_EYE_INNER, L_EYE_OUTER, L_EYE_INNER];
function drawDebug(lm) {
  const w = debugCanvas.width, h = debugCanvas.height;
  debugCtx.clearRect(0, 0, w, h);
  if (!showDebug || !lm) return;

  // Todos los puntos (478) en turquesa tenue
  debugCtx.fillStyle = 'rgba(0, 220, 190, 0.75)';
  for (let i = 0; i < lm.length; i++) {
    debugCtx.beginPath();
    debugCtx.arc(lm[i].x * w, lm[i].y * h, 1.4, 0, 6.283);
    debugCtx.fill();
  }
  // Contorno facial (los puntos que usa el oclusor) en azul, conectados
  debugCtx.strokeStyle = '#1f7ae0';
  debugCtx.lineWidth = 1.5;
  debugCtx.beginPath();
  for (let i = 0; i < OVAL.length; i++) {
    const p = lm[OVAL[i]];
    if (i === 0) debugCtx.moveTo(p.x * w, p.y * h);
    else debugCtx.lineTo(p.x * w, p.y * h);
  }
  debugCtx.closePath();
  debugCtx.stroke();
  // Esquinas de los ojos (anclaje del anteojo) en rojo, más grandes
  debugCtx.fillStyle = '#ff3b6b';
  for (const i of EYE_PTS) {
    debugCtx.beginPath();
    debugCtx.arc(lm[i].x * w, lm[i].y * h, 4, 0, 6.283);
    debugCtx.fill();
  }
}

// ============================================================================
//  REDIMENSIONADO (cámara pixel-perfect: el plano z=0 mide W x H px)
// ============================================================================
function resize() {
  const w = overlay.clientWidth, h = overlay.clientHeight;
  if (!w || !h) return;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  const fovRad = (camera.fov * Math.PI) / 180;
  camera.position.set(0, 0, (h / 2) / Math.tan(fovRad / 2));
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();

  debugCanvas.width = w;
  debugCanvas.height = h;
}
new ResizeObserver(resize).observe(stageEl);
window.addEventListener('resize', resize);

startBtn.addEventListener('click', startCamera);
debugBtn.addEventListener('click', () => {
  showDebug = !showDebug;
  debugBtn.textContent = showDebug ? 'Ocultar puntos faciales' : 'Mostrar puntos faciales';
  if (!showDebug) debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
});
function setStatus(t) { statusEl.textContent = t; }
