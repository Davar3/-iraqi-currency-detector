/**
 * Iraqi Currency Detector — Browser ONNX Inference Engine
 */

const MODEL_PATH = "model/best.onnx";
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.45;
const IOU_THRESHOLD = 0.45;
const SPEAK_COOLDOWN = 2500; // ms

// 14 model classes → 7 denominations (merge ar + en pairs)
const CLASS_TO_DENOM = {
  0: "10000", 1: "10000", 2: "1000", 3: "1000",
  4: "25000", 5: "25000", 6: "250",  7: "250",
  8: "50000", 9: "50000", 10: "5000", 11: "5000",
  12: "500",  13: "500",
};

// Pairs: for each denomination, which class indices to merge
const DENOM_CLASSES = {
  "10000": [0, 1],  "1000": [2, 3],   "25000": [4, 5],
  "250": [6, 7],    "50000": [8, 9],   "5000": [10, 11],
  "500": [12, 13],
};

const DENOM_DISPLAY = {
  "250": "٢٥٠ دينار", "500": "٥٠٠ دينار", "1000": "١,٠٠٠ دينار",
  "5000": "٥,٠٠٠ دينار", "10000": "١٠,٠٠٠ دينار",
  "25000": "٢٥,٠٠٠ دينار", "50000": "٥٠,٠٠٠ دينار",
};

const DENOM_EN = {
  "250": "250", "500": "500", "1000": "1,000",
  "5000": "5,000", "10000": "10,000",
  "25000": "25,000", "50000": "50,000",
};

const BOX_COLORS = {
  "250": [34, 197, 94], "500": [59, 130, 246], "1000": [168, 85, 247],
  "5000": [245, 158, 11], "10000": [239, 68, 68], "25000": [14, 165, 233],
  "50000": [236, 72, 153],
};

let session = null;
let isRunning = false;
let lastSpoken = {};
let soundOn = true;
let audioUnlocked = false;
let facingMode = "environment";

// ── Audio setup ─────────────────────────────────────────
const voiceAudio = {};
["250", "500", "1000", "5000", "10000", "25000", "50000"].forEach((d) => {
  const a = new Audio(`assets/${d}.m4a`);
  a.preload = "auto";
  voiceAudio[d] = a;
});

// Unlock audio on first user tap (mobile requirement)
function unlockAudio() {
  if (audioUnlocked) return;
  const dummy = new AudioContext();
  dummy.resume().then(() => dummy.close());
  // Touch each audio element to unlock
  Object.values(voiceAudio).forEach((a) => {
    a.play().then(() => a.pause()).catch(() => {});
    a.currentTime = 0;
  });
  audioUnlocked = true;
}

let isPlaying = false;

function speakDenom(denom) {
  if (!soundOn || isPlaying) return;
  const now = Date.now();
  if (now - (lastSpoken[denom] || 0) < SPEAK_COOLDOWN) return;
  lastSpoken[denom] = now;

  const audio = voiceAudio[denom];
  if (audio) {
    isPlaying = true;
    audio.currentTime = 0;
    audio.play().then(() => {
      audio.onended = () => { isPlaying = false; };
    }).catch(() => {
      isPlaying = false;
    });
  }
}

// ── Load Model ──────────────────────────────────────────
async function loadModel(statusEl) {
  statusEl.textContent = "جاري تحميل نموذج الذكاء الاصطناعي... (12 MB)";
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });

    statusEl.textContent = "تجهيز النموذج...";
    const warmup = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    await session.run({ images: new ort.Tensor("float32", warmup, [1, 3, INPUT_SIZE, INPUT_SIZE]) });

    statusEl.textContent = "النموذج جاهز — اضغط زر التشغيل ✅";
    return true;
  } catch (e) {
    statusEl.textContent = "فشل تحميل النموذج: " + e.message;
    console.error(e);
    return false;
  }
}

// ── Camera ──────────────────────────────────────────────
async function startCamera(videoEl) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  videoEl.srcObject = stream;
  await videoEl.play();
}

function stopCamera(videoEl) {
  if (videoEl.srcObject) {
    videoEl.srcObject.getTracks().forEach((t) => t.stop());
    videoEl.srcObject = null;
  }
}

async function flipCamera(videoEl) {
  facingMode = facingMode === "environment" ? "user" : "environment";
  if (isRunning) {
    stopCamera(videoEl);
    await startCamera(videoEl);
  }
  return facingMode === "environment" ? "خلفية" : "أمامية";
}

// ── Preprocessing ───────────────────────────────────────
function preprocess(videoEl) {
  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext("2d");

  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
  const nw = Math.round(vw * scale), nh = Math.round(vh * scale);
  const dx = (INPUT_SIZE - nw) / 2, dy = (INPUT_SIZE - nh) / 2;

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(videoEl, dx, dy, nw, nh);

  const pixels = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const float32 = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const area = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < area; i++) {
    float32[i] = pixels[i * 4] / 255;
    float32[area + i] = pixels[i * 4 + 1] / 255;
    float32[2 * area + i] = pixels[i * 4 + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", float32, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    scale, dx, dy,
  };
}

// ── Postprocessing — merge ar+en into 7 denominations ───
function postprocess(output, meta) {
  const data = output.data;
  const N = 8400; // number of boxes

  const boxes = [];
  for (let i = 0; i < N; i++) {
    // For each box, compute merged denomination score
    // Take the MAX score across ar+en pairs for each denomination
    let bestDenom = null, bestScore = 0;

    for (const [denom, classIds] of Object.entries(DENOM_CLASSES)) {
      // Merge: take the max of the ar and en class scores
      let denomScore = 0;
      for (const cid of classIds) {
        const s = data[(4 + cid) * N + i];
        if (s > denomScore) denomScore = s;
      }
      if (denomScore > bestScore) {
        bestScore = denomScore;
        bestDenom = denom;
      }
    }

    if (bestScore < CONF_THRESHOLD || !bestDenom) continue;

    const cx = data[0 * N + i];
    const cy = data[1 * N + i];
    const w  = data[2 * N + i];
    const h  = data[3 * N + i];

    const x1 = (cx - w / 2 - meta.dx) / meta.scale;
    const y1 = (cy - h / 2 - meta.dy) / meta.scale;
    const x2 = (cx + w / 2 - meta.dx) / meta.scale;
    const y2 = (cy + h / 2 - meta.dy) / meta.scale;

    boxes.push({ x1, y1, x2, y2, score: bestScore, denom: bestDenom });
  }

  return nms(boxes, IOU_THRESHOLD);
}

// ── NMS ─────────────────────────────────────────────────
function nms(boxes, threshold) {
  boxes.sort((a, b) => b.score - a.score);
  const kept = [];
  const used = new Set();
  for (let i = 0; i < boxes.length; i++) {
    if (used.has(i)) continue;
    kept.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (used.has(j)) continue;
      if (iou(boxes[i], boxes[j]) > threshold) used.add(j);
    }
  }
  return kept;
}

function iou(a, b) {
  const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const aA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const aB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (aA + aB - inter);
}

// ── Drawing (clean rounded style) ───────────────────────
function drawDetections(ctx, detections, videoEl) {
  const cw = ctx.canvas.width, ch = ctx.canvas.height;
  const sx = cw / videoEl.videoWidth, sy = ch / videoEl.videoHeight;

  detections.forEach((det) => {
    const x = det.x1 * sx, y = det.y1 * sy;
    const w = (det.x2 - det.x1) * sx, h = (det.y2 - det.y1) * sy;
    const [r, g, b] = BOX_COLORS[det.denom] || [14, 165, 233];
    const color = `rgb(${r},${g},${b})`;

    // Semi-transparent fill
    ctx.fillStyle = `rgba(${r},${g},${b},0.12)`;
    ctx.beginPath();
    roundRect(ctx, x, y, w, h, 8);
    ctx.fill();

    // Border with glow
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    roundRect(ctx, x, y, w, h, 8);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Label pill
    const pct = Math.round(det.score * 100);
    const label = `${DENOM_DISPLAY[det.denom]}  ${pct}%`;
    ctx.font = "bold 14px Arial";
    const tw = ctx.measureText(label).width;
    const lx = x, ly = y - 28;
    const lw = tw + 16, lh = 24;

    ctx.fillStyle = color;
    ctx.beginPath();
    roundRect(ctx, lx, ly, lw, lh, 6);
    ctx.fill();

    ctx.fillStyle = "#fff";
    ctx.fillText(label, lx + 8, ly + 17);
  });
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// ── Detection Loop ──────────────────────────────────────
async function detectLoop(videoEl, overlayCanvas, statusEl, resultEl) {
  if (!isRunning || !session) return;

  const ctx = overlayCanvas.getContext("2d");
  overlayCanvas.width = videoEl.videoWidth || 640;
  overlayCanvas.height = videoEl.videoHeight || 480;
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  try {
    const { tensor, ...meta } = preprocess(videoEl);
    const results = await session.run({ images: tensor });
    const detections = postprocess(results["output0"], meta);

    drawDetections(ctx, detections, videoEl);

    if (detections.length === 0) {
      resultEl.textContent = "وجّه الكاميرا نحو ورقة نقدية عراقية";
    } else {
      let total = 0;
      const parts = detections.map((d) => {
        total += parseInt(d.denom) || 0;
        speakDenom(d.denom);
        return `${DENOM_DISPLAY[d.denom]} (${DENOM_EN[d.denom]} IQD, ${Math.round(d.score * 100)}%)`;
      });
      resultEl.innerHTML = `<span class="text-cyan-400 font-bold">${parts.join("  ·  ")}</span>`;
      if (total > 0) {
        resultEl.innerHTML += `<br><span class="text-purple-400">المجموع: ${total.toLocaleString()} دينار عراقي</span>`;
      }
    }
  } catch (e) {
    console.error("Detection error:", e);
  }

  setTimeout(() => detectLoop(videoEl, overlayCanvas, statusEl, resultEl), 150);
}

// ── Public API ──────────────────────────────────────────
async function startDetection(videoEl, overlayCanvas, statusEl, resultEl) {
  if (isRunning) return;
  unlockAudio(); // unlock on user gesture
  statusEl.textContent = "جاري تشغيل الكاميرا...";
  try {
    await startCamera(videoEl);
    isRunning = true;
    statusEl.textContent = "الكاميرا نشطة — وجّه الكاميرا نحو عملة عراقية";
    detectLoop(videoEl, overlayCanvas, statusEl, resultEl);
  } catch (e) {
    statusEl.textContent = "فشل الوصول للكاميرا: " + e.message;
  }
}

function stopDetection(videoEl) {
  isRunning = false;
  stopCamera(videoEl);
}

function toggleSound() {
  soundOn = !soundOn;
  return soundOn;
}
