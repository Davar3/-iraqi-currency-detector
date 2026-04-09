/**
 * Iraqi Currency Detector — Browser ONNX Inference Engine
 * Runs YOLOv8 model in the browser via ONNX Runtime Web.
 */

const MODEL_PATH = "model/best.onnx";
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.65;
const IOU_THRESHOLD = 0.5;
const NUM_CLASSES = 14;
const SPEAK_COOLDOWN = 4000; // ms
const CONFIRM_COUNT = 3; // require N consecutive detections before announcing

// Class names from the model (index → name)
const CLASS_NAMES = {
  0: "10000ar", 1: "10000en", 2: "1000ar", 3: "1000en",
  4: "25000ar", 5: "25000en", 6: "250ar", 7: "250en",
  8: "50000ar", 9: "50000en", 10: "5000ar", 11: "5000en",
  12: "500ar", 13: "500en",
};

const DENOM_DISPLAY = {
  "250": "٢٥٠ دينار", "500": "٥٠٠ دينار", "1000": "١,٠٠٠ دينار",
  "5000": "٥,٠٠٠ دينار", "10000": "١٠,٠٠٠ دينار",
  "25000": "٢٥,٠٠٠ دينار", "50000": "٥٠,٠٠٠ دينار",
};

const DENOM_EN = {
  "250": "250 IQD", "500": "500 IQD", "1000": "1,000 IQD",
  "5000": "5,000 IQD", "10000": "10,000 IQD",
  "25000": "25,000 IQD", "50000": "50,000 IQD",
};

const DENOM_SPEECH = {
  "250": "مئتان وخمسون دينار",
  "500": "خمسمائة دينار",
  "1000": "ألف دينار",
  "5000": "خمسة آلاف دينار",
  "10000": "عشرة آلاف دينار",
  "25000": "خمسة وعشرون ألف دينار",
  "50000": "خمسون ألف دينار",
};

const BOX_COLORS = {
  "250": "#22c55e", "500": "#3b82f6", "1000": "#a855f7",
  "5000": "#f59e0b", "10000": "#ef4444", "25000": "#0ea5e9", "50000": "#ec4899",
};

let session = null;
let isRunning = false;
let lastSpoken = {};
let soundOn = true;
let detStreak = {}; // denom -> consecutive count
let facingMode = "environment"; // "environment" = back, "user" = front

// ── Load Model ──────────────────────────────────────────
async function loadModel(statusEl) {
  statusEl.textContent = "جاري تحميل نموذج الذكاء الاصطناعي... (12 MB)";
  try {
    // Configure WASM paths for CDN
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";

    // Try WASM first (most reliable), then WebGL
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });

    // Warm up with a dummy run
    statusEl.textContent = "تجهيز النموذج...";
    const warmup = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    const warmupTensor = new ort.Tensor("float32", warmup, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    await session.run({ images: warmupTensor });

    statusEl.textContent = "النموذج جاهز — اضغط زر التشغيل ✅";
    console.log("Model loaded and warmed up successfully");
    return true;
  } catch (e) {
    statusEl.textContent = "فشل تحميل النموذج: " + e.message;
    console.error("Model load error:", e);
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

  // Letterbox: fit video into 640x640 maintaining aspect ratio
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
  const nw = Math.round(vw * scale), nh = Math.round(vh * scale);
  const dx = (INPUT_SIZE - nw) / 2, dy = (INPUT_SIZE - nh) / 2;

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(videoEl, dx, dy, nw, nh);

  const imgData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = imgData.data;

  // Convert to CHW float32 normalized [0,1]
  const float32 = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const area = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < area; i++) {
    float32[i] = pixels[i * 4] / 255;             // R
    float32[area + i] = pixels[i * 4 + 1] / 255;  // G
    float32[2 * area + i] = pixels[i * 4 + 2] / 255; // B
  }

  return {
    tensor: new ort.Tensor("float32", float32, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    scale, dx, dy, vw, vh,
  };
}

// ── Postprocessing (YOLOv8 output) ──────────────────────
function postprocess(output, meta) {
  // output shape: [1, 18, 8400] → transpose to [8400, 18]
  const data = output.data;
  const numBoxes = 8400;
  const numValues = 4 + NUM_CLASSES; // 18

  const boxes = [];
  for (let i = 0; i < numBoxes; i++) {
    // Extract class scores (indices 4-17)
    let maxScore = 0, maxClass = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const score = data[(4 + c) * numBoxes + i];
      if (score > maxScore) {
        maxScore = score;
        maxClass = c;
      }
    }

    if (maxScore < CONF_THRESHOLD) continue;

    // Extract box (cx, cy, w, h) in model coords (640x640)
    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];

    // Convert to original video coords
    const x1 = (cx - w / 2 - meta.dx) / meta.scale;
    const y1 = (cy - h / 2 - meta.dy) / meta.scale;
    const x2 = (cx + w / 2 - meta.dx) / meta.scale;
    const y2 = (cy + h / 2 - meta.dy) / meta.scale;

    const raw = CLASS_NAMES[maxClass] || String(maxClass);
    const denom = raw.replace("ar", "").replace("en", "");

    boxes.push({ x1, y1, x2, y2, score: maxScore, classId: maxClass, denom, raw });
  }

  return nms(boxes, IOU_THRESHOLD);
}

// ── NMS ─────────────────────────────────────────────────
function nms(boxes, threshold) {
  boxes.sort((a, b) => b.score - a.score);
  const kept = [];
  const suppressed = new Set();

  for (let i = 0; i < boxes.length; i++) {
    if (suppressed.has(i)) continue;
    kept.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (suppressed.has(j)) continue;
      if (iou(boxes[i], boxes[j]) > threshold) {
        suppressed.add(j);
      }
    }
  }
  return kept;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

// ── Drawing ─────────────────────────────────────────────
function drawDetections(ctx, detections, videoEl) {
  const cw = ctx.canvas.width, ch = ctx.canvas.height;
  const sx = cw / videoEl.videoWidth, sy = ch / videoEl.videoHeight;

  detections.forEach((det) => {
    const x = det.x1 * sx, y = det.y1 * sy;
    const w = (det.x2 - det.x1) * sx, h = (det.y2 - det.y1) * sy;
    const color = BOX_COLORS[det.denom] || "#0ea5e9";

    // Box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    // Label background
    const label = `${DENOM_DISPLAY[det.denom] || det.denom}  ${Math.round(det.score * 100)}%`;
    ctx.font = "bold 16px Arial";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 24, tw + 12, 24);

    // Label text
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 6, y - 6);
  });
}

// ── Voice playback (custom recordings) ──────────────────
// Preload audio files
const voiceAudio = {};
["250", "500", "1000", "5000", "10000", "25000", "50000"].forEach((d) => {
  const audio = new Audio(`assets/${d}.m4a`);
  audio.preload = "auto";
  voiceAudio[d] = audio;
});

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
      // Fallback to Web Speech API if audio fails
      if ("speechSynthesis" in window) {
        const utt = new SpeechSynthesisUtterance(DENOM_SPEECH[denom] || denom);
        utt.lang = "ar-SA";
        utt.rate = 0.9;
        speechSynthesis.speak(utt);
      }
    });
  }
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
    const output = results["output0"];
    const detections = postprocess(output, meta);

    // Debug logging (first few frames only)
    if (!window._debugCount) window._debugCount = 0;
    if (window._debugCount < 5) {
      console.log(`Frame ${window._debugCount}: ${detections.length} detections, output shape: [${output.dims}]`);
      if (detections.length > 0) {
        detections.forEach(d => console.log(`  ${d.denom}: ${(d.score*100).toFixed(1)}%`));
      }
      window._debugCount++;
    }

    drawDetections(ctx, detections, videoEl);

    // Update streak counts for smoothing
    if (detections.length === 0) {
      detStreak = {};
      resultEl.textContent = "لم يتم الكشف عن أي عملة — وجّه الكاميرا نحو ورقة نقدية";
    } else {
      const currentDenoms = new Set(detections.map(d => d.denom));
      // Reset unseen, increment seen
      for (const d of Object.keys(detStreak)) {
        if (!currentDenoms.has(d)) detStreak[d] = 0;
      }
      for (const d of currentDenoms) {
        detStreak[d] = (detStreak[d] || 0) + 1;
      }

      // Only show confirmed detections (seen CONFIRM_COUNT times in a row)
      const confirmed = detections.filter(d => (detStreak[d.denom] || 0) >= CONFIRM_COUNT);

      if (confirmed.length === 0) {
        resultEl.textContent = "جاري التحقق... ثبّت الورقة أمام الكاميرا";
      } else {
        let total = 0;
        const parts = confirmed.map((d) => {
          total += parseInt(d.denom) || 0;
          speakDenom(d.denom);
          return `${DENOM_DISPLAY[d.denom] || d.denom} (${DENOM_EN[d.denom]}, ${Math.round(d.score * 100)}%)`;
        });
        resultEl.innerHTML = `<span class="text-cyan-400 font-bold">${parts.join("  ·  ")}</span>`;
        if (total > 0) {
          resultEl.innerHTML += `<br><span class="text-purple-400">المجموع: ${total.toLocaleString()} دينار عراقي</span>`;
        }
      }
    }
  } catch (e) {
    console.error("Detection error:", e);
  }

  // ~5 FPS
  setTimeout(() => detectLoop(videoEl, overlayCanvas, statusEl, resultEl), 200);
}

// ── Public API ──────────────────────────────────────────
async function startDetection(videoEl, overlayCanvas, statusEl, resultEl) {
  if (isRunning) return;
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
