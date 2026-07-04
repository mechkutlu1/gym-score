/* RepScore — on-device rep counting & form scoring with MediaPipe Pose Landmarker */

import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

/* ---------------- landmarks ---------------- */

const LM = {
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_ELBOW: 13,    R_ELBOW: 14,
  L_WRIST: 15,    R_WRIST: 16,
  L_HIP: 23,      R_HIP: 24,
  L_KNEE: 25,     R_KNEE: 26,
  L_ANKLE: 27,    R_ANKLE: 28,
};

const SKELETON = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28],
];

/* ---------------- exercise definitions ----------------
   Each exercise tracks one joint angle (three landmarks per side).
   A rep = extended -> flexed -> extended.
   enter/exit are hysteresis thresholds in degrees;
   perfectBottom/perfectTop define the 100-point range of motion. */

const EXERCISES = {
  squat: {
    label: "Squat",
    triples: { left: [LM.L_HIP, LM.L_KNEE, LM.L_ANKLE], right: [LM.R_HIP, LM.R_KNEE, LM.R_ANKLE] },
    enter: 110, exit: 155, perfectBottom: 78, perfectTop: 170,
    cues: {
      shallow: "Squat deeper — aim for the hip crease below the knee.",
      noLockout: "Stand up fully and finish the rep with hips locked out.",
      good: "Solid depth and lockout.",
    },
  },
  pushup: {
    label: "Push-up",
    triples: { left: [LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST], right: [LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST] },
    enter: 105, exit: 150, perfectBottom: 72, perfectTop: 168,
    cues: {
      shallow: "Lower further — chest close to the floor.",
      noLockout: "Press all the way up until the elbows are straight.",
      good: "Full range of motion.",
    },
  },
  pullup: {
    label: "Pull-up",
    triples: { left: [LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST], right: [LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST] },
    enter: 100, exit: 150, perfectBottom: 60, perfectTop: 168,
    requireWristsAboveShoulders: true,
    cues: {
      shallow: "Pull higher — chin over the bar.",
      noLockout: "Return to a full dead hang between reps.",
      good: "Chin over, full hang. Clean rep.",
    },
  },
  curl: {
    label: "Dumbbell curl",
    triples: { left: [LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST], right: [LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST] },
    enter: 95, exit: 150, perfectBottom: 45, perfectTop: 168,
    cues: {
      shallow: "Curl higher — full elbow flexion at the top.",
      noLockout: "Lower under control until the arm is straight.",
      good: "Full curl, full extension.",
    },
  },
};

/* ---------------- state ---------------- */

const state = {
  landmarker: null,
  running: false,
  stream: null,
  facingMode: "user",
  mirrored: true,
  exercise: "squat",
  phase: "idle",          // idle | extended | flexed
  smoothAngle: null,
  minAngle: 180,
  maxAngle: 0,
  reps: 0,
  scores: [],
  lastVideoTime: -1,
};

/* ---------------- elements ---------------- */

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const repCountEl = document.getElementById("repCount");
const lastScoreEl = document.getElementById("lastScore");
const avgScoreEl = document.getElementById("avgScore");
const phaseEl = document.getElementById("phaseText");
const coachList = document.getElementById("coachList");
const romRing = document.getElementById("romRing");
const repFlash = document.getElementById("repFlash");
const RING_C = 2 * Math.PI * 52; // matches r=52 in the SVG
const stageEmpty = document.getElementById("stageEmpty");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const flipBtn = document.getElementById("flipBtn");
const resetBtn = document.getElementById("resetBtn");

/* ---------------- geometry ---------------- */

function angleAt(a, b, c) {
  // angle ABC in degrees, using image-plane coordinates
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const m1 = Math.hypot(v1.x, v1.y);
  const m2 = Math.hypot(v2.x, v2.y);
  if (m1 === 0 || m2 === 0) return null;
  const cos = Math.min(1, Math.max(-1, dot / (m1 * m2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function visibilityOf(lms, ids) {
  return ids.reduce((s, i) => s + (lms[i]?.visibility ?? 0), 0) / ids.length;
}

/* Map a normalized landmark to canvas pixels, accounting for object-fit: cover
   and optional mirroring. */
function toPixels(lm, vw, vh, cw, ch) {
  const scale = Math.max(cw / vw, ch / vh);
  const dx = (vw * scale - cw) / 2;
  const dy = (vh * scale - ch) / 2;
  let x = lm.x * vw * scale - dx;
  const y = lm.y * vh * scale - dy;
  if (state.mirrored) x = cw - x;
  return { x, y };
}

/* ---------------- scoring ---------------- */

function scoreRep(def, minA, maxA) {
  const depthSpan = def.enter - def.perfectBottom;
  const lockSpan = def.perfectTop - def.exit;
  const depth = Math.min(1, Math.max(0, (def.enter - minA) / depthSpan));
  const lockout = Math.min(1, Math.max(0, (maxA - def.exit) / lockSpan));
  const total = Math.round(60 * depth + 40 * lockout);
  let cue = def.cues.good;
  if (depth < 0.75) cue = def.cues.shallow;
  else if (lockout < 0.6) cue = def.cues.noLockout;
  return { total, cue };
}

/* Live range-of-motion ring: fraction of the way from full extension
   (perfectTop) to the exercise's perfect bottom angle. */
function updateRing(angle, def) {
  let frac = 0;
  if (angle != null && def) {
    frac = (def.perfectTop - angle) / (def.perfectTop - def.perfectBottom);
    frac = Math.min(1, Math.max(0, frac));
  }
  romRing.style.strokeDashoffset = RING_C * (1 - frac);
}

function gradeClass(score) {
  return score >= 85 ? "grade-a" : score >= 65 ? "grade-b" : "grade-c";
}

function setScore(el, score) {
  el.textContent = score;
  el.classList.remove("grade-a", "grade-b", "grade-c");
  el.classList.add(gradeClass(score));
}

function beep() {
  try {
    const ac = beep.ctx || (beep.ctx = new (window.AudioContext || window.webkitAudioContext)());
    const o = ac.createOscillator();
    const g = ac.createGain();
    o.frequency.value = 880;
    g.gain.setValueAtTime(0.08, ac.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0001, ac.currentTime + 0.12);
    o.connect(g).connect(ac.destination);
    o.start();
    o.stop(ac.currentTime + 0.13);
  } catch (_) { /* audio optional */ }
}

/* ---------------- rep state machine ---------------- */

function updateMachine(def, angle) {
  if (state.phase === "idle") {
    if (angle > def.exit) state.phase = "extended";
    return;
  }
  if (state.phase === "extended") {
    state.maxAngle = Math.max(state.maxAngle, angle);
    if (angle < def.enter) {
      state.phase = "flexed";
      state.minAngle = angle;
    }
  } else if (state.phase === "flexed") {
    state.minAngle = Math.min(state.minAngle, angle);
    if (angle > def.exit) {
      // full cycle complete
      const { total, cue } = scoreRep(def, state.minAngle, angle);
      state.reps += 1;
      state.scores.push(total);
      addCoachNote(total, cue);
      state.phase = "extended";
      state.maxAngle = angle;
      state.minAngle = 180;
      repCountEl.textContent = state.reps;
      setScore(lastScoreEl, total);
      avgScoreEl.textContent = Math.round(state.scores.reduce((a, b) => a + b, 0) / state.scores.length);
      repCountEl.classList.remove("pulse");
      void repCountEl.offsetWidth; // restart animation
      repCountEl.classList.add("pulse");
      repFlash.classList.remove("go");
      void repFlash.offsetWidth;
      repFlash.classList.add("go");
      beep();
    }
  }
}

function addCoachNote(score, cue) {
  const g = gradeClass(score);
  const li = document.createElement("li");
  li.innerHTML =
    `<div class="note-head">` +
    `<span class="note-rep">Rep ${state.reps}</span>` +
    `<span class="note-bar ${g}"><i style="width:${score}%"></i></span>` +
    `<span class="note-score ${g}">${score}</span>` +
    `</div>` +
    `<span class="note-text">${cue}</span>`;
  coachList.prepend(li);
  while (coachList.children.length > 6) coachList.removeChild(coachList.lastChild);
}

/* ---------------- drawing ---------------- */

function accentColor() {
  return getComputedStyle(document.body).getPropertyValue("--accent").trim() || "#D23B33";
}

function drawFrame(lms, def, joint) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const cw = canvas.width, ch = canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!lms) return;

  const px = lms.map((lm) => toPixels(lm, vw, vh, cw, ch));
  const accent = accentColor();

  // skeleton
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(237, 232, 220, 0.55)";
  for (const [a, b] of SKELETON) {
    if ((lms[a]?.visibility ?? 0) < 0.4 || (lms[b]?.visibility ?? 0) < 0.4) continue;
    ctx.beginPath();
    ctx.moveTo(px[a].x, px[a].y);
    ctx.lineTo(px[b].x, px[b].y);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(237, 232, 220, 0.85)";
  for (const i of Object.values(LM)) {
    if ((lms[i]?.visibility ?? 0) < 0.4) continue;
    ctx.beginPath();
    ctx.arc(px[i].x, px[i].y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // highlight tracked limb + angle gauge at the joint
  if (joint) {
    const [ai, bi, ci] = joint.ids;
    ctx.lineWidth = 5;
    ctx.strokeStyle = accent;
    ctx.beginPath();
    ctx.moveTo(px[ai].x, px[ai].y);
    ctx.lineTo(px[bi].x, px[bi].y);
    ctx.lineTo(px[ci].x, px[ci].y);
    ctx.stroke();

    const b = px[bi];
    const a1 = Math.atan2(px[ai].y - b.y, px[ai].x - b.x);
    const a2 = Math.atan2(px[ci].y - b.y, px[ci].x - b.x);
    let start = a1, end = a2;
    let sweep = end - start;
    while (sweep < -Math.PI) sweep += 2 * Math.PI;
    while (sweep > Math.PI) sweep -= 2 * Math.PI;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(b.x, b.y, 30, start, start + sweep, sweep < 0);
    ctx.stroke();

    const deg = Math.round(joint.angle);
    ctx.font = "700 18px Archivo, sans-serif";
    ctx.fillStyle = accent;
    ctx.strokeStyle = "rgba(0,0,0,0.7)";
    ctx.lineWidth = 4;
    const tx = b.x + 38, ty = b.y - 12;
    ctx.strokeText(`${deg}°`, tx, ty);
    ctx.fillText(`${deg}°`, tx, ty);
  }
}

/* ---------------- main loop ---------------- */

function loop() {
  if (!state.running) return;
  const def = EXERCISES[state.exercise];

  if (video.currentTime !== state.lastVideoTime && video.videoWidth > 0) {
    state.lastVideoTime = video.currentTime;
    fitCanvas();

    const result = state.landmarker.detectForVideo(video, performance.now());
    const lms = result.landmarks?.[0];

    if (!lms) {
      drawFrame(null);
      updateRing(null, def);
      phaseEl.textContent = "No athlete detected";
    } else {
      // choose the better-visible side
      const visL = visibilityOf(lms, def.triples.left);
      const visR = visibilityOf(lms, def.triples.right);
      const ids = visL >= visR ? def.triples.left : def.triples.right;
      const vis = Math.max(visL, visR);

      let joint = null;
      if (vis < 0.5) {
        phaseEl.textContent = "Move into frame";
        state.phase = "idle";
      } else {
        const raw = angleAt(lms[ids[0]], lms[ids[1]], lms[ids[2]]);
        if (raw != null) {
          state.smoothAngle = state.smoothAngle == null ? raw : 0.65 * state.smoothAngle + 0.35 * raw;
          const angle = state.smoothAngle;
          joint = { ids, angle };

          let gated = false;
          if (def.requireWristsAboveShoulders) {
            const wristY = Math.min(lms[LM.L_WRIST].y, lms[LM.R_WRIST].y);
            const shoulderY = Math.min(lms[LM.L_SHOULDER].y, lms[LM.R_SHOULDER].y);
            if (wristY > shoulderY) {
              gated = true;
              phaseEl.textContent = "Hands on the bar";
              state.phase = "idle";
            }
          }
          if (!gated) {
            updateMachine(def, angle);
            updateRing(angle, def);
            phaseEl.textContent =
              state.phase === "flexed" ? "Down / hold" :
              state.phase === "extended" ? "Ready — go" : "Get set";
          } else {
            updateRing(null, def);
          }
        }
      }
      drawFrame(lms, def, joint);
    }
  }
  requestAnimationFrame(loop);
}

function fitCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(rect.width * dpr);
  const h = Math.round(rect.height * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

/* ---------------- camera & model ---------------- */

async function ensureModel() {
  if (state.landmarker) return;
  phaseEl.textContent = "Loading model…";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  state.landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

async function startCamera() {
  try {
    await ensureModel();
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: state.facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = state.stream;
    await video.play();
    state.mirrored = state.facingMode === "user";
    video.classList.toggle("mirrored", state.mirrored);
    state.running = true;
    state.phase = "idle";
    state.smoothAngle = null;
    stageEmpty.classList.add("hidden");
    stopBtn.disabled = false;
    flipBtn.disabled = false;
    phaseEl.textContent = "Get set";
    requestAnimationFrame(loop);
  } catch (err) {
    phaseEl.textContent = "Camera unavailable";
    stageEmpty.classList.remove("hidden");
    stageEmpty.querySelector("p").textContent =
      "Camera access failed. Allow camera permission and reload — note the page must be served over HTTPS (GitHub Pages is).";
    console.error(err);
  }
}

function stopCamera() {
  state.running = false;
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  phaseEl.textContent = "Camera off";
  stopBtn.disabled = true;
  flipBtn.disabled = true;
  stageEmpty.classList.remove("hidden");
}

/* ---------------- UI wiring ---------------- */

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

flipBtn.addEventListener("click", async () => {
  state.facingMode = state.facingMode === "user" ? "environment" : "user";
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    await startCamera();
  }
});

resetBtn.addEventListener("click", () => {
  state.reps = 0;
  state.scores = [];
  state.phase = "idle";
  state.minAngle = 180;
  state.maxAngle = 0;
  state.smoothAngle = null;
  repCountEl.textContent = "0";
  lastScoreEl.textContent = "—";
  lastScoreEl.classList.remove("grade-a", "grade-b", "grade-c");
  avgScoreEl.textContent = "—";
  updateRing(null, EXERCISES[state.exercise]);
  coachList.innerHTML = '<li><span class="note-text">Set reset. Complete a rep to see its score.</span></li>';
});

document.querySelectorAll(".card").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".card").forEach((c) => c.classList.remove("is-active"));
    chip.classList.add("is-active");
    state.exercise = chip.dataset.exercise;
    document.body.dataset.exercise = state.exercise;
    resetBtn.click();
  });
});

window.addEventListener("resize", () => { if (state.running) fitCanvas(); });
