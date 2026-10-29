// =============================================================================
// File:    sketch.js
// Project: MediaPipe Demos — MDes Prototyping, CCA
// Demo:    2 — Face Mesh (eye-tracking installation)
//
// Authors: Copilot
//          Thomas J McLeish
// License: MIT — see LICENSE in the root of this repository
// =============================================================================
//
// PURPOSE
// -------
// This sketch demonstrates how to use MediaPipe FaceMesh to detect and track
// the eyes and irises in real time using your webcam. When a face is
// detected, the model returns 478 landmark points covering the full face;
// this sketch uses only the eye and iris points.
//
// Built for an exhibition projection: an article about surveillance scrolls
// slowly up the screen (see index.html), and every visitor's eyes are drawn
// over the text as eye and iris landmark dots.
//
//   Position — each pair of eyes starts where that person stands in the
//              camera's view (mirrored, spread across the projection), then
//              shifts towards where they are looking. Gaze is estimated from
//              head direction plus where each iris sits within the eye — see
//              estimateGaze.
//   Size     — each pair of eyes is magnified around its own centre in
//              proportion to how large the face is in the camera, so people
//              standing closer get bigger eyes. Nearer faces draw on top.
//
// The eye canvas is not wiped between frames, so moving eyes leave a short,
// fading trail (adapted from a p5.js + clmtrackr eye-trail sketch). The
// webcam feed is used for tracking but is never shown.
//
// KEYS: F toggles fullscreen; any other key clears the trails.
// Clicking saves the eye layer (on black, without the text) as a PNG.
//
// WHAT IS MediaPipe FACE LANDMARKER?
// ------------------------------------
// MediaPipe Face Landmarker (part of the Tasks Vision API) is a
// machine-learning model from Google that analyses each video frame and
// returns the 3-D positions of 478 key points on any detected face (468 face
// surface points + 10 iris points). The model runs entirely in the browser —
// no data leaves your device.
//
// Each landmark has x, y, and z properties:
//   x, y — normalised position (0.0–1.0 relative to frame width/height)
//   z     — depth estimate (smaller = closer to camera)
//
// See drawFrame for how these are turned into positions and sizes on the
// projection.
//
// MULTIPLE FACES
// --------------
// Up to MAX_FACES faces are tracked at once; each detected face gets its
// own set of eye and iris dots.
//
// FACE REGIONS
// ------------
// This demo draws only the eye-tracking regions: the eye outlines and the
// irises. Each person gets their own colour palette (see PALETTES), so
// visitors can tell which eyes are theirs.
//
// The region index arrays below come from the MediaPipe Face Landmarker
// canonical face model documentation.
//
// HOW THE TASKS API DIFFERS FROM THE LEGACY SOLUTIONS API
// --------------------------------------------------------
// This demo uses @mediapipe/tasks-vision instead of the deprecated
// @mediapipe/face_mesh package. Key differences:
//
//   Legacy pattern                Tasks API pattern
//   ─────────────────────────     ──────────────────────────────────────
//   new FaceMesh({ locateFile })  FaceLandmarker.createFromOptions(...)
//   faceMesh.setOptions(...)      options passed to createFromOptions
//   faceMesh.onResults(callback)  result returned by detectForVideo(...)
//   await faceMesh.send(...)      result = landmarker.detectForVideo(video, ts)
//   results.multiFaceLandmarks    result.faceLandmarks
//
// Iris landmarks (indices 468–477) are included automatically in the
// Face Landmarker model — no refineLandmarks option is needed.
//
// DEBUGGING
// ---------
// Set debugMode = true to log landmark counts and model events to the
// browser console. Open DevTools (F12) → Console tab to see the output.

// =============================================================================
// CONSTANTS
// =============================================================================

const queryParams = new URLSearchParams(window.location.search);

// Toggle debug output: console logging plus an on-canvas face/eye counter.
// Set to true here, or add ?debug=1 to the page URL.
const debugMode = false || queryParams.get("debug") === "1";

// Phones and tablets are more reliable with the CPU delegate for MediaPipe.
// Use ?delegate=gpu or ?delegate=cpu to override this default for testing.
const isMobileDevice = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
const requestedDelegate = (queryParams.get("delegate") || "").toLowerCase();
const preferredDelegate = requestedDelegate === "gpu" || requestedDelegate === "cpu"
  ? requestedDelegate.toUpperCase()
  : (isMobileDevice ? "CPU" : "GPU");
const fallbackDelegate = preferredDelegate === "GPU" ? "CPU" : "GPU";

// Background used when saving the eye layer as a PNG. On screen the eye
// canvas is transparent so the article shows through.
const BACKGROUND_COLOR = "#000";

// Maximum number of faces (pairs of eyes) tracked at once. 10 faces = up to
// 20 eyes. Higher values cost more processing per frame.
const MAX_FACES = 10;

// Confidence thresholds (0–1) for the face model. MediaPipe's default is 0.5,
// which tends to drop faces that are small, turned, or dimly lit — common
// when several people share the frame. Lower values find more faces but may
// produce the occasional false detection; raise them if dots appear on
// things that aren't faces.
const MIN_DETECTION_CONFIDENCE = 0.3;  // finding a new face
const MIN_PRESENCE_CONFIDENCE  = 0.3;  // deciding a face is still there
const MIN_TRACKING_CONFIDENCE  = 0.3;  // following a face between frames

// ── Trail / buffering ────────────────────────────────────────────────────────
// The canvas is not wiped between frames, so each drawn eye leaves a trail
// behind it as it moves.
//
// DRAW_FPS     — how often a new set of eyes is stamped onto the canvas.
//                Lower = more spaced-out, stepped trail. Tracking itself
//                still runs as fast as the device allows.
// TRAIL_FADE   — how much of the old trail is darkened on each draw (0–1).
//                Higher = shorter trail. At 0.25 and 10 fps the trail is
//                gone in about 1.5 seconds. 0 keeps the trail forever until
//                you press a key; 1 wipes it so there is no trail at all.
// MIRROR       — flip horizontally so moving left moves the eyes left.
const DRAW_FPS   = 10;
const TRAIL_FADE = 0.25;
const MIRROR     = true;

// ── Eye size and placement ───────────────────────────────────────────────────
// Size is driven by how far apart a person's eyes are in the camera image,
// which shrinks the further they stand from the webcam.
//
// EYE_SIZE       — on-screen distance between the two eyes, as a fraction of
//                  screen width, for someone at the reference distance
//                  below. 0.15 ≈ 290 px on a 1920 px wide screen.
// REFERENCE_GAP  — the camera eye gap (fraction of camera width) that counts
//                  as the reference distance. 0.075 is roughly someone
//                  sitting at a laptop webcam, about 60 cm away.
// SIZE_CONTRAST  — how sharply eyes shrink with distance. 1 = in proportion
//                  (twice as far → half the size). At 2.5, twice as far
//                  gives about a fifth of the size and three times as far
//                  about a fifteenth, so people behind the main viewer look
//                  much smaller.
// MAX_EYE_SIZE   — cap (fraction of screen width) so someone leaning right
//                  into the webcam doesn't fill the screen.
// DOT_SIZE       — dot radius as a fraction of the on-screen distance between
//                  the two eyes, so dots grow and shrink with the eyes.
// MIN_DOT_RADIUS — smallest dot radius in pixels, so far-away eyes stay
//                  visible.
const EYE_SIZE       = 0.15;
const REFERENCE_GAP  = 0.075;
const SIZE_CONTRAST  = 2.5;
const MAX_EYE_SIZE   = 0.3;
const DOT_SIZE       = 0.025;
const MIN_DOT_RADIUS = 1.5;

// TRACKING_AREA — the part of the camera image (0–1 on each axis) that is
// spread across the full projection. Faces rarely reach the very top or
// bottom of a camera's view, so narrowing y lets eyes travel further up and
// down the screen. Adjust on site once the camera is mounted.
const TRACKING_AREA = { left: 0, right: 1, top: 0, bottom: 1 };

// ── Gaze ─────────────────────────────────────────────────────────────────────
// Each pair of eyes is pushed from the person's standing position towards
// where they are looking. A webcam can only estimate this roughly, so these
// values are meant to be tuned on site with ?debug=1 (which draws a line from
// each person's standing position to their gaze point).
//
// GAZE_REACH      — how far a sideways look moves the eyes, as a fraction of
//                   screen width per unit of gaze. Turning the head about
//                   30° moves the eyes about 0.5 × GAZE_REACH of the screen
//                   width. At 1.2, a ~25° head turn — enough to read the
//                   edge of a laptop screen — carries the eyes from the
//                   centre to the edge.
// GAZE_REACH_Y    — the same for looking up and down, as a fraction of
//                   screen HEIGHT. Kept separate (and gentler) because
//                   vertical gaze from a webcam is noisier, and measuring it
//                   in screen widths made small up/down errors throw eyes
//                   off the top or bottom of the screen.
// EYE_GAZE_WEIGHT — how much the eyes moving within the head (without
//                   turning the head) adds. 0 = head direction only.
// IRIS_TRAVEL     — how far the iris centre moves from the middle of the eye
//                   at a full sideways glance, as a fraction of eye width.
// GAZE_OFFSET     — added to every gaze estimate. Use it to correct for the
//                   camera sitting above or beside the projection, so that
//                   looking straight ahead leaves the eyes where the person
//                   stands. y > 0 moves everyone's gaze down.
//
// Smoothing adapts to how much the gaze changes, so small jitter while
// someone holds still is smoothed heavily but a deliberate head turn is
// followed almost straight away (no "drag"):
// GAZE_STEADY_MS   — catch-up time when the gaze barely changes. Higher =
//                    steadier eyes while reading.
// GAZE_FAST_MS     — catch-up time for a big, deliberate change. Lower =
//                    snappier head turns.
// GAZE_FAST_CHANGE — how big a change (in gaze units, ~0.15 ≈ 9°) counts as
//                    fully deliberate. Smaller changes blend between the two.
const GAZE_REACH      = 1.2;
const GAZE_REACH_Y    = 1.0;
const EYE_GAZE_WEIGHT = 0.5;
const IRIS_TRAVEL     = 0.15;
const GAZE_OFFSET     = { x: 0, y: 0 };

const GAZE_STEADY_MS   = 250;
const GAZE_FAST_MS     = 40;
const GAZE_FAST_CHANGE = 0.15;

// ── Multiple people ──────────────────────────────────────────────────────────
// Everyone sits or stands differently relative to the webcam — higher,
// lower, tilted — which biases their up/down gaze differently. These keep
// each person's eyes in a sensible place regardless.
//
// VERTICAL_RECENTER_MS — each person's "looking straight ahead" height is
//                   learned separately and slowly follows their average
//                   up/down gaze over about this long. This cancels each
//                   person's own posture bias. 0 turns it off.
// STANDING_Y_WEIGHT — how much a person's head height in the camera image
//                   moves their eyes up or down the screen (0–1). Low, so
//                   someone taller or standing doesn't start at the top
//                   edge; sideways position still follows fully.
const VERTICAL_RECENTER_MS = 8000;
const STANDING_Y_WEIGHT    = 0.3;

// ── Iris movement ────────────────────────────────────────────────────────────
// Real eyes only move a few camera pixels when someone glances around, so
// the drawn iris dots are pushed further from the centre of each eye to make
// that movement visible. Only the iris dots move; the eye outline is drawn
// exactly as tracked.
//
// The exaggeration adapts to each person: some people's irises travel less
// within their eye (smaller or narrower eyes), so each person's typical iris
// movement is measured over the last few seconds and quieter irises get
// extra amplification — up to a firm cap so it never looks cartoonish.
//
// IRIS_AMPLIFY      — the usual exaggeration. 1 = as tracked; 3 = three
//                     times. Everyone gets at least this.
// IRIS_AMPLIFY_MAX  — the most a quiet-eyed person is ever boosted to.
// IRIS_TYPICAL_MOVE — typical average iris offset from the eye centre, as a
//                     fraction of eye width. People who move less than this
//                     are boosted in proportion (half the movement → up to
//                     double the amplification, within the cap).
// IRIS_MEASURE_MS   — how long each person's iris movement is averaged over.
// IRIS_MAX_X/Y      — how far the iris centre may be pushed from the eye
//                     centre, as a fraction of eye width, so it stays roughly
//                     inside the eye outline.
// IRIS_SMOOTH_MS    — light smoothing on the exaggerated movement so it
//                     doesn't also exaggerate jitter. Lower = snappier,
//                     higher = calmer.
const IRIS_AMPLIFY      = 3;
const IRIS_AMPLIFY_MAX  = 4;
const IRIS_TYPICAL_MOVE = 0.04;
const IRIS_MEASURE_MS   = 4000;
const IRIS_MAX_X        = 0.3;
const IRIS_MAX_Y        = 0.12;
const IRIS_SMOOTH_MS    = 80;

// ── Background faces ─────────────────────────────────────────────────────────
// People noticeably further back than the nearest person are drawn fainter.
// "Nearness" compares each person's eye size with the nearest person's.
//
// BACKGROUND_ALPHA — opacity of eyes far in the background (0–1).
// FOREGROUND_RATIO — people at least this big relative to the nearest
//                    person (0–1) count as foreground and stay fully opaque.
// BACKGROUND_RATIO — people this small or smaller are fully "background";
//                    in between, opacity blends smoothly.
const BACKGROUND_ALPHA = 0.45;
const FOREGROUND_RATIO = 0.75;
const BACKGROUND_RATIO = 0.35;

// Faces are matched between frames by position so each person's gaze can be
// smoothed separately. TRACK_MATCH_DISTANCE is how far (0–1 of the camera
// width) a face may move between draws and still count as the same person;
// TRACK_TIMEOUT_MS is how long a lost face is remembered.
const TRACK_MATCH_DISTANCE = 0.12;
const TRACK_TIMEOUT_MS     = 1000;

// ── Article scroll ───────────────────────────────────────────────────────────
// SCROLL_SPEED — how fast the background article moves up, in pixels per
// second. At 25, a line of text rises one line height about every 2.4 s on
// a laptop screen, so lines don't linger at the bottom edge for long.
const SCROLL_SPEED = 25;

// =============================================================================
// FACE REGION INDEX SETS
// =============================================================================
// Selected landmark indices for specific facial features, drawn in distinct
// colours so you can see which landmarks correspond to which regions.
// These indices come from the MediaPipe Face Mesh canonical face model.

// Each eye's outline, listed in order around the eye (lower lid from the
// outer corner, then upper lid back). Each iris is a centre point followed by four points on its edge.
const EYES = [
  {
    // Right eye (subject's right)
    outline: [33, 7, 163, 144, 145, 153, 154, 155, 133,
              173, 157, 158, 159, 160, 161, 246],
    irisCenter: 468,
    irisEdge: [469, 470, 471, 472],
    corners: [33, 133],        // outer, inner — used for gaze
    lids: [159, 145]           // upper, lower — used for gaze
  },
  {
    // Left eye (subject's left)
    outline: [362, 382, 381, 380, 374, 373, 390, 249, 263,
              466, 388, 387, 386, 385, 384, 398],
    irisCenter: 473,
    irisEdge: [474, 475, 476, 477],
    corners: [263, 362],
    lids: [386, 374]
  }
];

// Face landmarks used to work out which way the head is facing.
const HEAD_POINTS = {
  rightEdge: 234,   // right side of the face (subject's right)
  leftEdge:  454,   // left side of the face
  forehead:  10,
  chin:      152
};

const EYE_INDICES  = new Set(EYES.flatMap(eye => eye.outline));
const IRIS_INDICES = new Set(EYES.flatMap(eye => [eye.irisCenter, ...eye.irisEdge]));

// =============================================================================
// COLOUR PALETTES
// =============================================================================
// One palette per person, so visitors can pick out their own eyes. The
// colours are the ones surveillance and tracking systems use to mark
// individual subjects:
//   - NATO joint military symbology (APP-6) marks tracked subjects by
//     affiliation: blue = friendly, red = hostile, green = neutral,
//     yellow = unknown.
//   - Person-tracking and CCTV analytics software adds alert colours:
//     magenta = flagged, orange = person of interest, violet = identified.
//
//   eye  — eye-outline dots, in the subject's marking colour
//   iris — iris dots, in a contrasting colour from the same set so the
//          iris always stands out from the eye
//
// Palettes are handed out in this order. Anyone currently in view gets a
// palette no one else is using; with more people than palettes, colours
// start to repeat.
const PALETTES = [
  { name: "FRIENDLY",           eye: "#4cc9ff", iris: "#ff3b4e" },
  { name: "HOSTILE",            eye: "#ff3b4e", iris: "#ffe14d" },
  { name: "NEUTRAL",            eye: "#3dff7a", iris: "#ff3df2" },
  { name: "UNKNOWN",            eye: "#ffe14d", iris: "#4cc9ff" },
  { name: "FLAGGED",            eye: "#ff3df2", iris: "#3dff7a" },
  { name: "PERSON OF INTEREST", eye: "#ff8a1f", iris: "#4cc9ff" },
  { name: "IDENTIFIED",         eye: "#a97bff", iris: "#ffe14d" }
];

// =============================================================================
// ENTRY POINT
// =============================================================================

window.onload = async function () {

  if (debugMode) {
    console.log("Page loaded, initializing MediaPipe Face Landmarker demo...");
  }

  // ── DOM elements ────────────────────────────────────────────────────────

  const video  = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx    = canvas.getContext("2d");   // transparent: article shows through

  if (!video || !canvas) {
    console.error("Error: Could not find #video or #canvas in the DOM.");
    return;
  }

  // Lightweight status banner for mobile where DevTools is unavailable.
  const statusBanner = document.createElement("div");
  const statusHost = canvas.parentElement || document.body;
  if (statusHost !== document.body && getComputedStyle(statusHost).position === "static") {
    statusHost.style.position = "relative";
  }
  statusBanner.style.position = statusHost === document.body ? "fixed" : "absolute";
  statusBanner.style.left = "12px";
  statusBanner.style.top = "12px";
  statusBanner.style.zIndex = "9999";
  statusBanner.style.padding = "8px 10px";
  statusBanner.style.borderRadius = "8px";
  statusBanner.style.background = "rgba(0, 0, 0, 0.72)";
  statusBanner.style.color = "#f3f4f6";
  statusBanner.style.font = "12px/1.3 monospace";
  statusBanner.style.maxWidth = "min(80vw, 420px)";
  statusBanner.style.pointerEvents = "none";
  statusBanner.style.display = "none";
  statusHost.appendChild(statusBanner);

  function showStatus(message, isError = false) {
    statusBanner.textContent = message;
    statusBanner.style.display = "block";
    statusBanner.style.border = isError
      ? "1px solid rgba(248, 113, 113, 0.9)"
      : "1px solid rgba(156, 163, 175, 0.7)";
  }

  function clearStatus() {
    statusBanner.style.display = "none";
  }

  startArticleScroll();

  showStatus("Loading MediaPipe runtime...");

  // ── MediaPipe Tasks Vision ───────────────────────────────────────────────

  const { FaceLandmarker, FilesetResolver } = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs"
  );

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm"
  );

  // ── MediaPipe FaceLandmarker setup ───────────────────────────────────────
  // numFaces replaces maxNumFaces. Iris landmarks (indices 468–477) are
  // included automatically in the float16 model — no refineLandmarks option
  // is needed. The GPU delegate is tried first; CPU is the fallback.

  function landmarkerOptions(delegate) {
    return {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker" +
          "/face_landmarker/float16/latest/face_landmarker.task",
        delegate
      },
      runningMode: "VIDEO",
      numFaces: MAX_FACES,     // track several faces at once
      minFaceDetectionConfidence: MIN_DETECTION_CONFIDENCE,
      minFacePresenceConfidence:  MIN_PRESENCE_CONFIDENCE,
      minTrackingConfidence:      MIN_TRACKING_CONFIDENCE
    };
  }

  let faceLandmarker;
  try {
    showStatus(`Loading face model (${preferredDelegate})...`);
    faceLandmarker = await FaceLandmarker.createFromOptions(
      vision, landmarkerOptions(preferredDelegate)
    );
  } catch (delegateErr) {
    console.warn(`${preferredDelegate} delegate unavailable, retrying with ${fallbackDelegate}:`, delegateErr);
    showStatus(`Loading face model (${fallbackDelegate} fallback)...`);
    faceLandmarker = await FaceLandmarker.createFromOptions(
      vision, landmarkerOptions(fallbackDelegate)
    );
  }

  if (debugMode) {
    console.log("FaceLandmarker ready.");
  }

  // Store the most recent detection results so the render loop can read them.
  // result.faceLandmarks is an array of faces; each face is an array of 478
  // { x, y, z } objects (468 face points + 10 iris points).
  let faceResults = [];

  // Monotonically increasing timestamp required by detectForVideo.
  let lastTimestamp = -1;

  // ── Camera management ────────────────────────────────────────────────────

  // Holds the active MediaStream so we can stop it when switching cameras.
  let currentStream = null;

  // Controls whether the frame loop is running.
  let frameLoopActive = false;

  // Stop loops and release camera tracks when leaving the page.
  window.addEventListener("beforeunload", () => {
    frameLoopActive = false;
    if (currentStream) {
      currentStream.getTracks().forEach(t => t.stop());
    }
  });

  /**
   * startCamera — opens the webcam with an optional specific device and
   * starts the per-frame loop that feeds images to the FaceMesh model.
   *
   * @param {string} [deviceId] — exact device ID to open, or omit / pass ""
   *                              to let the browser choose the default camera.
   */
  async function startCamera(deviceId) {
    frameLoopActive = false;
    showStatus("Requesting camera access...");

    if (currentStream) {
      currentStream.getTracks().forEach(t => t.stop());
      currentStream = null;
    }

    // A higher resolution helps the model find smaller faces further from
    // the camera, which matters when tracking several people at once.
    const videoConstraints = {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: { ideal: "user" }
    };
    if (deviceId) videoConstraints.deviceId = { exact: deviceId };

    try {
      currentStream = await navigator.mediaDevices.getUserMedia(
        { video: videoConstraints }
      );
    } catch (err) {
      console.error("Could not open camera:", err);
      showStatus("Camera access failed. Check permissions and reload.", true);
      return;
    }

    video.srcObject = currentStream;
    video.muted = true;
    video.autoplay = true;
    video.setAttribute("playsinline", "");

    video.onloadedmetadata = () => {
      if (debugMode) {
        console.log(`Camera running at ${video.videoWidth}×${video.videoHeight}`);
      }
    };

    showStatus("Starting video playback...");
    try {
      await video.play();
    } catch (err) {
      console.error("Video playback failed:", err);
      showStatus("Video playback was blocked. Tap the page and reload.", true);
      return;
    }

    if (debugMode) {
      console.log("Webcam started.");
    }

    showStatus("Camera running. Starting tracking...");
    frameLoopActive = true;
    requestAnimationFrame(frameLoop);
    requestAnimationFrame(renderLoop);
  }

  // Time of the last draw, used to limit drawing to DRAW_FPS.
  let lastDrawTime = 0;

  /**
   * renderLoop — stamps the latest eyes onto the canvas DRAW_FPS times per
   * second, independent of the inference rate.
   */
  function renderLoop(now) {
    if (!frameLoopActive) return;
    if (now - lastDrawTime >= 1000 / DRAW_FPS) {
      lastDrawTime = now;
      drawFrame();
    }
    requestAnimationFrame(renderLoop);
  }

  /**
   * frameLoop — runs the FaceLandmarker on the current video frame on every
   * animation tick. detectForVideo is synchronous and returns results
   * immediately. The timestamp guard ensures the value always increases.
   */
  function frameLoop() {
    if (!frameLoopActive) return;
    if (video.readyState >= 2) {
      clearStatus();
      const ts = performance.now();
      if (ts > lastTimestamp) {
        lastTimestamp = ts;
        if (debugMode) {
          console.log("Running FaceLandmarker on current frame...");
        }
        const result = faceLandmarker.detectForVideo(video, ts);
        faceResults = result.faceLandmarks ?? [];
        if (debugMode) {
          console.log(`Detected ${faceResults.length} face(s).`);
        }
      }
    }
    requestAnimationFrame(frameLoop);
  }

  /**
   * populateCameraSelect — enumerates video-input devices and fills the
   * on-page <select>. Requires camera permission to have been granted so
   * that device labels are populated. The wrapper is revealed only when
   * more than one camera is available.
   *
   * @param {string} activeDeviceId — the deviceId currently in use.
   */
  async function populateCameraSelect(activeDeviceId) {
    const select  = document.getElementById("cameraSelect");
    const wrapper = document.getElementById("cameraSelectWrapper");
    if (!select || !wrapper) return;

    const devices     = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d => d.kind === "videoinput");

    select.innerHTML = "";
    videoInputs.forEach((device, i) => {
      const opt    = document.createElement("option");
      opt.value    = device.deviceId;
      opt.text     = device.label || `Camera ${i + 1}`;
      opt.selected = device.deviceId === activeDeviceId;
      select.appendChild(opt);
    });

    wrapper.style.display = videoInputs.length > 1 ? "flex" : "none";

    select.onchange = () => startCamera(select.value);
  }

  // Initialise the camera. Enumerate devices afterwards so that the camera
  // selector shows real labels (requires permission to have been granted).
  await startCamera();
  const track    = currentStream ? currentStream.getVideoTracks()[0] : null;
  const activeId = track ? track.getSettings().deviceId : "";
  populateCameraSelect(activeId);

  // ── Controls ─────────────────────────────────────────────────────────────
  // F toggles fullscreen (for the projector); any other key wipes the trails.
  // Clicking saves the eye layer as a PNG.

  window.addEventListener("keydown", event => {
    if (event.key === "f" || event.key === "F") {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen().catch(() => {});
      return;
    }
    clearCanvas();
  });

  canvas.addEventListener("click", () => {
    // Composite the transparent eye layer onto black so the PNG isn't empty
    // where there are no eyes.
    const out = document.createElement("canvas");
    out.width  = canvas.width;
    out.height = canvas.height;
    const outCtx = out.getContext("2d");
    outCtx.fillStyle = BACKGROUND_COLOR;
    outCtx.fillRect(0, 0, out.width, out.height);
    outCtx.drawImage(canvas, 0, 0);

    out.toBlob(blob => {
      if (!blob) return;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `eyeTrail-${timestampString()}.png`;
      link.click();
      URL.revokeObjectURL(link.href);
    }, "image/png");
  });

  /**
   * timestampString — current local time as YYYYMMDD-HHMMSS, for filenames.
   */
  function timestampString() {
    const d   = new Date();
    const pad = n => String(n).padStart(2, "0");
    return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-` +
           `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  }

  // Debug counter lives in the page rather than on the canvas so it doesn't
  // smear into the trail.
  const debugCounter = document.createElement("div");
  if (debugMode) {
    debugCounter.style.cssText =
      "position:fixed;left:12px;bottom:12px;z-index:10;" +
      "font:14px monospace;color:#888;pointer-events:none;";
    document.body.appendChild(debugCounter);
  }

  // ── Drawing ──────────────────────────────────────────────────────────────

  /**
   * clearCanvas — erases all eyes and trails, leaving the canvas transparent.
   */
  function clearCanvas() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  /**
   * drawFrame — fades the existing trails, then stamps the latest eyes for
   * every detected face on top. The canvas is never fully wiped (unless
   * TRAIL_FADE is 1), which is what produces the trail.
   */
  function drawFrame() {
    const dpr = window.devicePixelRatio || 1;
    const w   = window.innerWidth;
    const h   = window.innerHeight;

    // Keep the canvas backing store in sync with the window size. Rendering
    // at devicePixelRatio keeps the dots sharp on high-density screens.
    if (canvas.width  !== Math.round(w * dpr) ||
        canvas.height !== Math.round(h * dpr)) {
      canvas.width  = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Fade old trails towards transparent (rather than towards black) so the
    // article underneath stays clean.
    if (TRAIL_FADE > 0) {
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = `rgba(0, 0, 0, ${TRAIL_FADE})`;
      ctx.fillRect(0, 0, w, h);
      ctx.globalCompositeOperation = "source-over";
    }

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;

    const now = performance.now();

    // Forget people who have left the camera's view, then pair each face
    // with its person's track from previous draws.
    gazeTracks = gazeTracks.filter(t => now - t.seen < TRACK_TIMEOUT_MS);
    const tracks = matchTracks(faceResults, now);

    const faces = faceResults
      .map((landmarks, i) => placeFace(landmarks, tracks[i], vw, vh, w, h, now))
      .filter(Boolean)
      // Draw far-away (small) faces first so nearer faces sit on top.
      .sort((a, b) => a.eyeGap - b.eyeGap);

    // The nearest person sets what counts as "background".
    const nearestGap = faces.length ? faces[faces.length - 1].eyeGap : 0;

    if (debugMode) {
      debugCounter.textContent =
        `Faces: ${faces.length}  Eyes: ${faces.length * 2}` +
        faces.map((f, i) =>
          `  |  #${i + 1} gaze ${f.gaze.x.toFixed(2)}, ${f.gaze.y.toFixed(2)}`
        ).join("");
    }

    faces.forEach((face, faceIndex) => {
      if (debugMode) drawGazeGuide(face);
      drawNodes(face, faceOpacity(face.eyeGap, nearestGap));

      if (debugMode) {
        console.log(`Face ${faceIndex + 1}: gaze ${face.gaze.x.toFixed(2)}, ${face.gaze.y.toFixed(2)}`);
      }
    });
  }

  // One entry per person currently in view: { x, y } is their last position
  // in the camera image, { gx, gy } their smoothed gaze, `baseY` their
  // learned "straight ahead" height, `seen` when they were last matched.
  let gazeTracks = [];

  /**
   * pickPalette — the palette used by the fewest people currently in view
   * (in list order on a tie), so people in view never share a colour until
   * there are more people than palettes.
   */
  function pickPalette() {
    const counts = PALETTES.map(p => gazeTracks.filter(t => t.palette === p).length);
    return PALETTES[counts.indexOf(Math.min(...counts))];
  }

  /**
   * faceAnchor — midpoint between the two iris centres, in camera 0–1
   * coordinates, or null if the iris landmarks are missing.
   */
  function faceAnchor(landmarks) {
    const right = landmarks[EYES[0].irisCenter];
    const left  = landmarks[EYES[1].irisCenter];
    if (!right || !left) return null;
    return { x: (right.x + left.x) / 2, y: (right.y + left.y) / 2 };
  }

  /**
   * matchTracks — pairs each detected face with the track of the same person
   * from earlier draws. MediaPipe doesn't keep faces in a fixed order, so
   * faces are matched by position: closest face–track pairs are taken first,
   * so two people side by side can't steal each other's track. Faces with no
   * track nearby start a new one.
   *
   * @returns {Array} one track per face (null if the face has no anchor).
   */
  function matchTracks(faceList, now) {
    const anchors = faceList.map(faceAnchor);

    const pairs = [];
    anchors.forEach((a, i) => {
      if (!a) return;
      gazeTracks.forEach((t, j) => {
        const d = Math.hypot(t.x - a.x, t.y - a.y);
        if (d < TRACK_MATCH_DISTANCE) pairs.push({ i, j, d });
      });
    });
    pairs.sort((p, q) => p.d - q.d);

    const result    = faceList.map(() => null);
    const takenTrack = new Set();
    pairs.forEach(({ i, j }) => {
      if (result[i] || takenTrack.has(j)) return;
      result[i] = gazeTracks[j];
      takenTrack.add(j);
    });

    anchors.forEach((a, i) => {
      if (!a || result[i]) return;
      const track = {
        x: a.x, y: a.y, gx: null, gy: null, baseY: null, seen: now,
        palette: pickPalette()
      };
      gazeTracks.push(track);
      result[i] = track;
    });

    return result;
  }

  /**
   * updateGaze — eases a person's track towards a new gaze estimate and
   * returns their smoothed gaze, with their own vertical bias removed.
   *
   * The easing is time-based and adaptive: the catch-up time slides from
   * GAZE_STEADY_MS for tiny changes down to GAZE_FAST_MS for big ones, so
   * jitter is absorbed but real head turns are followed immediately.
   */
  function updateGaze(track, anchor, gaze, now) {
    const dt = now - track.seen;

    if (track.gx === null) {
      // First sighting: start from the raw estimate, and treat the height
      // they are looking at now as their "straight ahead".
      track.gx    = gaze.x;
      track.gy    = gaze.y;
      track.baseY = gaze.y;
    } else {
      const change = Math.hypot(gaze.x - track.gx, gaze.y - track.gy);
      const t      = clamp(change / GAZE_FAST_CHANGE, 0, 1);
      const tau    = GAZE_STEADY_MS + (GAZE_FAST_MS - GAZE_STEADY_MS) * t;
      const ease   = 1 - Math.exp(-dt / tau);
      track.gx += (gaze.x - track.gx) * ease;
      track.gy += (gaze.y - track.gy) * ease;

      if (VERTICAL_RECENTER_MS > 0) {
        track.baseY += (track.gy - track.baseY) * (1 - Math.exp(-dt / VERTICAL_RECENTER_MS));
      }
    }

    track.x    = anchor.x;
    track.y    = anchor.y;
    track.seen = now;

    return {
      x: track.gx,
      y: VERTICAL_RECENTER_MS > 0 ? track.gy - track.baseY : track.gy
    };
  }

  /**
   * irisShifts — for each eye, how far (in camera pixels) to move its iris
   * dots so the iris's offset from the eye centre is multiplied by
   * IRIS_AMPLIFY, capped at IRIS_MAX_X/Y of the eye width and lightly
   * smoothed per person.
   *
   * @returns {Map} landmark index → { x, y } shift, for every iris landmark.
   */
  function irisShifts(landmarks, track, vw, vh, now) {
    const shifts = new Map();
    const dt     = track.irisTime ? now - track.irisTime : Infinity;
    const ease   = 1 - Math.exp(-dt / IRIS_SMOOTH_MS);
    track.irisTime   = now;
    track.irisShifts = track.irisShifts || EYES.map(() => ({ x: 0, y: 0 }));
    track.irisMove   = track.irisMove   || IRIS_TYPICAL_MOVE;

    // Personal amplification: quieter irises get boosted, within the cap.
    const amplify = clamp(
      IRIS_AMPLIFY * IRIS_TYPICAL_MOVE / track.irisMove,
      IRIS_AMPLIFY,
      IRIS_AMPLIFY_MAX
    );
    let moveSum = 0;
    let moveCount = 0;

    EYES.forEach((eye, k) => {
      const iris = landmarks[eye.irisCenter];
      const [c1, c2] = eye.corners.map(i => landmarks[i]);
      if (!iris || !c1 || !c2) return;

      // Eye centre and width in camera pixels.
      const cx    = (c1.x + c2.x) / 2 * vw;
      const cy    = (c1.y + c2.y) / 2 * vh;
      const width = Math.hypot((c2.x - c1.x) * vw, (c2.y - c1.y) * vh) || 1;

      // Where the iris is, and where the exaggerated iris should be.
      const ox = iris.x * vw - cx;
      const oy = iris.y * vh - cy;
      const tx = clamp(ox * amplify, -IRIS_MAX_X * width, IRIS_MAX_X * width);
      const ty = clamp(oy * amplify, -IRIS_MAX_Y * width, IRIS_MAX_Y * width);

      moveSum += Math.abs(ox) / width;
      moveCount++;

      const smoothed = track.irisShifts[k];
      smoothed.x += (tx - ox - smoothed.x) * ease;
      smoothed.y += (ty - oy - smoothed.y) * ease;

      [eye.irisCenter, ...eye.irisEdge].forEach(i => shifts.set(i, smoothed));
    });

    // Update this person's average sideways iris movement (skipped on the
    // first sighting so one glance doesn't set it).
    if (moveCount > 0 && dt !== Infinity) {
      const settle = 1 - Math.exp(-dt / IRIS_MEASURE_MS);
      track.irisMove += (moveSum / moveCount - track.irisMove) * settle;
    }

    return shifts;
  }

  /**
   * drawGazeGuide — debug only: a ring where the person stands and a line to
   * where their eyes are drawn, for tuning the gaze settings on site.
   */
  function drawGazeGuide({ standing, screenAnchor }) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(standing.x, standing.y, 6, 0, Math.PI * 2);
    ctx.moveTo(standing.x, standing.y);
    ctx.lineTo(screenAnchor.x, screenAnchor.y);
    ctx.stroke();
  }

  /**
   * placeFace — works out where on the projection a face's eyes go and how
   * big they are.
   *
   * Where: the anchor is the midpoint between the two iris centres. Its
   * position in the camera image (within TRACKING_AREA) is stretched across
   * the full window — that is where the person stands. The smoothed gaze
   * estimate then pushes it towards where they are looking.
   *
   * How big: the person's eye gap in the camera (larger when closer) sets
   * the on-screen eye gap via EYE_SIZE and SIZE_CONTRAST. Each landmark is
   * then drawn at the anchor plus its camera offset times `scale`, so nearer
   * visitors get bigger eyes and irises, and people further back get much
   * smaller ones.
   *
   * The eyes are kept fully on screen: the anchor is clamped with a margin
   * based on the eye size, so no one's eyes get pinned half off an edge.
   *
   * @param {Array}  landmarks - Array of { x, y, z } landmark objects.
   * @param {Object} track     - This person's track, from matchTracks.
   * @param {number} vw, vh    - Camera frame size in pixels.
   * @param {number} w, h      - Window size in CSS pixels.
   * @param {number} now       - Current time, for gaze smoothing.
   * @returns {Object|null} { landmarks, toPixel, eyeGap, dotRadius,
   *                          gaze, standing, screenAnchor }
   */
  function placeFace(landmarks, track, vw, vh, w, h, now) {
    const anchor = faceAnchor(landmarks);
    if (!anchor || !track) return null;
    const right = landmarks[EYES[0].irisCenter];
    const left  = landmarks[EYES[1].irisCenter];

    // Anchor position within the tracking area, 0–1 on each axis.
    const areaW = TRACKING_AREA.right  - TRACKING_AREA.left;
    const areaH = TRACKING_AREA.bottom - TRACKING_AREA.top;
    let ax = (anchor.x - TRACKING_AREA.left) / areaW;
    let ay = (anchor.y - TRACKING_AREA.top)  / areaH;
    if (MIRROR) ax = 1 - ax;

    // Sideways position follows where they stand fully; head height only
    // nudges the eyes up or down from the middle of the screen.
    const standing = {
      x: ax * w,
      y: h / 2 + (ay - 0.5) * h * STANDING_Y_WEIGHT
    };
    const mirror = MIRROR ? -1 : 1;

    // On-screen eye gap: EYE_SIZE at the reference distance, shrinking
    // steeply (SIZE_CONTRAST) as the person's camera eye gap gets smaller.
    const cameraGap = Math.hypot((right.x - left.x) * vw, (right.y - left.y) * vh);
    const eyeGap    = Math.min(
      EYE_SIZE * w * Math.pow(cameraGap / (REFERENCE_GAP * vw), SIZE_CONTRAST),
      MAX_EYE_SIZE * w
    );

    // Shift towards the gaze point. Gaze is in camera-image directions, so
    // mirror x the same way as the position. The margins keep the whole
    // pair of eyes on screen.
    const gaze    = updateGaze(track, anchor, estimateGaze(landmarks, vw, vh), now);
    const marginX = Math.min(eyeGap * 0.8, w / 2);
    const marginY = Math.min(eyeGap * 0.3, h / 2);
    const screenAnchor = {
      x: clamp(standing.x + mirror * gaze.x * GAZE_REACH   * w, marginX, w - marginX),
      y: clamp(standing.y +          gaze.y * GAZE_REACH_Y * h, marginY, h - marginY)
    };

    // Camera pixels → screen pixels: every landmark offset is multiplied by
    // the same factor, so the eye shape is kept, just resized.
    const scale = eyeGap / (cameraGap || 1);

    // Extra shift (camera pixels) for each eye's iris dots, exaggerating
    // where the iris sits within the eye. Keyed by landmark index.
    const irisShift = irisShifts(landmarks, track, vw, vh, now);

    const toPixel = index => {
      const p     = landmarks[index];
      const shift = irisShift.get(index);
      const dx    = (p.x - anchor.x) * vw + (shift ? shift.x : 0);
      const dy    = (p.y - anchor.y) * vh + (shift ? shift.y : 0);
      return {
        x: screenAnchor.x + mirror * dx * scale,
        y: screenAnchor.y +          dy * scale
      };
    };

    // Dot size follows the on-screen distance between the eyes.
    const dotRadius = Math.max(MIN_DOT_RADIUS, eyeGap * DOT_SIZE);

    return {
      landmarks, toPixel, eyeGap, dotRadius, gaze, standing, screenAnchor,
      palette: track.palette
    };
  }

  /**
   * faceOpacity — full opacity for people about as near as the nearest
   * person, fading smoothly to BACKGROUND_ALPHA for people much further back.
   */
  function faceOpacity(eyeGap, nearestGap) {
    if (!nearestGap) return 1;
    const ratio = eyeGap / nearestGap;
    const t = clamp(
      (ratio - BACKGROUND_RATIO) / (FOREGROUND_RATIO - BACKGROUND_RATIO), 0, 1
    );
    return BACKGROUND_ALPHA + (1 - BACKGROUND_ALPHA) * t;
  }

  /**
   * drawNodes — draws a coloured dot at each eye and iris landmark of one
   * placed face.
   *
   * @param {Object} face - Result of placeFace.
   */
  function drawNodes({ landmarks, toPixel, dotRadius, palette }, opacity = 1) {
    ctx.globalAlpha = opacity;

    // Group dots by colour so each colour is drawn in a single fill call.
    const groups = [
      { color: palette.eye,  indices: EYE_INDICES },
      { color: palette.iris, indices: IRIS_INDICES }
    ];

    groups.forEach(({ color, indices }) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      indices.forEach(index => {
        if (index >= landmarks.length) return;
        const { x, y } = toPixel(index);
        ctx.moveTo(x + dotRadius, y);
        ctx.arc(x, y, dotRadius, 0, Math.PI * 2);
      });
      ctx.fill();
    });
    ctx.globalAlpha = 1;
  }
};

// =============================================================================
// ARTICLE SCROLL
// =============================================================================

/**
 * startArticleScroll — moves the background article up the screen at
 * SCROLL_SPEED pixels per second, forever. A copy of the article is placed
 * directly underneath the original; when the first copy has scrolled fully
 * out of view the position jumps back by exactly one article height, which
 * looks identical, so the loop is seamless.
 */
function startArticleScroll() {
  const track   = document.getElementById("articleTrack");
  const article = document.getElementById("article");
  if (!track || !article) return;

  const copy = article.cloneNode(true);
  copy.removeAttribute("id");
  copy.setAttribute("aria-hidden", "true");
  track.appendChild(copy);

  let offset   = 0;
  let lastTime = performance.now();

  function step(now) {
    // Cap the time step so a backgrounded tab doesn't jump on return.
    const dt = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;

    offset += SCROLL_SPEED * dt;
    const loopHeight = article.offsetHeight;
    if (loopHeight > 0 && offset >= loopHeight) offset -= loopHeight;

    track.style.transform = `translateY(${-offset}px)`;
    requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}

// =============================================================================
// GAZE ESTIMATION
// =============================================================================

/**
 * estimateGaze — a rough estimate of where a face is looking, as a 2-D
 * direction in camera-image terms: x > 0 is towards the right of the camera
 * image, y > 0 is downwards. About 0.5 corresponds to a 30° look.
 *
 * It adds two parts:
 *   Head direction — a 3-D "facing" vector built from MediaPipe's landmark
 *     depths: the cross product of a left-to-right vector across the face and
 *     a chin-to-forehead vector. This is the more reliable part.
 *   Eye direction — where each iris centre sits between the eye corners
 *     (sideways) and between the eyelids (up/down), averaged over both eyes.
 *     This catches glances without a head turn, but is noisier.
 *
 * @param {Array}  lm     - Array of { x, y, z } landmark objects.
 * @param {number} vw, vh - Camera frame size in pixels.
 * @returns {{x: number, y: number}}
 */
function estimateGaze(lm, vw, vh) {
  // Landmarks in camera pixels. MediaPipe's z uses roughly the same scale as
  // x, so it is multiplied by the frame width.
  const P = i => ({ x: lm[i].x * vw, y: lm[i].y * vh, z: lm[i].z * vw });

  // ── Head direction ──
  const across = subtract(P(HEAD_POINTS.leftEdge), P(HEAD_POINTS.rightEdge));
  const up     = subtract(P(HEAD_POINTS.forehead), P(HEAD_POINTS.chin));
  const facing = normalize(cross(across, up));   // points out of the face

  // ── Eye direction ──
  let eyeX = 0;
  let eyeY = 0;
  EYES.forEach(eye => {
    const iris  = P(eye.irisCenter);
    const [c1, c2] = eye.corners.map(P);
    const [lidTop, lidBottom] = eye.lids.map(P);
    const width = Math.hypot(c2.x - c1.x, c2.y - c1.y) || 1;
    eyeX += (iris.x - (c1.x + c2.x) / 2) / width;
    eyeY += (iris.y - (lidTop.y + lidBottom.y) / 2) / width;
  });
  eyeX /= EYES.length;
  eyeY /= EYES.length;

  return {
    x: facing.x + clamp(eyeX / IRIS_TRAVEL, -1, 1) * EYE_GAZE_WEIGHT + GAZE_OFFSET.x,
    y: facing.y + clamp(eyeY / IRIS_TRAVEL, -1, 1) * EYE_GAZE_WEIGHT + GAZE_OFFSET.y
  };
}

// ── Small vector helpers ─────────────────────────────────────────────────────

function subtract(a, b) {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function cross(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x
  };
}

function normalize(v) {
  const len = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
