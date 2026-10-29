# MediaPipe Eye Tracking

A projection piece for an exhibition installation about surveillance. Visitors walk into a dark room where an article scrolls slowly up a large projection. As they read, a camera tracks their eyes and draws them over the text, so every reader sees themselves being watched. It runs entirely in the browser using [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) Face Landmarker, adapted from the MediaPipe Demos project in the MDes Prototyping course at CCA.

```
webcam → Face Landmarker → eye + iris landmarks → placed + scaled per person → drawn over the scrolling article
```

**What you see:**
- The article “Flock Cameras Can Track Every Car in America. Police Love Them. Citizens Don’t.” (The New York Times, Aug. 10, 2026), laid out in the newspaper’s format and scrolling slowly upwards on a seamless loop, fading out at the top and bottom edges
- Every visitor's eyes drawn over the text in their own colour palette, so people can tell which eyes are theirs
- Each pair of eyes placed where that person stands (mirrored, and spread across the whole projection), then shifted towards where they are looking
- Each pair of eyes and irises sized by distance — people standing closer get bigger eyes, and nearer faces draw on top
- A short fading trail as the eyes move (about 1.5 seconds), stamped 10 times per second
- Up to 10 faces (20 eyes) tracked at once
- The webcam feed itself is never shown

📂 **Sketch:** [`demo-2-face-mesh/sketch.js`](demo-2-face-mesh/sketch.js)

---

## Running the Demo

Webcam access requires the page to be served over HTTPS or from `localhost` — opening `index.html` directly as a `file://` URL will not work.

### Option 1: VS Code Live Server

Click **Go Live** in the VS Code status bar. Live Server opens the repo root on port 5501, and the root `index.html` forwards to the demo.

### Option 2: Vite dev server

```sh
npm install
npm run dev
```

Vite opens `http://localhost:5500/` and the root page forwards to the demo.

### Option 3: Any static server

From the repository root:

```sh
python3 -m http.server 5500
```

Then open `http://localhost:5500/demo-2-face-mesh/`.

### Option 4: HTTPS on a phone

Mobile browsers block webcam access on plain `http://192.168.x.x` URLs. To test on a phone, run the local server with an HTTPS tunnel:

```sh
npm install
npm run dev:https
```

Open the `https://*.loca.lt` URL from the terminal output on your phone.

### Option 5: OpenProcessing

1. Create a new sketch on [OpenProcessing](https://openprocessing.org/).
2. Paste the HTML from [`demo-2-face-mesh/index.html`](demo-2-face-mesh/index.html) and the JavaScript from [`demo-2-face-mesh/sketch.js`](demo-2-face-mesh/sketch.js).
3. Run the sketch and allow webcam access.

---

## Controls and Settings

**Controls:**
- **F** — toggle fullscreen (use this on the projector)
- Any other key — clear the eye trails
- Click — save the eye layer as a PNG on black (`eyeTrail-YYYYMMDD-HHMMSS.png`); the article text is not included
- Hover the top-right corner — reveals the camera picker (only when more than one camera is connected)

**The article:** edit the `<article>` in [`demo-2-face-mesh/index.html`](demo-2-face-mesh/index.html). Text size and colour are in the `.article` styles in the same file.

**Settings at the top of `sketch.js`:**

| Setting | What it does |
|---|---|
| `SCROLL_SPEED` | Article scroll speed in pixels per second (default 25) |
| `EYE_SIZE` | Eye size for someone at laptop-webcam distance, as a fraction of screen width (default 0.15) |
| `SIZE_CONTRAST` | How sharply eyes shrink with distance: 1 = in proportion, 2.5 (default) = twice as far gives about a fifth of the size |
| `REFERENCE_GAP`, `MAX_EYE_SIZE` | The distance `EYE_SIZE` applies to, and a size cap for someone leaning into the camera |
| `DOT_SIZE`, `MIN_DOT_RADIUS` | Dot size relative to the eyes, and the smallest allowed dot |
| `TRACKING_AREA` | The part of the camera image spread across the projection. Narrow `top`/`bottom` so eyes travel further up and down the screen |
| `DRAW_FPS` | How often eyes are stamped (default 10) |
| `TRAIL_FADE` | Trail length: higher = shorter; 0 = never fades, 1 = no trail |
| `MIRROR` | Mirror the view so moving left moves the eyes left |
| `MAX_FACES` | Maximum faces tracked at once (default 10) |
| `IRIS_AMPLIFY` | How much the drawn iris movement inside each eye is exaggerated (default 3; 1 = as tracked) |
| `IRIS_AMPLIFY_MAX`, `IRIS_TYPICAL_MOVE` | People whose irises move less than typical (e.g. smaller eyes) are boosted automatically, up to this cap (default 4) |
| `BACKGROUND_ALPHA` | Opacity of eyes for people well behind the nearest person (default 0.45) |
| `FOREGROUND_RATIO`, `BACKGROUND_RATIO` | Eye size relative to the nearest person at which eyes start fading, and are fully faded |
| `IRIS_MAX_X`, `IRIS_MAX_Y` | How far the iris may be pushed from the eye centre, so it stays inside the eye |
| `IRIS_SMOOTH_MS` | Smoothing on the exaggerated iris movement; lower = snappier, higher = calmer |
| `PALETTES` | The per-person colour palettes (see below) |
| `GAZE_REACH` | How far looking left/right moves the eyes, in screen widths (default 1.2) |
| `GAZE_REACH_Y` | How far looking up/down moves the eyes, in screen heights (default 1.0) |
| `VERTICAL_RECENTER_MS` | Each person's "straight ahead" height is learned separately over about this long, cancelling their own posture bias (default 8000; 0 = off) |
| `STANDING_Y_WEIGHT` | How much head height in the camera moves eyes up/down (default 0.3), so taller or standing people don't start at the top edge |
| `EYE_GAZE_WEIGHT` | How much glancing with the eyes alone (no head turn) counts; 0 = head direction only |
| `IRIS_TRAVEL` | How far the iris moves within the eye at a full glance (fraction of eye width) |
| `GAZE_OFFSET` | Shifts everyone's gaze, to correct for camera placement so looking straight ahead leaves the eyes where the person stands |
| `GAZE_STEADY_MS`, `GAZE_FAST_MS` | Adaptive smoothing: catch-up time while holding still (higher = steadier) and for a deliberate head turn (lower = snappier) |
| `MIN_*_CONFIDENCE` | Detection thresholds; lower finds more faces, higher avoids false detections |

**Colour palettes:** each person gets a palette, kept for as long as they're tracked. The colours are the ones surveillance and tracking systems use to mark individual subjects: NATO military symbology's affiliation colours (blue friendly, red hostile, green neutral, yellow unknown), plus the alert colours of person-tracking software (magenta flagged, orange person of interest, violet identified). Each iris uses a contrasting colour from the same set. People in view never share a palette until there are more than seven people.

| Palette | Eye outline | Iris |
|---|---|---|
| FRIENDLY | blue `#4cc9ff` | red `#ff3b4e` |
| HOSTILE | red `#ff3b4e` | yellow `#ffe14d` |
| NEUTRAL | green `#3dff7a` | magenta `#ff3df2` |
| UNKNOWN | yellow `#ffe14d` | blue `#4cc9ff` |
| FLAGGED | magenta `#ff3df2` | green `#3dff7a` |
| PERSON OF INTEREST | orange `#ff8a1f` | blue `#4cc9ff` |
| IDENTIFIED | violet `#a97bff` | yellow `#ffe14d` |

**Tuning gaze on site:** open the page with `?debug=1`. Each person gets a small ring where they stand and a line to where their eyes are drawn, and the bottom-left readout shows every person's gaze numbers. Have someone stand still and look straight ahead: if the line doesn't stay short, adjust `GAZE_OFFSET` (for example, a camera mounted above the projection makes everyone appear to look down, so use a negative `y`). Then have them look at the screen's edges and adjust `GAZE_REACH` until the eyes land roughly where they look.

**Installation tips:**
- Mount the camera at eye height near the projection, facing the readers. The face model works best within about 2 metres.
- Light the visitors' faces softly (for example from the projection itself or a dim front light). A completely dark room hides faces from the camera.
- Run Chrome in kiosk mode (`--kiosk`) or press **F** for fullscreen.

---

## How It Works

The Face Landmarker model (from `@mediapipe/tasks-vision`) returns 478 landmarks for each detected face: 468 face-surface points plus 10 iris points. The model always returns the full set; the sketch draws only the eye and iris points and skips the rest.

Each landmark has normalised `x`, `y`, and `z` coordinates (0.0–1.0 relative to the frame). To draw on a canvas, convert to pixels:

```js
const pixelX = landmark.x * canvas.width;
const pixelY = landmark.y * canvas.height;
```

The eye and iris points are picked out by index:

```js
const EYE_INDICES  = new Set([33, 7, 163, ...]);   // eye outlines
const IRIS_INDICES = new Set([468, 469, ..., 477]); // irises

if      (IRIS_INDICES.has(index)) { /* draw in pink-red */ }
else if (EYE_INDICES.has(index))  { /* draw in light blue */ }
```

| Indices | Region |
|---|---|
| 468–472 | Right iris (468 is the centre) |
| 473–477 | Left iris (473 is the centre) |
| 33, 133, 7, 163, 144, 145, 153, 154, 155, 173, 157, 158, 159, 160, 161, 246 | Right eye outline |
| 263, 362, 382, 381, 380, 374, 373, 390, 249, 466, 388, 387, 386, 385, 384, 398 | Left eye outline |

"Right" and "left" are from the subject's point of view, so the right eye appears on the left of an unmirrored webcam image.

The sketch runs two loops: `frameLoop` calls `faceLandmarker.detectForVideo()` on each frame, and `renderLoop` redraws the canvas at display refresh rate using the latest results.

> **Tip:** The iris centre (468 or 473) relative to the eye corners (33/133 or 362/263) tells you where someone is looking horizontally. The distance between the upper and lower eyelid points (159/145 or 386/374) tells you whether the eye is open or closed.

---

## Folder Structure

```
index.html          ← root page; forwards to the demo
demo-2-face-mesh/
├── index.html      ← page: the article, eye canvas, camera selector
└── sketch.js       ← all demo logic
package.json        ← Vite dev scripts
vite.config.js      ← dev server config
```

---

## Usage Notes

1. The MediaPipe library and model are loaded from a CDN, so an internet connection is required.
2. MediaPipe runs entirely client-side. No video data is sent to any server.
3. For best performance, use Chrome or Edge.
4. Phones default to the CPU delegate and desktops to GPU. Add `?delegate=cpu` or `?delegate=gpu` to the URL to override.
5. If you have more than one camera, a selector appears in the top-right corner.
6. Add `?debug=1` to the URL (or set `debugMode = true` in `sketch.js`) to show a faces/eyes counter and log detection info to the browser console.

---

## Troubleshooting

1. **No webcam prompt or blank video:** make sure you're opening the demo via `http://localhost:<port>/...`, not `file://...`.
2. **Camera blocked:** click the camera icon in the browser address bar, allow access, then reload.
3. **Old behaviour after code changes:** hard refresh (`Cmd+Shift+R` on Mac, `Ctrl+F5` on Windows).
4. **No dots appear:** the face must be clearly visible and reasonably well lit. Add `?debug=1` to the URL to see how many faces are being detected.
5. **Some people are missed:** the face model is built for faces within about 2 metres of the camera, facing it. Move closer, add light, or lower the `MIN_*_CONFIDENCE` values in `sketch.js`.

---

## License

MIT — see [LICENSE](LICENSE) in the root of this repository.
