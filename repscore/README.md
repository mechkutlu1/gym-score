# RepScore

A browser-based rep counter and movement-form scorer for gym exercises — squat, push-up, pull-up and dumbbell curl — built on the MediaPipe Pose Landmarker (Tasks Vision, JavaScript). All inference runs on-device in the browser; no video ever leaves the phone. Installable as a PWA.

## How it works

1. **Pose estimation.** The MediaPipe `pose_landmarker_lite` model runs in `VIDEO` mode on each camera frame and returns 33 body landmarks with visibility scores.
2. **Joint angle.** Each exercise tracks one joint angle from three landmarks (e.g. hip–knee–ankle for the squat, shoulder–elbow–wrist for the others). The side with higher landmark visibility is selected automatically, and the angle is smoothed with an exponential moving average (α = 0.35).
3. **Rep detection.** A hysteresis state machine (`extended → flexed → extended`) counts a rep only on a full cycle, with separate enter/exit thresholds to reject jitter. Pull-ups additionally require the wrists above the shoulders before tracking starts.
4. **Form score.** Each rep is scored 0–100: 60 points for depth (how far the minimum angle progresses from the flexion threshold towards the exercise's "perfect bottom" angle) and 40 points for lockout (maximum angle towards full extension). A cue is issued for shallow depth or missing lockout.

Thresholds per exercise are defined in `EXERCISES` at the top of `app.js` and are straightforward to tune.

## Publish on GitHub Pages

1. Create a new repository (e.g. `repscore`) and push these files to the root of the `main` branch.
2. In the repository: **Settings → Pages → Source: Deploy from a branch → Branch: `main` / `(root)` → Save**.
3. After a minute the app is live at `https://<username>.github.io/repscore/`.

GitHub Pages serves over HTTPS, which is required for `getUserMedia` camera access. On a phone, open the URL, allow camera access, and use "Add to Home Screen" to install it as an app.

## Files

| File | Purpose |
|---|---|
| `index.html` | UI shell |
| `style.css` | Styling (accent colour follows the competition plate colour of the selected lift) |
| `app.js` | Pose pipeline, rep state machine, scoring, rendering |
| `manifest.json`, `sw.js`, `icon-*.png` | PWA install + offline app shell |

## Notes and limitations

- Camera placement matters: side-on for squats and push-ups, front-on for pull-ups and curls, whole body in frame, 2–4 m away.
- The lite model favours frame rate on mid-range phones; swap the `modelAssetPath` to `pose_landmarker_full` or `heavy` for higher accuracy on desktop.
- Angles are computed in the image plane, so severe camera obliquity biases the measured angle — a known limitation of 2-D keypoint methods.
- The MediaPipe WASM runtime and model are loaded from CDN on first run and cached by the service worker thereafter.
