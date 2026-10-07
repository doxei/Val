# PrintOS / PrinterAgent: audit for the Valdar successor module

Audited copy: `/mnt/user-data/uploads/klipper_brain/` (read-only). Audit date: 2026-10-07.
The SQLite DB was inspected from a copy in the scratch directory, using `python3 -I`.

Legend: **[FACT]** means it is directly in the code, logs or DB. **[INFERRED]** is my reasoning from indirect evidence. **[UNSURE]** means it could not be verified from the files provided.

---

## 0. Scope and caveats (read first)

1. **The upload is incomplete.** Modules that are imported but absent:
   - `config.py`, which defines `SafetyLimits`, `DetectorConfig`, `VLMConfig`, `CameraConfig`, `Config` and every default value.
   - `esp32_companion.py`, which defines `ESP32Client`, `LightingController` and `EnvironmentMonitor`.
   - `llm.py`, `mind.py`, `plugins.py`, `slicer.py`, `appservice.py`, `__main__.py` and the finetune module.
   - `scripts/download_obico_model.sh`, `esp32_firmware/` and the ONNX weights (`models/model-weights-5a6b1be1fa.onnx`).

   Consequences:
   - The `SafetyLimits` **class** cannot be quoted. Only its config values (`config.yaml`) and every place that enforces them can be quoted (§4).
   - The ESP32 lighting logic can only be described from its call sites (§3).
   - Defaults for keys missing from `config.yaml` are unknown, for example `agent.post_print_inspection`.
2. **The runtime evidence is thin.**
   - `data/agent.log` covers a single day, 2026-09-05 00:59 → 19:44 (1,330 lines).
   - `data/observer.out` is a strict subset of `agent.log`. All 1,060 lines match `agent.log` exactly, so it adds no information.
   - All three agent starts ran with **`dry_run=True`**, so nothing was ever sent to the printer.
   - The agent **always attached mid-print** ("Reprise du suivi"), so the vision pipeline never saw a print from layer 1.
3. **No real failure was ever observed or recorded.** That is the core finding of §6 and §7.

---

## 1. Architecture as actually implemented

```
FrameGrabber threads (camera.py, 1 fps/cam, 24/7)
   └─> VisionWatcher._loop / _process (vision_watcher.py)        [Level 1 "Vigie"]
         ObicoOnnxDetector.detect -> _frame_score -> EMA -> alert latch
         └─ on_alert = PrinterAgent._alerts.put (queue)
PrinterAgent._loop -> _tick (printer_agent.py, 1 s poll, SINGLE THREAD)
   ├─ _thermal_watchdog (MoonrakerClient.check_thermal_safety)
   ├─ job transitions: _on_print_started / _on_print_finished
   ├─ _maybe_apply_profile / _enforce_temp_delta  (learned filament profile)
   ├─ watcher.enable_detection(detect)
   ├─ _drain_env_alerts (ESP32)
   └─ _handle_alert (synchronous, blocks the tick)              [Level 2]
         triage: VLMAnalyzer.analyze(stage="triage")  (no pause)
           └ VLM down -> RuleBasedDecider.decide (protocols.py)
         _inspect_and_correct: pause -> move head (geometry.compute_inspection_target)
           -> fresh frames under LED -> VLMAnalyzer.analyze(stage="inspection")
           -> ProtocolExecutor.execute (protocols.py)          [Level 3, bounded by SafetyLimits]
           -> _resume_after_inspection -> _enter_cooldown
   _on_print_finished -> KnowledgeBase.update_profile_from_success / _persist_to_klipper
                       -> _post_print_inspection (VLM stage="post_print") -> SlicerTuner
```

### Level by level

| Level | Implementation (file → symbols) | Notes |
|---|---|---|
| L1 "Vigie" | `vision_watcher.py`: `ObicoOnnxDetector`, `UltralyticsDetector`, `build_detector`, `VisionWatcher._loop/_process/_frame_score`, `VisionAlert`, `CameraStats` | One shared detector behind `_infer_lock`. It runs on **both** cameras because both have `detect: true`. |
| Cameras | `camera.py`: `CameraSource.snapshot/read_stream/grab/assess_health`, `FrameGrabber.run`, `build_cameras` | HTTP snapshot from crowsnest/mjpg-streamer through nginx on the Moonraker host. Falls back to an MJPEG stream through OpenCV. |
| L2 VLM triage and inspection | `vlm_analyzer.py`: `VLMAnalyzer.analyze/_chat/_parse/build_prompt`, `SYSTEM_PROMPT`, `DIAGNOSIS_SCHEMA` | Ollama `/api/chat` with a JSON schema in `format`. Fallback chain over 5 models. |
| Orchestrator / state machine | `printer_agent.py`: `AgentState` (STARTING, IDLE, MONITORING, TRIAGE, INSPECTING, CORRECTING, COOLDOWN, WAITING_USER, FINALIZING, ERROR), `PrinterAgent._tick/_handle_alert/_inspect_and_correct/_post_print_inspection` | Everything runs on one thread. VLM calls block the thermal watchdog (§9). |
| L3 corrective protocols | `protocols.py`: `ProtocolExecutor.execute`, `RuleBasedDecider.decide`, `Adjustments`, `ExecutionResult`, `InterventionRecord` | Maps a VLM `action` to G-code helpers with clipping. |
| SafetyLimits | Class in **missing** `config.py`. Values in `config.yaml: safety:`. Enforcement is spread across `moonraker_client.py` (`_check_extruder_target`, `_check_bed_target`, `clamp_xyz`, `move_*`, `set_*_percent`, `adjust_z_offset`, `save_config`, `check_thermal_safety`), `protocols.py` (step clipping, caps) and `printer_agent.py` (`_align_safety_with_klipper`, `_PluginHost.gated_gcode`) | There is **no central G-code allow-list**. See §4. |
| Moonraker client | `moonraker_client.py`: `MoonrakerClient` (HTTP via `requests` + WebSocket `printer.objects.subscribe`), `PrinterState` typed view | Reusable and well written. The `dry_run` flag sits in `run_gcode` and in the job calls. |
| ESP32 | `esp32_companion.py` (**missing**), used through `LightingController.request` (capture hook), `.illuminated()` (context manager), `.hold_on()/.release()`, and `EnvironmentMonitor` (humidity/temperature alerts, telemetry) | `senses.py` (PIR/LDR/MPU6050/BME280 and ADXL345 via Klipper) only reads sensors. |
| Geometry | `geometry.py`: `compute_homography` (cv2 RANSAC or DLT), `CameraGeometry.pixel_to_bed`, `compute_inspection_target` | `homography_pairs: []` in the config, so the target is **always** `current_position`. |
| Knowledge base (RAG) | `knowledge/__init__.py` `seed_knowledge` loads `base_3d_printing.yaml` into the `notes` + FTS5 table of `memory.py` (`Memory`). `PrinterAgent._build_context` calls `memory.briefing(...)` to inject up to 1,400 characters into the VLM context | Idempotent through a `sha:` tag. |
| Learning | `learning.py`: `KnowledgeBase` (SQLite tables `prints`, `interventions`, `filament_profiles`, `env_telemetry`, `events`), `FilamentProfile`, `update_profile_from_success` (weighted blend), `ConfigPersister` (printer.cfg edit with a backup). `printer_agent._persist_to_klipper`: `Z_OFFSET_APPLY_PROBE` + `SAVE_CONFIG` only when idle | The learning is about **tuning parameters**, not failure detection. Nothing learns from the vision data. |
| Extras not relevant to failure prediction | `mind_loop.py` (an LLM "heartbeat" every 25–60 s through the same Ollama), `psyche.py`, `scout.py` (forum reader), `calibration.py`, `budget.py` (spools/grams), `memory.py` (notes/tasks), plugins | `mind_loop` shares the GPU and Ollama with triage. Its policy table is "auto / ask / never" (`pause_print` = ask, `cancel_print` = never). |

### Alert life cycle as coded (`printer_agent.py`)

- **Vigie enable condition**, `_tick` lines 413-416:
  ```python
  detect = (state.is_printing and self.session is not None
            and state.print_duration >= self.cfg.agent.ignore_first_seconds
            and time.time() >= self._cooldown_until
            and self.state in (AgentState.MONITORING, AgentState.COOLDOWN))
  ```
  `ignore_first_seconds: 90` means the Vigie is blind for the first 90 s of `print_duration`.
- **Alert handling** in `_handle_alert` (864):
  1. Disable detection and set the state to TRIAGE.
  2. Run the VLM triage on the alert frame, plus the latest toolhead frame if it is usable.
  3. Not confirmed with `confidence ≥ triage_min_confidence_to_dismiss (0.7)` → cooldown of 60 s.
  4. Rules fallback → pause only, or a 120 s cooldown.
  5. `cancel_print` + critical + confidence ≥ 0.85 → direct cancel.
  6. Otherwise `_inspect_and_correct`: PAUSE, then `SET_IDLE_TIMEOUT 1800`, move to Z+20 above the pause position, LED on, capture both cameras, run the VLM "inspection", run `ProtocolExecutor`, move back, RESUME, then cooldown of 180 s.

---

## 2. Obico ONNX: exactly how it is run

### 2.1 Loading (vision_watcher.py:92-125, verbatim)

```python
        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            # Charge les libs CUDA/cuDNN installées via pip (nvidia-*) avant le provider
            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()
                except Exception as e:  # noqa: BLE001
                    log.warning("preload_dlls CUDA: %s (repli CPU possible)", e)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cuda":
            log.warning("CUDAExecutionProvider indisponible (pip install onnxruntime-gpu) -> CPU")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.in_h = int(shape[2]) if isinstance(shape[2], int) else 416
        self.in_w = int(shape[3]) if isinstance(shape[3], int) else 416
        self.names = names or ["failure"]
```

- Model file: `model-weights-5a6b1be1fa.onnx`, which comes from Obico's download script (not included).
- The log confirms the **input is 416×416**: `Détecteur Obico ONNX chargé (model-weights-5a6b1be1fa.onnx, entrée 416x416, providers=[...])`.
- `names_path: null`, so the only label is `"failure"`, and Obico is a single-class model.
- The first run used `['CPUExecutionProvider']` because onnxruntime-gpu was not installed yet. The four later loads used CUDA.
- Config comment: "cuda (~0,7 Go VRAM, 10 ms/img)" [UNSURE: never measured in the logs].

### 2.2 Preprocessing and output parsing (vision_watcher.py:127-162, verbatim)

```python
    def detect(self, img: np.ndarray) -> List[Detection]:
        h, w = img.shape[:2]
        resized = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.expand_dims(np.transpose(x, (2, 0, 1)), 0)
        try:
            outputs = self.session.run(None, {self.input_name: x})
        except Exception as e:  # noqa: BLE001
            raise DetectorError(f"Inférence ONNX échouée: {e}") from e
        if len(outputs) < 2:
            raise DetectorError("Sorties ONNX inattendues (2 tenseurs attendus)")
        boxes = np.asarray(outputs[0])
        confs = np.asarray(outputs[1])
        if boxes.ndim == 4:
            boxes = boxes[:, :, 0, :]
        boxes = boxes[0]
        confs = confs[0]
        if confs.ndim == 1:
            confs = confs[:, None]
        max_conf = confs.max(axis=1)
        cls_id = confs.argmax(axis=1)
        mask = max_conf > self.conf_thresh
        boxes, max_conf, cls_id = boxes[mask], max_conf[mask], cls_id[mask]
        dets: List[Detection] = []
        for c in np.unique(cls_id):
            sel = cls_id == c
            keep = _nms(boxes[sel], max_conf[sel], self.nms_thresh)
            for k in keep:
                b = boxes[sel][k]
                dets.append(Detection(
                    label=self.names[int(c)] if int(c) < len(self.names) else str(int(c)),
                    confidence=float(max_conf[sel][k]),
                    x1=int(np.clip(b[0] * w, 0, w - 1)), y1=int(np.clip(b[1] * h, 0, h - 1)),
                    x2=int(np.clip(b[2] * w, 0, w - 1)), y2=int(np.clip(b[3] * h, 0, h - 1)),
                ))
        return dets
```

NMS (vision_watcher.py:64-82, verbatim). This is greedy IoU NMS, applied per class:

```python
def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[np.where(iou <= thresh)[0]]
    return np.array(keep, dtype=int)
```

Summary for reimplementation:

- **Input:** the BGR frame at its native resolution (after optional rotate/flip) is resized to 416×416 with `INTER_LINEAR`. No letterbox and no aspect-ratio preservation. It is then converted BGR→RGB, cast to float32, divided by 255, laid out as NCHW `[1,3,416,416]`. There is no mean/std normalization.
- **Output 0 (boxes):** `[1, N, 1, 4]`, holding `x1, y1, x2, y2` normalized to 0..1 relative to the 416 input, which is the same as relative to the original frame because no letterbox is used. Boxes are scaled by the original `w, h` and clipped.
- **Output 1 (confs):** `[1, N, C]` with C = 1. Per-box score = max over classes. A box is kept if the score is **strictly above `conf_threshold` = 0.25**. Then per-class NMS at IoU **0.45**.
- **Comparison with upstream Obico** (fetched 2026-10-07 from `TheSpaghettiDetective/obico-server`, branch `release`):
  - `ml_api/lib/onnx.py` uses the identical preprocessing: `cv2.resize(image,(input_w,input_h),interpolation=cv2.INTER_LINEAR)`, `BGR2RGB`, transpose, float32, `/255`, input size read from the session. It uses the same `max/argmax` + per-class `nms_cpu` post-processing with nms 0.45 (the default in the `detect` wrapper).
  - Upstream returns `(label, conf, (xc, yc, w, h))` in pixels. Upstream has a known height bug, `width*(y2-y1)`, which PrinterAgent does **not** reproduce because it keeps corners.
  - **The important difference:** upstream calls `detect(..., thresh=THRESH)` with `THRESH = 0.08` (`ml_api/server.py`). PrinterAgent uses **0.25**, so it discards the weak detections that Obico's temporal logic is designed to accumulate.

### 2.3 Per-frame score (vision_watcher.py:432-446, verbatim)

```python
    @staticmethod
    def _frame_score(dets: List[Detection], w: int, h: int) -> float:
        """
        Score [0,1] : somme des confiances pondérées par la taille relative
        (une grosse zone de spaghetti pèse plus qu'un petit artefact isolé).
        """
        if not dets:
            return 0.0
        total = 0.0
        for d in dets:
            size_w = 0.6 + 0.4 * min(1.0, d.area_ratio(w, h) * 25.0)
            total += d.confidence * size_w
        if len(dets) > 1:
            total *= 1.0 + 0.1 * min(4, len(dets) - 1)
        return float(min(1.0, total))
```

`area_ratio` is the box area divided by the frame area. The size weight saturates at 4% of the frame.

### 2.4 EMA and alert latch (vision_watcher.py:374-430, verbatim core)

```python
        if not frame.health.usable:
            now = time.time()
            if self.on_camera_issue and now - st.dark_notified > 300:
                st.dark_notified = now
                issue = "image sombre/obstruée" if frame.health.dark else "image figée"
                self.on_camera_issue(cam.name, f"{issue} ({frame.health.describe()})")
            return

        t0 = time.perf_counter()
        with self._infer_lock:
            dets = self.detector.detect(frame.image)
        st.inference_ms = (time.perf_counter() - t0) * 1000.0
        if self.cfg.class_filter:
            dets = [d for d in dets if d.label in self.cfg.class_filter]

        w, h = frame.shape
        score = self._frame_score(dets, w, h)
        with self._lock:
            st.last_score = score
            st.last_detections = len(dets)
            st.ema = self.cfg.ema_alpha * score + (1.0 - self.cfg.ema_alpha) * st.ema
            st.history.append(st.ema)
            if len(st.history) > 300:
                del st.history[:-300]
            ema = st.ema
            if ema >= self.cfg.alert_threshold:
                st.above_count += 1
            else:
                st.above_count = 0
            if not st.armed and ema < self.cfg.clear_threshold:
                st.armed = True
                log.info("Vigie %s : score retombé (%.2f), alerte réarmée", cam.name, ema)
            fire = st.armed and st.above_count >= self.cfg.alert_min_frames
            if fire:
                st.armed = False
                st.above_count = 0
        ...
        if fire:
            alert = VisionAlert(camera=cam.name, role=cam.role, timestamp=frame.timestamp,
                                score=ema, peak_score=max(st.history[-self.cfg.alert_min_frames:] or [ema]),
                                detections=dets, frame=frame)
```

**Parameters** (`config.yaml: detector`):
- `conf_threshold 0.25`, `nms_threshold 0.45`
- `ema_alpha 0.3`, `alert_threshold 0.55`, `alert_min_frames 3` (consecutive EMA frames ≥ threshold), `clear_threshold 0.20` (re-arm)
- `max_fps 2.0`. The effective rate is `min(camera fps 1.0, 2.0)` = **1 fps per camera**.
- `save_frames true` saves an annotated JPEG to `data/captures/<job>/..._alert.jpg`
- `class_filter []`

**Other temporal behaviour:**
- `reset_scores()` zeroes the EMA, `above_count` and history at print start and at every cooldown entry. The cooldown is 60 s after a dismissed triage, 120 s after a rules decision, and 180 s after a correction.
- Each camera has an independent EMA. Either camera can fire.

**Numerical behaviour** (my simulation of the exact code):

| Constant per-frame score | Frames until alert (1 fps ⇒ seconds) |
|---|---|
| 1.00 | 5 |
| 0.80 | 6 |
| 0.70 | 7 |
| 0.60 | 9 |
| 0.56 | 14 |
| ≤ 0.55 | never |

| Single detection | Frame score |
|---|---|
| conf 0.5, area 1% | 0.35 → **never alerts** |
| conf 0.5, area ≥ 4% | 0.50 → **never alerts** |
| conf 0.7, area 1% | 0.49 → never |
| conf 0.9, area 0.1% | 0.549 → never |
| 2 dets, conf 0.4, area 1% | 0.62 → alerts in ~9 s |

⇒ **A single detection with confidence ≤ 0.55 can never trigger an alert, whatever its size.** That is the opposite of what early detection needs, since early spaghetti shows up as weak, small detections.

**Upstream Obico temporal logic, for comparison** (`backend/lib/prediction.py`, structure confirmed by fetch, constants **[UNSURE: from memory, verify in the repo's settings]**):
- `p = sum(conf of detections ≥ 0.08)` (no area weighting).
- `ewm_mean` with `alpha = 2/(EWM_SPAN+1)`, plus a short and a long rolling mean.
- `is_failing`: returns False for the first `INIT_SAFE_FRAME_NUM` frames. Then `adjusted = (ewm_mean - rolling_mean_long) * sensitivity / escalating_factor`. It returns False if `adjusted < THRESHOLD_LOW`, True if `adjusted > THRESHOLD_HIGH`, otherwise True if `adjusted > (rolling_mean_short - rolling_mean_long) * ROLLING_MEAN_SHORT_MULTIPLE`.
- The values I remember are THRESHOLD_LOW 0.38, THRESHOLD_HIGH 0.78, INIT_SAFE_FRAME_NUM 30, ROLLING_WIN_SHORT 310, ROLLING_WIN_LONG 7200, MULTIPLE 3.8, EWM span 12. **Treat these as unverified.**

**The key difference:** Obico subtracts a long-run per-print baseline, so a static false-positive object such as a brim, a cable, a purge line or glare is cancelled out. PrinterAgent has **no baseline subtraction**: a persistent false detection keeps the EMA high, and a real one competes with nothing.

### 2.5 Alert → action mapping

- `VisionAlert.defect_type` maps the Obico label `"failure"` / `"fail"` / `"0"` / `""` to `"spaghetti"` (vision_watcher.py:234-240).
- `peak_score` is the max of the **EMA** history over the last 3 frames, not of the raw scores. The name is misleading, and `RuleBasedDecider` consumes it as if it were a raw peak.

---

## 3. Camera acquisition

**URLs** (config.yaml → `CameraSource._resolve`, camera.py:99-105). A relative URL is resolved to `http://<moonraker.host>/<url>` (port 80, nginx/crowsnest):
- `cam_pc` (overview): `http://192.168.1.4/webcam/?action=snapshot`, stream `/webcam/?action=stream`
- `cam_extrudeur` (toolhead): `http://192.168.1.4/webcam2/?action=snapshot`, stream `/webcam2/?action=stream`
- Both use `mode: snapshot`, `fps: 1.0`, rotation 0, no flip, `dark_threshold 18.0`, `blur_threshold 15.0`, `detect: true`.

Snapshot and fallback (camera.py:123-172, verbatim):

```python
    def snapshot(self, timeout: float = 6.0) -> Frame:
        if not self.snapshot_url:
            raise CameraError(f"{self.name}: pas de snapshot_url")
        self._run_hook()
        try:
            r = self._session.get(self.snapshot_url, timeout=timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise CameraError(f"{self.name}: snapshot échoué ({e})") from e
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise CameraError(f"{self.name}: image indécodable ({len(r.content)} octets)")
        return self._make_frame(img)
    ...
    def read_stream(self) -> Frame:
        self._run_hook()
        with self._cap_lock:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open_stream()
            # Purge du buffer pour obtenir l'image la plus récente
            for _ in range(2):
                self._cap.grab()
            ok, img = self._cap.read()
            if not ok or img is None:
                self._cap.release()
                self._cap = None
                raise CameraError(f"{self.name}: lecture du flux échouée")
        return self._make_frame(img)

    def grab(self) -> Frame:
        """Capture selon le mode configuré, avec repli snapshot <-> stream."""
        primary = self.read_stream if self.cfg.mode == "stream" else self.snapshot
        fallback = self.snapshot if self.cfg.mode == "stream" else self.read_stream
        try:
            return primary()
        except CameraError as e:
            if (self.cfg.mode == "stream" and self.snapshot_url) or (self.cfg.mode == "snapshot" and self.stream_url):
                log.debug("%s: %s -> repli", self.name, e)
                return fallback()
            raise
```

Health check (camera.py:187-205, verbatim):

```python
    def assess_health(self, img: np.ndarray) -> FrameHealth:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        # Laplacien sur une version réduite : rapide et stable
        small = cv2.resize(gray, (320, int(320 * gray.shape[0] / max(1, gray.shape[1]))))
        sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
        digest = _frame_digest(img)
        if digest == self._last_digest:
            self._same_count += 1
        else:
            self._same_count = 0
        self._last_digest = digest
        return FrameHealth(
            brightness=brightness,
            sharpness=sharpness,
            dark=brightness < self.cfg.dark_threshold,
            blurry=sharpness < self.cfg.blur_threshold,
            frozen=self._same_count >= 5,
        )
```

- `usable = not (dark or frozen)`. A **blurry frame is still usable** and goes to the detector.
- `_frame_digest` is the MD5 of a 32×24 INTER_AREA thumbnail. Only exact repeats count as "frozen".

Retry loop (`FrameGrabber.run`, camera.py:230-255, verbatim):

```python
    def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            if not self._enabled.is_set():
                time.sleep(0.2)
                continue
            t0 = time.time()
            try:
                frame = self.source.grab()
                with self._lock:
                    self._latest = frame
                self.frames += 1
                backoff = 1.0
            except CameraError as e:
                self.errors += 1
                if self.errors % 10 == 1:
                    log.warning("%s: %s (erreurs=%d)", self.source.name, e, self.errors)
                time.sleep(min(backoff, 10.0))
                backoff *= 1.7
                continue
            except Exception:  # noqa: BLE001
                log.exception("%s: erreur inattendue de capture", self.source.name)
                time.sleep(2.0)
                continue
            elapsed = time.time() - t0
            self._stop.wait(max(0.0, self.interval - elapsed))
```

Backoff is ×1.7 up to 10 s. `errors` is never reset, so after a recovery, warnings are only logged every 10th error overall.

**Lighting through ESP32** (the implementation is missing; this is call-site behaviour only):
- `PrinterAgent.__init__`: `capture_hook = self.lighting.request if self.lighting else None` is passed to `build_cameras`. Therefore `LightingController.request(camera_name)` runs **before every single grab**, at 1 fps per camera, 24/7.
- `with self._lit():`, which is `self.lighting.illuminated()` or `nullcontext()`, wraps:
  - the inspection captures (printer_agent.py:953-955). This is correct.
  - the triage **VLM call** (887-889) and the inspection **VLM call** (973-975). That is useless, because the images were already captured.
- Config: `light_mode: on_capture`, `light_hold_s 8`, `light_warmup_s 0.6`, `light_auto_off_s 120`, `light_relays [0]`.
- `_on_camera_issue` calls `lighting.request(camera)` when a frame is dark.
- [UNSURE] Whether `request()` blocks for `light_warmup_s` before returning. If it does not, the first frames after a dark period are captured unlit.
- `esp32.enabled: true` with `transport: serial`. The logs show `sens indisponibles : pyserial non installé`, so on 2026-09-05 the ESP32 path was **not working** in the app service.

Grabbers start in `VisionWatcher.start()` and run permanently, idle included. In the logs, the detector stayed loaded and the grabbers stayed alive for 13 h of idle.

---

## 4. SafetyLimits and G-code guardrails

### 4.1 The class

`SafetyLimits` is defined in `printer_agent/config.py`, which is **not in the upload**, so its verbatim definition cannot be provided. Every field it must have is used somewhere. All 24 fields used in code are listed below, with their values from `config.yaml` (verbatim):

```yaml
safety:
  # Températures (restent sous les max_temp Klipper, alignées au démarrage)
  extruder_max_temp: 245.0
  extruder_min_print_temp: 180.0
  bed_max_temp: 110.0
  bed_min_temp: 0.0
  # Dérive thermique tolérée (target vs mesure) avant intervention
  max_temp_deviation: 25.0
  temp_deviation_seconds: 90.0
  # Bornes des facteurs
  flow_min: 85.0
  flow_max: 115.0
  speed_min: 50.0
  speed_max: 120.0
  max_flow_step: 5.0
  max_speed_step: 15.0
  max_temp_step: 10.0
  max_bed_temp_step: 5.0
  max_z_offset_step: 0.05
  max_z_offset_total: 0.30
  pressure_advance_min: 0.0
  pressure_advance_max: 1.0
  max_interventions_per_print: 6
  z_offset_max_layer: 3
  inspection_z_hop: 20.0
  travel_speed_mm_s: 120.0
  z_speed_mm_s: 10.0
  xy_margin: 3.0
```

### 4.2 Enforcement code (verbatim)

**Alignment with Klipper at startup** (printer_agent.py:348-361):

```python
    def _align_safety_with_klipper(self, settings: Dict[str, Any]) -> None:
        """Ne jamais dépasser les max_temp de printer.cfg, quelle que soit la config agent."""
        ext_max = settings.get("extruder", {}).get("max_temp")
        bed_max = settings.get("heater_bed", {}).get("max_temp")
        s = self.cfg.safety
        if ext_max and s.extruder_max_temp > float(ext_max) - 5:
            s.extruder_max_temp = float(ext_max) - 5
            log.info("safety.extruder_max_temp aligné sur Klipper: %.0f°C", s.extruder_max_temp)
        if bed_max and s.bed_max_temp > float(bed_max) - 5:
            s.bed_max_temp = float(bed_max) - 5
            log.info("safety.bed_max_temp aligné sur Klipper: %.0f°C", s.bed_max_temp)
        min_ext = settings.get("extruder", {}).get("min_extrude_temp")
        if min_ext and s.extruder_min_print_temp < float(min_ext) + 5:
            s.extruder_min_print_temp = float(min_ext) + 5
```

**Thermal targets** (moonraker_client.py:585-612):

```python
    def _check_extruder_target(self, target: float) -> None:
        if target < 0 or target > self.safety.extruder_max_temp:
            raise SafetyViolation(f"Cible extrudeur {target}°C hors bornes "
                                  f"[0, {self.safety.extruder_max_temp}]")
        if 0 < target < self.safety.extruder_min_print_temp:
            log.warning("Cible extrudeur %.0f°C < température d'impression mini %.0f°C",
                        target, self.safety.extruder_min_print_temp)

    def _check_bed_target(self, target: float) -> None:
        if target < self.safety.bed_min_temp or target > self.safety.bed_max_temp:
            raise SafetyViolation(f"Cible lit {target}°C hors bornes "
                                  f"[{self.safety.bed_min_temp}, {self.safety.bed_max_temp}]")

    def set_heater_temperature(self, target: float, wait: bool = False, timeout: float = 600.0) -> None:
        """M104 / M109 sur l'extrudeur, avec contrôle des seuils."""
        target = round(float(target), 1)
        self._check_extruder_target(target)
        log.info("Extrudeur -> %.1f°C%s", target, " (attente)" if wait else "")
        self.run_gcode(f"{'M109' if wait else 'M104'} S{target}",
                       timeout=timeout if wait else None)

    def set_bed_temperature(self, target: float, wait: bool = False, timeout: float = 900.0) -> None:
        """M140 / M190 sur le lit, avec contrôle des seuils."""
        target = round(float(target), 1)
        self._check_bed_target(target)
        log.info("Lit -> %.1f°C%s", target, " (attente)" if wait else "")
        self.run_gcode(f"{'M190' if wait else 'M140'} S{target}",
                       timeout=timeout if wait else None)
```

**Thermal watchdog** (moonraker_client.py:629-653):

```python
    def check_thermal_safety(self, state: PrinterState) -> List[str]:
        """
        Retourne une liste d'anomalies thermiques :
        - dépassement des maxima absolus
        - dérive prolongée target/mesure (chauffe qui décroche, thermistance...)
        """
        issues: List[str] = []
        now = time.time()
        if state.extruder_temp > self.safety.extruder_max_temp + 5:
            issues.append(f"CRITIQUE: extrudeur {state.extruder_temp:.0f}°C > max {self.safety.extruder_max_temp:.0f}°C")
        if state.bed_temp > self.safety.bed_max_temp + 5:
            issues.append(f"CRITIQUE: lit {state.bed_temp:.0f}°C > max {self.safety.bed_max_temp:.0f}°C")

        for name, temp, target in (("extruder", state.extruder_temp, state.extruder_target),
                                   ("heater_bed", state.bed_temp, state.bed_target)):
            # On ne surveille la dérive que si la cible est active et déjà atteinte une fois
            # (évite les faux positifs pendant la chauffe initiale)
            deviating = target > 0 and (temp - target) < -self.safety.max_temp_deviation
            elapsed = self._thermal.update(name, deviating, now)
            if deviating and elapsed >= self.safety.temp_deviation_seconds:
                issues.append(f"{name}: {temp:.0f}°C sous la cible {target:.0f}°C depuis {elapsed:.0f}s")
            if target > 0 and (temp - target) > self.safety.max_temp_deviation:
                issues.append(f"{name}: {temp:.0f}°C dépasse la cible {target:.0f}°C de plus de "
                              f"{self.safety.max_temp_deviation:.0f}°C")
        return issues
```

Its consumer, `PrinterAgent._thermal_watchdog` (printer_agent.py:463-491), reacts as follows:
- "CRITIQUE" → `pause_print()` (if printing), then `turn_off_heaters()`. If that raises, `emergency_stop()` (M112). Then WAITING_USER.
- Any other issue → notify at most once every 10 min. If printing and in MONITORING, `pause_print()` and WAITING_USER.

**Motion** (moonraker_client.py:664-750). This covers `clamp_xyz` (X/Y bounded to `[axis_min+xy_margin, axis_max-xy_margin]`, Z to `[axis_min, axis_max-0.5]`, pass-through when limits are unknown), `move_absolute`/`move_relative` (refused unless `"xyz" in homed_axes` or dry-run) and the inspection move:

```python
    def move_to_inspection_position(self, x: float, y: float, z: float) -> None:
        state = self.get_state()
        cx, cy, cz = self.clamp_xyz(x, y, z, state)
        homed = state.homed_axes or ""
        if not self.dry_run and ("x" not in homed or "y" not in homed):
            raise SafetyViolation("Inspection refusée : axes X/Y non référencés")
        cur_z = state.gcode_position[2]
        if not self.dry_run and cz is not None and cz < cur_z and "z" not in homed:
            raise SafetyViolation("Inspection refusée : descente Z sans référentiel Z")
        zf = self.safety.z_speed_mm_s * 60.0
        xyf = self.safety.travel_speed_mm_s * 60.0
        script_lines = ["G90"]
        if cz is not None and cz > cur_z:
            script_lines.append(f"G1 Z{cz:.3f} F{zf:.0f}")
        script_lines.append(f"G1 X{cx:.3f} Y{cy:.3f} F{xyf:.0f}")
        if cz is not None and cz <= cur_z:
            script_lines.append(f"G1 Z{cz:.3f} F{zf:.0f}")
        script_lines.append("M400")
        ...
        self.run_gcode("\n".join(script_lines), timeout=180)
```

**Factors** (moonraker_client.py:758-805):

```python
    def set_flow_percent(self, percent: float) -> float:
        p = min(max(float(percent), self.safety.flow_min), self.safety.flow_max)
        ...
        self.run_gcode(f"M221 S{p:.1f}")
        return p
    def adjust_flow(self, delta_percent: float) -> float:
        d = max(-self.safety.max_flow_step, min(self.safety.max_flow_step, float(delta_percent)))
        current = self.get_state().flow_factor
        return self.set_flow_percent(current + d)
    def set_speed_percent(self, percent: float) -> float:
        p = min(max(float(percent), self.safety.speed_min), self.safety.speed_max)
        ...
        self.run_gcode(f"M220 S{p:.1f}")
        return p
    def set_fan_percent(self, percent: float) -> float:
        p = min(max(float(percent), 0.0), 100.0)
        pwm = int(round(p * 255.0 / 100.0))
        self.run_gcode(f"M106 S{pwm}" if pwm > 0 else "M107")
        return p
    def set_pressure_advance(self, value: float) -> float:
        v = min(max(float(value), self.safety.pressure_advance_min), self.safety.pressure_advance_max)
        self.run_gcode(f"SET_PRESSURE_ADVANCE ADVANCE={v:.4f}")
        return v
    def adjust_z_offset(self, delta_mm: float, move: bool = True) -> float:
        """SET_GCODE_OFFSET Z_ADJUST (babystepping). Le total cumulé est borné par l'appelant."""
        d = max(-self.safety.max_z_offset_step, min(self.safety.max_z_offset_step, float(delta_mm)))
        self.run_gcode(f"SET_GCODE_OFFSET Z_ADJUST={d:.3f} MOVE={1 if move else 0}")
        return d
```

(Log lines are elided with `...`. The logic is verbatim.)

**SAVE_CONFIG** (moonraker_client.py:893-907):

```python
    def save_config(self, wait_timeout: float = 90.0) -> None:
        """SAVE_CONFIG : écrit printer.cfg et redémarre Klipper. Refusé si job actif."""
        state = self.get_state()
        if state.is_active_job:
            raise SafetyViolation("SAVE_CONFIG refusé pendant une impression")
```

**Dry-run gate** (moonraker_client.py:561-576). This is the *only* gate on raw G-code:

```python
    def run_gcode(self, script: str, timeout: Optional[float] = None) -> str:
        script = script.strip()
        if not script:
            return ""
        if self.dry_run:
            log.info("[DRY-RUN] G-code non envoyé: %s", script.replace("\n", " | "))
            return "ok (dry-run)"
        ...
        res = self._post("/printer/gcode/script", json_body={"script": script},
                         timeout=timeout or self.cfg.gcode_timeout)
```

**Plugin G-code gate** (printer_agent.py:131-141):

```python
    def gated_gcode(self, plugin: str, script: str, timeout: float) -> str:
        up = script.strip().upper()
        for forbidden in ("M112", "SAVE_CONFIG", "FIRMWARE_RESTART", "RESTART", "CANCEL_PRINT", "M18", "M84"):
            if up.startswith(forbidden):
                raise PermissionError(f"plugin {plugin} : commande interdite aux plugins : {forbidden}")
        if self._a.state in (AgentState.TRIAGE, AgentState.INSPECTING, AgentState.CORRECTING):
            raise RuntimeError(f"plugin {plugin} : l'agent intervient ({self._a.state.value}), G-code refusé")
        return self._a.client.run_gcode(script, timeout=timeout)
```

**Protocol layer** (protocols.py:71-189, verbatim):

```python
    def execute(self, diag: VLMDiagnosis, state: PrinterState, adj: Adjustments,
                current_layer: Optional[int], interventions_so_far: int) -> ExecutionResult:
        action = diag.action
        p = diag.parameters
        try:
            if action == "continue":
                return ExecutionResult(True, action, message="aucune correction (fausse alerte / incertain)")

            if interventions_so_far >= self.safety.max_interventions_per_print and action not in (
                    "cancel_print", "pause_for_user"):
                log.warning("Nombre max d'interventions atteint (%d) : escalade vers l'utilisateur",
                            self.safety.max_interventions_per_print)
                return self._pause_for_user(diag, "trop d'interventions automatiques")

            if action == "cancel_print":
                if not (diag.is_critical and diag.confidence >= 0.7):
                    log.warning("cancel_print demandé sans critère critique/confiance -> pause utilisateur")
                    return self._pause_for_user(diag, "annulation proposée mais non certaine")
                self.client.cancel_print()
                self.client.display_message(f"AGENT: print annulé ({diag.defect_type})")
                return ExecutionResult(True, action, message=f"annulé: {diag.description}",
                                       resume=False, cancelled=True)

            if action == "pause_for_user":
                return self._pause_for_user(diag, diag.description)

            if action == "adjust_flow":
                d = self._clip(p.get("delta_percent", 0.0), self.safety.max_flow_step)
                d = self._default_if_zero(d, diag.defect_type, {"under_extrusion": 3.0, "over_extrusion": -3.0})
                new_total = adj.flow_delta + d
                target = state.flow_factor + d
                if not (self.safety.flow_min <= target <= self.safety.flow_max):
                    return ExecutionResult(False, action, message=f"flux {target:.0f}% hors bornes")
                applied = self.client.set_flow_percent(target)
                adj.flow_delta = new_total
                return ExecutionResult(True, action, {"flow_percent": applied, "delta": d},
                                       f"flux {d:+.1f}% -> {applied:.0f}%")

            if action == "adjust_temperature":
                d = self._clip(p.get("delta_c", 0.0), self.safety.max_temp_step)
                d = self._default_if_zero(d, diag.defect_type,
                                          {"stringing": -5.0, "under_extrusion": 5.0, "clog": 5.0, "blob": -5.0})
                target = state.extruder_target + d
                target = min(max(target, self.safety.extruder_min_print_temp), self.safety.extruder_max_temp)
                d = target - state.extruder_target
                if abs(d) < 0.5:
                    return ExecutionResult(False, action, message="température déjà à la borne")
                self.client.set_heater_temperature(target, wait=False)
                adj.temp_delta += d
                # On attend que la nouvelle consigne soit approchée avant de reprendre (max 90 s)
                try:
                    self.client.wait_for_temperature("extruder", target, tolerance=3.0, timeout=90)
                except MoonrakerError as e:
                    log.warning("Température non atteinte à temps: %s", e)
                return ExecutionResult(True, action, {"extruder_target": target, "delta": d},
                                       f"buse {d:+.0f}°C -> {target:.0f}°C")

            if action == "adjust_bed_temperature":
                d = self._clip(p.get("delta_c", 0.0), self.safety.max_bed_temp_step)
                d = self._default_if_zero(d, diag.defect_type, {"warping": 5.0, "detachment": 5.0})
                target = min(max(state.bed_target + d, self.safety.bed_min_temp), self.safety.bed_max_temp)
                d = target - state.bed_target
                if abs(d) < 0.5:
                    return ExecutionResult(False, action, message="lit déjà à la borne")
                self.client.set_bed_temperature(target, wait=False)
                adj.bed_delta += d
                return ExecutionResult(True, action, {"bed_target": target, "delta": d},
                                       f"lit {d:+.0f}°C -> {target:.0f}°C")

            if action == "adjust_speed":
                d = self._clip(p.get("delta_percent", 0.0), self.safety.max_speed_step)
                d = self._default_if_zero(d, diag.defect_type, {"default": -10.0})
                target = state.speed_factor + d
                if not (self.safety.speed_min <= target <= self.safety.speed_max):
                    target = min(max(target, self.safety.speed_min), self.safety.speed_max)
                    d = target - state.speed_factor
                applied = self.client.set_speed_percent(target)
                adj.speed_delta += d
                return ExecutionResult(True, action, {"speed_percent": applied, "delta": d},
                                       f"vitesse {d:+.0f}% -> {applied:.0f}%")

            if action == "adjust_fan":
                pct = p.get("percent")
                if pct is None:
                    pct = {"warping": 30.0, "detachment": 20.0, "stringing": 100.0}.get(diag.defect_type, state.fan_percent)
                applied = self.client.set_fan_percent(pct)
                adj.fan_percent = applied
                return ExecutionResult(True, action, {"fan_percent": applied}, f"ventilateur -> {applied:.0f}%")

            if action == "adjust_z_offset":
                if current_layer is not None and current_layer > self.safety.z_offset_max_layer:
                    return ExecutionResult(False, action,
                                           message=f"babystep refusé à la couche {current_layer} (> {self.safety.z_offset_max_layer})")
                d = self._clip(p.get("delta_mm", 0.0), self.safety.max_z_offset_step)
                if abs(d) < 1e-4:
                    d = -0.02 if diag.defect_type == "poor_first_layer" else 0.0
                if abs(adj.z_adjust + d) > self.safety.max_z_offset_total:
                    return ExecutionResult(False, action, message="cumul babystep Z max atteint")
                applied = self.client.adjust_z_offset(d, move=False)
                adj.z_adjust += applied
                return ExecutionResult(True, action, {"z_adjust": applied, "total": adj.z_adjust},
                                       f"Z {applied:+.3f} mm (cumul {adj.z_adjust:+.3f})")

            if action == "adjust_pressure_advance":
                v = p.get("value")
                if v is None:
                    v = state.pressure_advance + (0.02 if diag.defect_type in ("blob", "over_extrusion") else -0.02)
                applied = self.client.set_pressure_advance(v)
                adj.pressure_advance = applied
                return ExecutionResult(True, action, {"pressure_advance": applied}, f"PA -> {applied:.3f}")

            return ExecutionResult(False, action, message=f"action inconnue '{action}'")

        except SafetyViolation as e:
            log.error("Protocole refusé par les garde-fous: %s", e)
            return ExecutionResult(False, action, message=f"garde-fou: {e}")
        except MoonrakerError as e:
            log.error("Protocole '%s' échoué: %s", action, e)
            return ExecutionResult(False, action, message=f"erreur Moonraker: {e}")
```

**VLM-side clamp:** `_parse` in vlm_analyzer.py forces `action = "continue"` when `anomaly_confirmed is False` or `defect == "none"`, and coerces `unclear` to `continue` unless the action is `pause_for_user`.

**Fallback rules** (protocols.py:213-222):

```python
        if peak >= 0.9 and n_detections >= 2 and consecutive_alerts >= 2:
            return VLMDiagnosis(True, "spaghetti", "high", 0.6, "...", "pause_for_user", {}, "règle de repli")
        if score >= 0.75 and consecutive_alerts >= 3:
            return VLMDiagnosis(True, "spaghetti", "medium", 0.5, "...", "pause_for_user", {}, "règle de repli")
        return VLMDiagnosis(False, "unclear", "low", 0.3, "...", "continue", {}, "règle de repli")
```

(The French description strings are elided as `"..."`.)

### 4.3 Every rule, enumerated

1. Extruder target must be in [0, `extruder_max_temp`]. Otherwise `SafetyViolation`. Targets below `extruder_min_print_temp` only log a **warning** from the client. The protocol clamps to [min_print, max].
2. Bed target must be in [`bed_min_temp`, `bed_max_temp`], otherwise `SafetyViolation`.
3. At startup, the extruder and bed maxima are lowered to Klipper `max_temp - 5`, and `extruder_min_print_temp` is raised to Klipper `min_extrude_temp + 5`.
4. Thermal CRITIQUE when the measured temperature is above `max + 5`. Response: pause, `TURN_OFF_HEATERS`, and M112 if that call fails.
5. Thermal drift: temperature below `target - 25` for ≥ 90 s, or temperature above `target + 25`. Response: notify every 10 min at most, and pause if printing and MONITORING.
6. Flow factor in [85, 115] %. Step ±5 per action (protocol clip). Out of bounds → the action fails, it is not clamped.
7. Speed factor in [50, 120] %. Step ±15. Out of bounds → clamped.
8. Hotend change ±10 °C per action (clip). After applying, wait up to 90 s for ±3 °C.
9. Bed change ±5 °C per action.
10. Fan 0..100 % absolute, **with no step limit and no SafetyLimits field**.
11. Pressure advance in [0, 1] absolute, **with no step limit**.
12. Z babystep ±0.05 mm per action, cumulative |Σ| ≤ 0.30 mm, only when layer ≤ 3. `MOVE=0`.
13. At most 6 interventions per print (the counter includes triage records, see §9). Above that, everything except cancel and pause becomes `pause_for_user`.
14. `cancel_print` requires severity = critical and confidence ≥ 0.7 in the executor. The direct-from-triage path additionally requires confidence ≥ 0.85.
15. Moves need X/Y homed (inspection) or XYZ homed (absolute and relative moves). A Z descent without a Z home is refused. Coordinates are clamped with a 3 mm XY margin and Z max − 0.5. Z is raised first, then XY, then any Z descent. Feeds are 120 mm/s XY and 10 mm/s Z.
16. Inspection Z = pause Z + 20 mm (`inspection_z_hop`). It never goes below the pause Z (`compute_inspection_target`). Post-print Z = max(current Z, object height + 20).
17. `SAVE_CONFIG` is refused while a job is active. `Z_OFFSET_APPLY_PROBE` + `SAVE_CONFIG` only happen after `wait_for(not active and klippy ready)`.
18. printer.cfg edits (`ConfigPersister.persist`) upload a timestamped `.bak` first and never touch the SAVE_CONFIG block.
19. Plugins cannot send M112, SAVE_CONFIG, FIRMWARE_RESTART, RESTART, CANCEL_PRINT, M18 or M84 (prefix match), and cannot send anything while the agent is in TRIAGE, INSPECTING or CORRECTING.
20. `dry_run` drops every G-code and the pause, resume, cancel and restart calls (`[DRY-RUN]` log).
21. Vigie timing rules: ignore the first 90 s, then cooldowns of 60, 120 or 180 s.
22. `mind_loop` policy: `pause_print`/`resume_print`/`adjust`/`calibrate`/`relay` = ask; `cancel_print`/`config_write` = never.

**What the README claims but the code does not have:**
- "doublons" (a duplicate-command guard): there is none.
- A central G-code validation layer: `run_gcode` accepts anything.
- "toute anomalie thermique → pause + TURN_OFF_HEATERS": only CRITIQUE turns heaters off. Drift only pauses.

---

## 5. VLM triage: prompt and JSON schema (verbatim)

Model config:
- `qwen3-vl:8b-instruct`, with fallbacks `[qwen3-vl:8b, qwen2.5vl:7b, gemma3:12b, qwen3-vl:4b]`.
- `timeout 180`, `keep_alive 30m`, `temperature 0.1`, `top_p 0.9`, `seed 42`, `num_ctx 8192`, `num_predict 400`, `disable_thinking true`, `max_retries 2`.
- `max_image_side 1024` for the first image, 768 for the others. JPEG quality 88.

**Schema** (vlm_analyzer.py:31-63):

```python
DEFECT_TYPES = ["none", "spaghetti", "detachment", "warping", "stringing", "under_extrusion",
                "over_extrusion", "layer_shift", "blob", "clog", "poor_first_layer", "unclear"]
SEVERITIES = ["none", "low", "medium", "high", "critical"]
ACTIONS = ["continue", "adjust_flow", "adjust_temperature", "adjust_bed_temperature",
           "adjust_speed", "adjust_fan", "adjust_z_offset", "adjust_pressure_advance",
           "pause_for_user", "cancel_print"]

DIAGNOSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "anomaly_confirmed": {"type": "boolean"},
        "defect_type": {"type": "string", "enum": DEFECT_TYPES},
        "severity": {"type": "string", "enum": SEVERITIES},
        "confidence": {"type": "number"},
        "description": {"type": "string"},
        "location_x": {"type": "number"},
        "location_y": {"type": "number"},
        "action": {"type": "string", "enum": ACTIONS},
        "parameters": {
            "type": "object",
            "properties": {
                "delta_percent": {"type": "number"},
                "delta_c": {"type": "number"},
                "percent": {"type": "number"},
                "delta_mm": {"type": "number"},
                "value": {"type": "number"},
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["anomaly_confirmed", "defect_type", "severity", "confidence",
                 "description", "action", "reasoning"],
}
```

The schema is sent as Ollama `format`. If Ollama returns HTTP 400, it falls back to `format: "json"`.

Note: `reasoning` comes **after** the decision fields. With constrained decoding, the model commits to `anomaly_confirmed` and `confidence` before it reasons.

**System prompt** (vlm_analyzer.py:65-95):

```text
You are an expert FDM 3D-printing failure analyst embedded in an autonomous
monitoring agent for a Klipper printer. You receive camera images of a print in progress plus
the live printer context. Your job: decide whether a real defect is present, identify it, and
choose ONE corrective action from the allowed list. Be conservative: false positives cost print
time; missed critical failures cost the whole print.

Defect types: none, spaghetti (extruded plastic in the air / tangled strands), detachment
(part lifted off or knocked over from the bed), warping (corners curling up), stringing
(fine hairs between parts), under_extrusion (gaps, thin lines), over_extrusion (bulging,
rough surface, blobs on corners), layer_shift (layers misaligned), blob (large molten mass on
the nozzle or part), clog (nothing extruding, nozzle scraping), poor_first_layer (uneven,
not squished or too squished), unclear (image too dark/blurry/obstructed to judge).

Allowed actions and parameters:
- continue: no change (also use when unclear or a false alarm)
- adjust_flow {delta_percent}: -5..+5  (under_extrusion -> +, over_extrusion -> -)
- adjust_temperature {delta_c}: -10..+10 on the hotend (stringing -> -, under_extrusion/clog -> +)
- adjust_bed_temperature {delta_c}: -5..+5 (warping / adhesion -> +)
- adjust_speed {delta_percent}: -15..+15 (quality issues -> -)
- adjust_fan {percent}: 0..100 absolute part-cooling (warping/adhesion -> lower, stringing/overhang -> higher)
- adjust_z_offset {delta_mm}: -0.05..+0.05, ONLY during the first 3 layers (poor_first_layer)
- adjust_pressure_advance {value}: 0..1 absolute (blobs at corners / gaps after corners)
- pause_for_user: keep paused, ask a human (blob on nozzle, clog, uncertain but risky)
- cancel_print: unrecoverable (part detached, massive spaghetti, layer_shift)

Rules:
- If the image is too dark or the nozzle area is not visible, answer defect_type=unclear,
  action=continue, low confidence.
- cancel_print only with severity=critical and confidence >= 0.7.
- location_x / location_y are the normalized (0..1) image coordinates of the defect center.
- Respond ONLY with the JSON object matching the requested schema, no prose outside it.
```

**User prompt builder** (vlm_analyzer.py:204-241, verbatim):

```python
    @staticmethod
    def build_prompt(context: Dict[str, Any], stage: str,
                     personality: Optional[Dict[str, str]] = None) -> str:
        lines = [f"Stage: {stage.upper()}"]
        if stage == "triage":
            lines.append("A lightweight detector flagged a possible failure on the wide-angle camera. "
                         "Decide quickly if this is a real failure worth pausing the print for. "
                         "If in doubt but plausible, set anomaly_confirmed=true so a close-up "
                         "inspection is performed.")
        elif stage == "post_print":
            lines.append("The print is FINISHED and the toolhead camera has been positioned above "
                         "the last layer (safety: never below the printed part). Inspect the FINAL "
                         "part quality on the close-up (first image) and overview (second if present): "
                         "stringing/strings, surface finish, blobs, layer adhesion, warping already "
                         "printed, over/under-extrusion. anomaly_confirmed=true only for a REAL "
                         "quality defect worth correcting on the NEXT print; for a clean part set it "
                         "to false and action=continue. When a defect is confirmed, pick the "
                         "corrective action that will tune the slicer profile for the next print "
                         "(e.g. adjust_pressure_advance for corner blobs or gaps after corners, "
                         "adjust_temperature/stringing or adjust_fan for strings, adjust_flow for "
                         "extrusion level). Do NOT suggest pause/cancel/z-offset: the print is over.")
        else:
            lines.append("The print is PAUSED and the toolhead camera has been positioned at the "
                         "suspected defect. Analyse the close-up (first image) and the overview "
                         "(second image if present) and choose the corrective action.")
        if personality:
            title = personality.get("title") or ""
            tone = personality.get("tone") or ""
            pname = personality.get("name") or ""
            flavor = " ".join(x for x in (pname, title, tone) if x)
            lines.append("Style note (do not let this affect the JSON schema): " + flavor + ".")
            lines.append("You may express the description with a touch of workshop humor, "
                         "but keep every JSON field valid and the action fact-based.")
        lines.append("Printer context:")
        for k, v in context.items():
            if v is None or v == "":
                continue
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)
```

**Context keys** (`PrinterAgent._build_context`, printer_agent.py:1092-1144):
- `now`, `print_started`, `file`, `filament`, `layer` ("n / total"), `z_mm`, `progress`, `hotend`, `bed`, `speed_factor`, `flow_factor`, `part_fan`, `pressure_advance`, `z_babystep_total`, `detector_alert`.
- `adjustments_this_print`, `previous_interventions` (last 3, non-triage), `triage_result`, `inspection_position`.
- ESP32 env context plus `filament_box_humidity_trend_1h`.
- `knowledge_base`: the FTS briefing, up to 1,400 characters, queried on the defect type, filament type, "intervention protocole" and the filament key.

The personality block (`name`/`title`/`tone`) is injected in **every** stage, because `agent_personality.enabled: true`.

**Request body** (vlm_analyzer.py:277-298):
- `{"model", "stream": false, "keep_alive": "30m", "options": {temperature, top_p, seed, num_ctx, num_predict}, "messages": [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt,"images":[b64...]}], "format": DIAGNOSIS_SCHEMA}`.
- Plus `"think": false` when the model name contains "qwen".
- If `content` is empty, it salvages JSON from `message.thinking`.

**Parsing** (`_parse`, vlm_analyzer.py:384-446):
- Enums are normalized (lower case, spaces and dashes become `_`). Unknown values become `unclear` / `low` / `continue`.
- Confidence is clamped to [0, 1], with a default of 0.5.
- Location: normalized 0..1. If > 1, it is divided by 100 when ≤ 100, otherwise dropped.
- Consistency coercion: if not confirmed or `none`, the action becomes `continue`.

---

## 6. Evidence for why it worked poorly

### 6.1 What the logs show (agent.log, 1,330 lines; observer.out is a subset)

| Metric | Count |
|---|---|
| Agent starts (`=== Démarrage PrinterAgent`) | 3 (00:59, 01:53, 02:49), **all `dry_run=True`**, 0 with dry_run=False |
| `Configuration chargée` (all CLI/app invocations) | 38 |
| Obico model loads | 5 (1× CPU-only at 00:59, 4× CUDA) |
| Prints monitored | 1 job (`cd_phone_holder_base_9h30m_0.18mm_205C_PLA_CR10S.gcode`, Moonraker job 000038). Attached at 20.6 %, 29.3 % and 44.3 %. Completed 06:09 |
| Vision alerts (`ALERTE VIGIE`) | **0** |
| Triage / inspection VLM calls during printing | **0** |
| ERROR / Exception / Traceback / timeout / "échoué" lines | **0** in both files |
| WARNING lines | 5: `Ollama injoignable` ×1 (00:59), `sens indisponibles : pyserial non installé` ×2, plugin warnings ×2 |
| User pauses during the print | 3 (03:24, 03:32, 03:34). The agent correctly suspended the Vigie and did nothing |
| Heartbeats while printing | 239 |
| `cam_pc` EMA over the print | max **0.092**, mean 0.004, 0 heartbeats above 0.2 |
| `cam_extrudeur` EMA over the print | **0.0 in all 239 heartbeats** |

Representative lines:
```
2026-09-05 00:59:07,835 | WARNING | printer_agent.printer_agent  | Ollama injoignable (http://127.0.0.1:11434) : repli sur règles jusqu'à disponibilité
2026-09-05 02:03 ... [monitoring] printing | ... | vigie cam_pc:ema=0.092 cam_extrudeur:ema=0.0      (peak of the whole print)
2026-09-05 06:09:15,400 | INFO | printer_agent.printer_agent | Fin d'impression (complete) : 0 intervention(s), réglages {...all zero...}
2026-09-05 14:33:23,006 | WARNING | printer_agent.appservice | sens indisponibles : pyserial non installé (pip install pyserial)
```

**Interpretation:**
- On a print that completed successfully, the Vigie produced no false positive. That is genuinely good.
- It is also **zero evidence of sensitivity**. The pipeline was never exercised end to end on a real failure, and never in non-dry-run mode.
- The toolhead camera EMA is exactly 0.0 for the whole print. [INFERRED] The Obico model, trained on wide shots of spaghetti, returns nothing above 0.25 on a nozzle close-up. Running it on that camera is wasted compute and adds no signal.
- **No latency data exists.** `inference_ms` is only in `snapshot_stats()`, which is never logged. `VLMDiagnosis.latency_s` is not persisted, because `InterventionRecord.as_row` omits it.
- The two post-print VLM runs (DB, 22:13 → 22:14:17 and 22:53:39 → 22:53:59) put an **upper bound** of about 20–21 s on the full inspection: moves, settle, captures and VLM. They are not in `agent.log`, because the test script logs to stdout only.

### 6.2 The real failure mode: image quality and VLM reliability (DB conversations and notes)

The `conv_messages` and `notes` tables are the richest evidence.

**Repeated, consistent statements that the camera images are unusable** (from the app and CLI chat; the VLM is describing live camera frames):
- 09-05 01:54 (during the monitored print), CLI: *"Caméra très sombre et floue — impossible d'observer la pièce avec précision."*
- 09-05 15:49, tool `[caméras]`: *"L'image est très sombre et floue… On distingue à peine une buse d'extrudeur et un ventilateur… L'éclairage est insuffisant."*
- 09-05 16:04: the user says they cleaned the nozzle camera. The VLM still answers *"Caméra trop floue et sombre → impossible de détecter un défaut visuel."*
- 09-05 19:30, tool `[caméras]`: *"très floue et sombre, avec une forte surexposition sur la gauche… aucune couche n'est visible."*
- 09-05 19:37: the user says both cameras are now aimed and focused. The VLM still answers *"l'image reste trop sombre et floue."*
- Stored as a fact (`notes`, source=conversation, 16:04): *"Caméra buse nettoyée par l'agent — mais image trop sombre et floue pour détecter des défauts. Éclairage insuffisant."*

[INFERRED] The health gate (`dark_threshold 18`, `blur_threshold 15`) did **not** flag these frames: no "image sombre/obstruée" camera-issue notification appears in `events` or in the log. The Obico detector was therefore fed dark, blurry frames. That explains EMA ≈ 0 better than "no defect present". The thresholds are far too permissive to guarantee input quality.

**The VLM is uncalibrated and sycophantic** (`interventions` table, the only 2 rows):
- Both are `stage=post_print`, `defect_type=none`, `severity=none`, **`confidence=0.98`**, `action=continue`, `model=qwen3-vl:8b-instruct`, `image=NULL`, `print_id=0` (written by the `postprint_inspect_run.py` harness, `kb_id=0`).
- `description`: *"Printos, tu es un génie ! La pièce est parfaite, pas un seul filament dégoulinant… brille comme une œuvre d'art. Tu as bien fait de ne pas t'embêter avec les réglages…"* The second row is nearly identical, including the coined word "squissé".
- So on the **same day** the same model says the cameras are too dark to judge anything, and it also reports **0.98 confidence** that the part is perfect.
- The personality "style note" in `build_prompt` visibly took over the description field: flattery, French, addressing "Printos". [INFERRED] It probably biased the verdict as well.
- **The model's confidence is not usable as a probability.**
- `image=NULL` while the VLM was still called. [INFERRED] No toolhead station frame passed `fr.health.usable`, so `saved` stayed empty and only the **unsaved overview frame** was sent. The prompt tells the model "close-up (first image)", which was false (printer_agent.py:675-693).

**Looping and degenerate LLM output:**
- 63 assistant chat messages but only 43 distinct openings. The same answer was repeated verbatim 10+ times between 16:11 and 17:49, for example *"L'impression avance à 95 % sans anomalie apparente. La caméra est trop floue (net=255/50)…"*.
- That answer also says "95 %", although `agent.log` shows the print **completed at 06:09**. [INFERRED] The app/chat context served stale printer state. `appservice.py` is not provided, so this cannot be confirmed.
- The chat contradicts itself ("je peux allumer les caméras (mais elles sont hors service)" while describing camera images).
- `mind_loop` wrote **162** agent notes (101 insights, 61 thoughts) in about 4 h on 09-05 and 09-09. They are highly repetitive: 92 distinct openings for 101 insights, and 36 for 61 thoughts. They contain unfounded reasoning, for example "Z offset à 0.0 sur plateau à 26 °C → risque de pied d'éléphant" on a print that had finished and cooled.
- It also created **13 open "Expérience" tasks**, 10 distinct, mostly duplicates about measuring the Z offset by hand.
- All of this pollutes the FTS memory that is injected into VLM prompts (`knowledge_base` context key).

**Infrastructure fragility** (from the logs):
- Ollama was down at the first start.
- onnxruntime-gpu was missing at the first start (CPU provider).
- pyserial was missing in the app venv, so the ESP32 senses were unavailable.
- The ESP32 was configured `enabled: true`, but no ESP32 info or telemetry line appears and `env_telemetry` has 0 rows. **The LED lighting path was most likely never effective.** [INFERRED, since `esp32_companion.py` is missing]

### 6.3 Structural reasons it could not have worked well (from the code, detailed in §9)

1. The Obico thresholds and scoring are tuned against early detection (§2.4). There is also no baseline subtraction.
2. The camera health gate is too permissive, so garbage frames reach the detector and the VLM.
3. The VLM confidence drives decisions: dismissal at ≥ 0.7, cancel at ≥ 0.85. It is demonstrably uncalibrated (0.98 on images it elsewhere calls unreadable).
4. VLM calls are synchronous on the main tick. Worst case: 5 models × up to 180 s timeout, with model swaps on a shared 12 GB GPU. During that time the thermal watchdog and the job transitions do not run.
5. Any non-Moonraker exception inside alert handling leaves the agent stuck in TRIAGE or INSPECTING with detection off, possibly with the printer paused (§9 B1/B2).
6. There is no logging of per-frame scores and no outcome labels, so nothing could ever be measured or tuned.

---

## 7. agent.db contents

The file is 462,848 bytes, the journal is in WAL mode, and it was inspected on a copy.

| Table | Rows | Schema (abridged) | Content summary |
|---|---|---|---|
| `prints` | 3 | id, job_id, filename, filament_key, started, ended, status, interventions, adjustments(JSON), learned, notes | **The same Moonraker job 000038, 3 times** (one row per agent restart). Rows 1–2 are stuck at `status='printing'`, `ended=NULL`. Row 3 is `complete`, 0 interventions, all-zero adjustments |
| `interventions` | 2 | id, print_id, ts, camera, stage, defect_type, severity, confidence, action, params, ok, applied, message, model, layer, image, description | 2× `post_print`, both `none`/0.98/continue, `print_id=0`, `image=NULL`, flattering descriptions (§6.2) |
| `filament_profiles` | 0 | key, flow_percent, temp_delta, bed_delta, speed_percent, pressure_advance, fan_percent, samples, success_count, updated | No profile ever learned |
| `env_telemetry` | 0 | ts, source, temperature, humidity, pressure, dew_point | The ESP32 never reported |
| `events` | 14 | ts, level, message | Notifications: 3× "Reprise du suivi", 7× plugin progress, spool debit, memory add, 2× "Inspection finale de la pièce..." (22:13, 22:53) |
| `notes` (+ `notes_fts*`) | 262 | id, ts, kind, topic, text, tags, source, task_id | 69 seed KB notes; 101 agent insights + 61 thoughts (mind_loop); 10 user facts + 3 preferences; 7 conversation facts and preferences; misc. Contains personal profile facts about the user (not reproduced here) |
| `tasks` | 13 | id, created, updated, goal, constraints, amendments, status, state, question | 13 open "Expérience" tasks (Z-offset/temperature measurements), mostly duplicates |
| `decisions` | 0 | ts, task_id, step, tool, args, result, ok, note | — |
| `conv_sessions` | 73 | id, started, title, channel | 70 `app`, 3 `cli` |
| `conv_messages` (+ `conv_fts*`) | 129 | id, session_id, ts, role, content, tokens, compacted | 63 user, 63 assistant, 2 tool (camera descriptions), 1 summary. Range 09-05 → 10-02 |
| `conv_compactions` | 1 | … | One compaction with qwen3-vl:8b-instruct |
| `spools` | 3 | id, name, material, brand, color, net_g, …, remaining_g, filament_key, state, price_eur | 3 DEEPLEE spools (black PLA+ ~152 g left, blue-grey, white) |
| `spool_usage` | 34 | id, spool_id, ts, grams, source, ref | 34 `moonraker_job` debits for job refs 000001…000038 (848 g total). Imported from the Moonraker history. Many jobs are 0.1–0.6 g, which suggests early aborts |
| `budgets`, `budget_usage` | 0 | — | — |

### Is any of this labeled data for evaluating a failure predictor?

**No.** Specifically:
- **0 vision alerts and 0 triage records.** There are 0 stored frames with labels. `interventions.image` is NULL in both rows, and the `data/captures/` directory is not in the upload; it would only contain alert or inspection frames, and no alert ever fired.
- **0 failed prints** recorded by the agent. The only fully tracked print succeeded, so there are no positives at all.
- No per-frame detector scores are persisted anywhere (DB or log), only one EMA value per minute in the heartbeat text.
- The VLM outputs in `interventions` are unlabeled and demonstrably unreliable.

**Weak signal that might be salvageable:**
- `spool_usage` lists Moonraker jobs 000001–000038. Their **status** (complete, cancelled, error) is not stored here, but it can be pulled again from Moonraker `/server/history/list`. That would give per-print outcome labels without images. Many sub-gram jobs suggest cancelled starts.
- Crowsnest/Moonraker timelapses, if any exist, would be the only image source for those jobs. [UNSURE whether any exist]
- The 239 per-minute EMA values for one successful print are a tiny negative-only sample.

---

## 8. Knowledge base YAML (`printer_agent/knowledge/base_3d_printing.yaml`)

**Structure:** `version: 1`, then `notes:`, a list of `{topic, kind (default "fact"), tags[], text}`. `seed_knowledge` loads it into the `notes` table with tags `sha:<12hex>`, `seed` and the file stem. It is idempotent through a SHA-1 of (topic, normalized text). It is retrieved with FTS5 through `Memory.briefing`.

**69 notes. Counts by topic:**
- defauts 18, calibration 12, prusaslicer 11, materiaux 7, klipper 5, securite 4, cr10s 4, intervention 4, design 3, contexte 1.
- Kinds: 68 fact, 1 preference.
- Text length 155–598 characters, median 329.

**Entries by section:**
- **Sécurité (4):** CR-10S PTFE hotend ≤ 240 °C; thermal-failure signs and reaction; SAVE_CONFIG only when idle; ooze during long pauses (drop to 170–180 °C after 10 min).
- **Défauts (18):** spaghetti (cancel immediately), stringing, warping, detachment, under-extrusion, over-extrusion, blobs/zits, layer shift, elephant foot, ghosting/ringing, Z-banding, pillowing, cracks/layer separation, clog/heat creep, overhangs, bridging, visual temperature cues (matte vs glossy), ideal first-layer appearance.
- **Matériaux (7):** PLA, PETG, ABS/ASA, TPU, Nylon, densities, moisture/drying.
- **Calibration (12):** order of operations, rotation_distance, flow (single wall), temperature tower, pressure advance (TUNING_TOWER), retraction, Z-offset/BLTouch, bed mesh, PID, input shaper, max volumetric flow, cost of tests in grams.
- **Klipper (5):** live commands (M221/M220/M106/M104…), PAUSE/RESUME/CANCEL macros, PA and smooth_time, firmware_retraction, useful Moonraker objects.
- **PrusaSlicer (11):** extrusion_multiplier, temperatures, retraction, first layer, speeds, cooling, perimeters/infill, seam, supports, CLI, G-code metadata.
- **CR-10S (4):** kinematics/limits, bed surfaces, BLTouch offsets (x −46.77 / y −3.29), hotend fan and heat creep.
- **Intervention (4):** safe live interventions (one at a time, wait 2–3 layers), when to cancel, the inspection procedure (pause ≤ 60–90 s, frequent false positives: shadows, supports, brim, glass reflections, purge line), end-of-print learning.
- **Design (3):** FDM design rules, small details, CadQuery/OpenSCAD.
- **Contexte (1):** user setup (CR-10S, Moonraker 192.168.1.4, PrusaSlicer 2.9.6 profile "doxei detail", DEEPLEE PLA, two cameras, Qwen3-VL, Obico on GPU).

### Quality assessment

**Good:**
- Concise, quantified, operational and accurate for mainstream FDM practice.
- Well structured as "recognition → causes → ordered corrective actions".
- The "Intervention" notes are a sensible policy: one change at a time, wait 2–3 layers, cancel criteria.
- It is the most reusable artifact of the project for a VLM or LLM explanation layer.

**Weaknesses:**
- It is advice for a human or an LLM. It is **not** a detector and does not encode visual signatures in a way a model can learn from. It cannot help reach a 98 % prediction target.
- **Inconsistent with the code and config:**
  - KB says PTFE ≤ 240 °C, "Plafond agent: 250"; config `extruder_max_temp: 245`.
  - KB flow bounds 90–115 %; config 85–115 %.
  - KB inspection lift +5–10 mm; config 20 mm.
  - KB pause ≤ 60–90 s; but VLM `timeout: 180` and a 90 s temperature wait.
  - KB "attendre 2-3 couches avant de juger"; but the cooldown is time-based (180 s), not layer-based.
  - KB bowden retraction 4–6 mm; the user's profile note says 0.8 mm (perhaps a direct-drive upgrade). [UNSURE which is current]
- No provenance or sources, and no confidence per note.
- FTS retrieval pulls up to 1,400 characters into every VLM call, competing for context with images on an 8B model.
- The agent's own 162 low-quality "insight/thought" notes live in the **same** table and are retrievable by the same briefing. That dilutes the curated KB.

---

## 9. Recommendations, bugs and design flaws

### 9.1 Reuse as-is (or nearly)

- **`MoonrakerClient` + `PrinterState`** (`moonraker_client.py`). It is solid: HTTP with retries, WS subscribe with deep-merge, oneshot token, typed state, `wait_for`, a clean `dry_run`, pause/resume/cancel with state waits, gcode_store capture.
  - Port it to `httpx` (sync or async) for Valdar.
  - Keep `SUBSCRIBE_OBJECTS`. Also subscribe to `print_stats.info` (current_layer) and `motion_report` if useful.
- **The Obico ONNX preprocessing and decoding** (`ObicoOnnxDetector.detect` + `_nms`). It is numerically identical to upstream Obico `ml_api/lib/onnx.py`. Keep it, but change the threshold and the temporal logic (§9.2).
- **`CameraSource` snapshot/stream fallback, `encode_jpeg`, `draw_detections`, `FrameGrabber`.** These are fine building blocks.
- **`geometry.py`** (homography by RANSAC/DLT, `compute_inspection_target` with "never below the pause Z"). The math is fine; it just needs calibration data.
- **`ConfigPersister.set_value`/`persist`.** It is careful: it leaves the SAVE_CONFIG block alone and writes a backup.
- **The KB YAML content** as an explanation/RAG layer, after reconciling the numbers with SafetyLimits.
- **The Ollama call mechanics** in `VLMAnalyzer._chat`: schema `format`, the `think:false` handling, salvaging JSON from `thinking`, the 400 fallback, and the robust `_extract_json`/`_parse`.

### 9.2 Reimplement better

1. **Temporal failure logic.**
   - Replace `conf_threshold 0.25` + area-weighted score + plain EMA with Obico-style logic:
     - per-frame `p = Σ conf` with detections ≥ 0.08;
     - an EWM plus short and long rolling means;
     - **baseline subtraction** per print;
     - a warm-up period;
     - two-level thresholds.
   - Better still, make the score a calibrated probability by fitting a logistic or isotonic model on logged features (Σconf, max conf, n_dets, box-area growth over time, EWM minus long mean, layer, Z, time since start).
   - **Log every frame's raw detections and score** to a time-series table: that is the prerequisite for measuring anything.
2. **Camera quality gate.**
   - `dark_threshold 18` with mean luminance and `blur_threshold 15` with Laplacian variance let through images the VLM itself calls unreadable.
   - Calibrate per camera from reference frames (for example a percentile of luminance, the fraction of saturated pixels, and Laplacian variance on a fixed ROI). Treat "blurry" as **unusable**.
   - Expose a per-print "vision coverage %". **Shadow-mode metrics must exclude or flag uncovered periods**: a predictor cannot be credited for frames it never saw.
3. **Lighting.**
   - Turn the LEDs on *during the print* (`light_mode: during_print`) rather than toggling per capture at 1 fps.
   - Verify the effect on luminance in a closed loop (capture, measure, retry).
   - Never wrap the VLM *inference* in the light context (printer_agent.py:887-889, 973-975). Wrap the *capture* only.
4. **Concurrency.**
   - Move the VLM work off the control loop: async task or worker thread with a hard overall deadline, for example 30 s.
   - Keep the thermal watchdog and the job-state transitions on their own loop that **never blocks**.
   - Drop the 5-model fallback chain on a shared 12 GB GPU. Use one model (gemma4:12b in Valdar) with a bounded queue, and fail closed to "notify only".
5. **VLM as a second opinion, never as a calibrated probability.**
   - Remove the personality from diagnostic prompts.
   - Put `reasoning` (or observations) **before** the verdict in the schema.
   - Add an explicit `image_quality` / `can_judge` field and refuse to act when it is false.
   - Do **not** use `confidence ≥ 0.7` to dismiss detector alerts until you have measured the VLM's precision and recall on your own labeled frames.
   - Persist `latency_s`, the raw response, the model and the image hashes.
6. **Guardrails.**
   - Create a **single G-code gateway** with an allow-list of command templates (M220/M221/M104/M140/M106/SET_PRESSURE_ADVANCE/SET_GCODE_OFFSET/PAUSE/RESUME/CANCEL_PRINT/G1 inspection moves/M117/RESPOND) and parameter bounds.
   - Add per-print **cumulative** caps on temperature, flow, fan and PA deltas, step limits on fan and PA, and rate limiting with duplicate suppression (the README claims this already exists).
   - Add an explicit **shadow mode** that logs the *would-be* action alongside the prediction. That is separate from `dry_run`, which pretends success.
   - In shadow mode, keep the camera loop and the predictor fully live, and only gate the gateway.
7. **Data model.**
   - One `prints` row per Moonraker `job_id`. Upsert on resume; do not insert (printer_agent.py:505).
   - Add tables for `frames` (path/hash, camera, ts, layer, z, health metrics), `detections` and `scores` (per frame), `predictions` (shadow alerts with lead time), and `outcomes`. Outcomes are the user-confirmed label per print, plus failure onset time, failure type and the source of the label (Moonraker status, user, post-hoc review).
   - That gives you what the ≥ 98 % gate needs. Rough requirement: with zero misses, a one-sided 95 % Clopper-Pearson lower bound of ≥ 0.98 on recall needs **n ≥ 149 caught failure events** (0.98ⁿ ≤ 0.05). With a few misses it needs several hundred.
   - Natural failures are rare, so plan for **deliberately induced failures** (no adhesion aid, knocked parts, starved extrusion) and public spaghetti datasets for pre-training. Your own camera data stays the acceptance test.

### 9.3 Drop

- The personality injection into VLM prompts and the "Printos" persona in diagnostic records (vlm_analyzer.py:228-235). It contaminated the stored diagnoses.
- Running Obico on the toolhead close-up camera (`cam_extrudeur detect: true`). It returned 0.0 for the whole print, and the model is trained on wide shots.
- `mind_loop` / `psyche` / `scout` / nightly LoRA fine-tune from this module's scope. They generate low-value, repetitive notes into the same retrieval store and compete for the GPU. If they are kept anywhere in Valdar, isolate their memory from the diagnostic RAG.
- The autonomous slicer and filament-profile learning (`_maybe_apply_profile`, `_enforce_temp_delta`, `SlicerTuner`) until the predictor is validated. They change printer behaviour during prints without any vision evidence, which confounds the shadow-mode evaluation.
- `RuleBasedDecider` as written. Its "consecutive_alerts" counter is cumulative, and it acts on EMA-as-peak.

### 9.4 Specific bugs and design flaws (file:line)

**Blocking or safety-relevant**

- **B1. printer_agent.py:869-871. The agent gets stuck in TRIAGE after a Moonraker hiccup.** `_handle_alert` disables detection and sets TRIAGE, then calls `self.client.refresh_state()` outside any try. A `MoonrakerError` propagates to `_loop`, which just logs "Tick: Moonraker indisponible". The detect condition at 413-416 requires state ∈ {MONITORING, COOLDOWN}, so **the Vigie stays off for the rest of the print**. The same applies to any exception from `_build_context`, `_record`, etc.
- **B2. printer_agent.py:1007-1022. Paused printer left dangling on unexpected exceptions.** `_inspect_and_correct` catches only `SafetyViolation` and `MoonrakerError`. Any other exception (cv2, geometry, KeyError…) leaves the printer **PAUSED** with `SET_IDLE_TIMEOUT 1800` and the agent in INSPECTING/CORRECTING. The `finally` block restores the idle timeout, so Klipper's idle timeout eventually disables the heaters and ruins the print.
- **B3. printer_agent.py:934 + 1004 + 1011-1018. The agent can resume a print the user paused.**
  - If the *user* pauses during the (blocking) triage, `pause_print()` returns silently (moonraker_client.py:816-818, "Impression déjà en pause"). The agent then moves the head, analyses, and calls `_resume_after_inspection` → `resume_print()`, overriding the user's pause.
  - Separately, the `MoonrakerError` handler calls `resume_print()` whenever the printer is paused, whoever paused it.
  - The agent never records who initiated a pause.
- **B4. Blocking VLM on the control thread** (printer_agent.py:887-889, 973-975; vlm_analyzer.py:257-273). The thermal watchdog (`_thermal_watchdog`) and job-transition handling do not run while the VLM is called. The worst case is the primary model plus 4 fallbacks, each up to `timeout: 180` s (timeouts break per model, other errors retry 3×), plus VRAM model swaps. The KB itself says to pause for ≤ 60–90 s.
- **B5. moonraker_client.py:642-649. The comment says drift is only checked "si la cible est … déjà atteinte une fois", but the code does not implement that.** During initial heat-up (print_state is already `printing` while start G-code runs M190/M109), bed or hotend temperature below `target - 25` for ≥ 90 s raises a drift issue. `_thermal_watchdog` (printer_agent.py:480-491) then **pauses the print** if the state is MONITORING. [INFERRED, not observed: every run was dry-run and attached mid-print]
- **B6. No central G-code filter** (moonraker_client.py:561-576). `run_gcode` sends anything. Plugin filtering (printer_agent.py:134-137) uses `startswith` on the whole script, so `"G4 P1\nSAVE_CONFIG"` bypasses it.
- **B7. Fan and PA are unbounded per step** (protocols.py:152-158, 174-180; moonraker_client.py:786-798). The VLM can set the fan 0 → 100 % or PA 0 → 1.0 in one action. There is no SafetyLimits field for the fan.
- **B8. No cumulative caps on temperature, flow and speed per print** (protocols.py:97-150). Six +10 °C actions are bounded only by the absolute maximum of 245 °C, which is **above** the KB's own 240 °C PTFE limit.

**Detection-quality bugs**

- **B9. vision_watcher.py:148 + config `conf_threshold 0.25`.** Detections Obico would count (≥ 0.08) are discarded.
- **B10. vision_watcher.py:432-446.** A single detection with conf ≤ 0.55 can never fire, whatever its size (§2.4).
- **B11. vision_watcher.py:380-386.** Unusable frames return before the EMA update, so the EMA **freezes** instead of decaying or flagging lost coverage. A camera that goes dark during a failure means a silent miss, with only one notification every 300 s.
- **B12. vision_watcher.py:422.** `peak_score` is the max of the *EMA* history, not of raw scores. `RuleBasedDecider` (protocols.py:214) treats it as a raw peak (`peak >= 0.9`), which is almost unreachable.
- **B13. config `dark_threshold: 18`, `blur_threshold: 15`, camera.py:40 (`usable` ignores `blurry`).** This gate let through frames that the VLM consistently described as too dark and blurry (§6.2).
- **B14. printer_agent.py:900-903. Non-confirmed triage with confidence < 0.7 falls through to PAUSE + inspection.** An uncertain "no anomaly" costs a pause, so false-positive pauses are driven by VLM uncertainty.
- **B15. printer_agent.py:868. `consecutive_alerts` is never reset.** It is cumulative per print, which defeats the "consecutive" semantics in `RuleBasedDecider`.
- **B16. printer_agent.py:918, 986.** `len(sess.interventions)` includes `triage` records, so `max_interventions_per_print = 6` is reached after only 3 alert cycles.
- **B17. printer_agent.py:675-693.** In post-print inspection, the overview frame is appended to `images` without being saved. When all toolhead stations are unusable, the VLM gets only the overview image while the prompt claims "close-up (first image)". The DB rows with `image=NULL` match this.
- **B18. vlm_analyzer.py:404-406. Dead code.** `confidence` is clamped to [0, 1] *before* the `if confidence > 1.0: /100` branch, so a model answering "85" becomes 1.0, not 0.85.
- **B19. vlm_analyzer.py:228-235.** The personality "style note" pollutes the `description` and very likely biases verdicts (stored 0.98 "tu es un génie" diagnoses).
- **B20. protocols.py:235-243.** `InterventionRecord.as_row` does not persist `latency_s`, the raw response or the reasoning in a structured way, and `reasoning` is not stored. There is no data for latency analysis.
- **B21. printer_agent.py:505.** Every agent restart inserts a new `prints` row for the same `job_id`, and earlier rows are never closed (DB rows 1–2 are stuck in `printing`).
- **B22. camera.py:126/148 + printer_agent.py:205.** The LED capture hook fires on every grab at 1 fps, 24/7, because the grabbers run even when idle.
- **B23. camera.py:243-246.** `FrameGrabber.errors` is never reset, so warning throttling breaks after the first outage.
- **B24. printer_agent.py:887-889, 973-975.** `_lit()` wraps the VLM inference instead of the capture of the images being analyzed. The triage image is the unlit `alert.frame`.
- **B25. printer_agent.py:630.** `cfg.agent.post_print_inspection` is not in `config.yaml`. Its default lives in the missing `config.py`. [UNSURE whether it is on]
- **B26. KB/config inconsistencies:** 240 vs 245 °C, 90 vs 85 % flow floor, 5–10 mm vs 20 mm lift, ≤ 60–90 s pause vs 180 s VLM timeout (§8).
- **B27. Shared-GPU contention.** `mind_loop` (every 25–60 s while printing), chat and triage all use the same Ollama model; nothing prioritizes triage. In Valdar (gemma4:12b + XTTS on 12 GB) this gets worse. Consider Obico on CPU (the i5-11400F should handle 416² at about 1 fps; [UNSURE of latency, measure it]) and keep VRAM for the VLM.

### 9.5 Suggested shape for the Valdar successor (brief)

- `capture` (httpx snapshot, per-camera quality gate, coverage metric)
- `detector` (Obico ONNX, faithful preprocessing, thresh 0.08, per-frame logging)
- `predictor` (Obico-style EWM with a baseline, then a learned calibrator; outputs P(failure within T) and lead time)
- `vlm_review` (gemma4:12b, async, no persona, explicit `can_judge`)
- `policy` (shadow until the gate is met; then a limited action set: notify → pause; corrective tuning stays out of scope until separately validated)
- `gateway` (single allow-listed G-code path, cumulative caps, audit log)
- `eval` (per-print outcomes, event-level recall with lead time, false alarms per print-hour, coverage-adjusted)

Promotion criterion: recall ≥ 98 % *with a confidence bound*, measured on events with vision coverage, plus a false-alarm budget. Report both figures, not just recall.
