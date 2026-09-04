# Spec — Cross-Sensor Temporal Super Resolution for Geostationary Thermal Imagery

**Group:** SY E 14 (VIT Computer Engineering, AY 2026-27, Sem I)
**Members:** Jaimin Chudasama, Arnav Jakate, Jayesh Vishwakarma, Jiya Bhandari
**Guide:** Prof. Yashwant Dongre
**Source documents:** `SY-E-14_SYNOPSIS.pdf`, `SY-E-14_SYNOPSIS_02.pdf`, plus the originating
problem statement *"Fill in the Frames Seamlessly — Enhancing Temporal Resolution of Satellite
Imagery using the techniques of AI/ML based on Optical Flow"*.

---

## 1. Problem

Geostationary satellites image on a fixed cadence — INSAT-3DR/3DS at ~30 min, GOES/Himawari at
~10 min. That cadence limits near-real-time monitoring of fast phenomena: fire, cyclones,
thunderstorms, floods, rapid land change.

Classical optical-flow interpolation assumes **smooth motion** and **constant brightness**
between frames. Neither assumption holds for convective cloud fields, which grow, decay, and
change brightness temperature between observations. The result is blurred output and motion
artefacts.

A second problem is validation: INSAT has no routine high-cadence ground truth, so synthesized
intermediate frames are hard to score against reality.

## 2. Goal

Build a working prototype that takes two consecutive thermal-infrared geostationary frames and
synthesizes physically plausible intermediate frames, raising effective cadence from 30 min → 15
min → 7.5 min, and demonstrate it on INSAT-3DS/3DR.

## 3. Functional requirements

### FR-1 Frame interpolation
- Estimate motion between consecutive frames with a learned (AI/ML) optical-flow model, trained
  or fine-tuned on satellite data — not a hand-tuned classical estimator.
- Synthesize intermediate frames with a deep frame-interpolation network.
- **Model input and output files must be `.nc`.** (`.h5` is accepted as an *ingest* format for
  INSAT L1B, which is distributed as HDF5; all model-facing products are NetCDF.)
- Given frames at 00:00 and 00:20, the model must output the frame at 00:10 — i.e. arbitrary and
  midpoint temporal positions, applied recursively for 30 → 15 → 7.5 min.

### FR-2 Validation
- Compare generated frames against real higher-cadence observations.
- Metrics: **MSE, PSNR, SSIM, FSIM** at minimum, plus additional metrics chosen to capture
  **cloud movement** specifically (not just per-pixel similarity).
- Report results with plots.

### FR-3 Visualisation dashboard (web)
- Time-lapse animation of the **original** frames.
- Time-lapse animation of the **interpolated** frames.
- Displayed side by side for comparison.
- Metric charts comparing generated frames to ground truth.
- A generated comparison report.
- Web GUI design is an explicit evaluation criterion.

### FR-4 INSAT application
- Take the best model from FR-1/FR-2 and apply it to INSAT-3DR/3DS.
- Produce INSAT animations at 15-minute temporal resolution.
- Visual quality of the INSAT interpolation is an explicit evaluation criterion.

## 4. Data

| Source | Band | Format | Role |
| --- | --- | --- | --- |
| GOES-19 ABI Channel 13 (~10.3 µm), NOAA AWS bucket | TIR | `.nc` | Primary pretraining + validation (10-min cadence) |
| Himawari-8/9 AHI Band 13 (~10.4 µm), JAXA/NOAA gridded | TIR | `.nc` | Cross-sensor benchmark |
| INSAT-3DS/3DR TIR1 (~10.8 µm), MOSDAC | TIR | `.h5` | Deployment target |

INSAT-native validation sources named in the synopsis:
- **Rapid-scan mode** (~4 min) that INSAT-3DR activates during documented severe weather.
- **Staggered mode** (~15 min) under routine conditions.

These let INSAT accuracy be measured against INSAT's own sensor characteristics rather than a
borrowed proxy. Both are *opportunistic* — availability is not guaranteed, so the plan must
degrade gracefully to GOES/Himawari-based validation plus qualitative INSAT review.

## 5. Approach (from synopsis)

Two stages:

1. **Pretrain** a learned optical-flow interpolation model — RAFT-style flow estimator +
   RIFE-style synthesis network — on GOES-19 and Himawari, where high-cadence data is abundant.
2. **Adapt** to INSAT-3DS/3DR via radiometric renormalization and partial-freeze fine-tuning,
   then validate on INSAT-native high-cadence data.

Training losses: reconstruction (L1/Charbonnier), perceptual/structural (SSIM), temporal
consistency, flow smoothness, radiometric consistency.

Results must be **broken down by meteorological event category** — clear-sky, stratiform,
convective, cyclonic — so performance on operationally important events is reported explicitly
rather than averaged away.

## 6. Non-functional requirements

- **Environmental / cost:** no new hardware; fine-tune a pretrained model rather than train from
  scratch; FP16/AMP training; open datasets (NOAA, JMA); temporal subsampling and deletion of
  verified raw files to bound storage; open-source tooling only.
- **Sustainability:** one model handles seasonal variation via conditioning; partial-freeze
  fine-tuning keeps domain adaptation cheap.
- **Safety:** evaluate separately across clear/stratiform/convective/cyclonic to surface
  severe-weather weakness; temporal-consistency and flow-smoothness losses discourage physically
  implausible output; generated frames are decision-support augmentation, **never** a replacement
  for real observations.
- **Ethics:** every synthesized frame must be machine-identifiable as synthetic via metadata.
  Limitations and failure cases reported transparently. Note INSAT access restrictions.

## 7. Out of scope

- Precipitation nowcasting / forecasting beyond the observation window (the DGMR / NowcastNet /
  PreDiff / MetNet references are context, not deliverables).
- Spatial super-resolution.
- Operational deployment, real-time ingest, or safety-critical autonomous use.
- Non-thermal bands.

## 8. Acceptance criteria

1. `.nc` in → `.nc` out, with synthetic frames flagged in metadata.
2. Learned model beats both baselines (linear blend, classical Farneback warp) on MSE, PSNR,
   SSIM and FSIM against held-out real frames.
3. Cadence demonstrably raised 30 → 15 → 7.5 min on GOES-19 and 30 → 15 min on INSAT.
4. Metrics reported per event category, not only in aggregate.
5. Web dashboard plays original and interpolated animations side by side, renders metric charts,
   and serves the comparison report.
6. INSAT-3DS/3DR animations produced and visually reviewed.
