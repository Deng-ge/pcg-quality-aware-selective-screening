# Project Summary

## Overview

This repository contains a compact implementation of a quality-aware selective
prediction workflow for heart sound screening. The workflow combines PCG
preprocessing, Log-Mel feature extraction, ResNet-attention classification,
confidence-based rejection, and signal-quality scoring.

The code package is intended for research-code review and lightweight
reproducibility checks. It excludes the original medical recordings and trained
checkpoints.

## Technical Route

1. Read a wav-format PCG recording.
2. Resample to 2 kHz.
3. Apply 20-800 Hz band-pass filtering and peak normalization.
4. Crop or pad to an 8 second segment.
5. Convert the waveform to a 64-bin Log-Mel spectrogram.
6. Estimate normal and abnormal probabilities.
7. Estimate signal quality from amplitude, clipping, flatness, and frequency
   band energy.
8. Combine model confidence and quality indicators to support rejection or
   review routing.

## Experimental Snapshot

| Method | MAcc | AUC | Reject rate |
| --- | ---: | ---: | ---: |
| Baseline ResNet | 0.9034 | 0.9663 | 0.0000 |
| ResNet + confidence rejection | 0.9145 | 0.9757 | 0.0833 |

At 90% selective coverage, the local quality-aware rejection experiment reached
MAcc 0.9323 and AUC 0.9784.

## Repository Status

- `demo/demo_predict.py` runs on a single wav file.
- `examples/synthetic_pcg.wav` is synthetic and included only for demo testing.
- `models/resnet_attention.py` defines the model architecture.
- `models/predict.py` can run with a local PyTorch checkpoint when provided.
- The default no-weight backend is deterministic and is intended only for
  reproducibility checks of the software pipeline.
