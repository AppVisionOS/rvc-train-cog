# CLAUDE.md - RVC Training Cog

## Project Overview

RVC (Retrieval-based Voice Conversion) training framework based on VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech). This codebase supports training and inference for voice conversion models, allowing users to clone voices from audio samples.

**Deployment Options:**
- **Cog/Replicate** - Primary deployment method for cloud inference
- **Docker** - Containerized local deployment
- **Local** - Direct Python execution with Gradio web UI

## Directory Structure

```
├── predict.py              # Main Cog prediction interface
├── infer-web.py            # Gradio web UI entry point
├── infer/
│   ├── modules/            # Training and inference modules
│   │   ├── train/          # Training pipeline components
│   │   └── vc/             # Voice conversion modules
│   └── lib/                # Core ML libraries
│       ├── audio.py        # Audio processing utilities
│       ├── train/          # Training utilities
│       └── rmvpe.py        # Pitch extraction
├── configs/                # Model configurations
│   ├── v1/                 # Version 1 configs (32k, 40k, 48k)
│   └── v2/                 # Version 2 configs (32k, 40k, 48k)
├── assets/                 # Pretrained models directory
│   ├── hubert/             # HuBERT feature extractor
│   ├── rmvpe/              # Pitch extraction model
│   └── pretrained_v2/      # Pretrained RVC weights
└── tools/                  # Utility scripts
```

## Key Classes

### `Predictor` (predict.py)
Main Cog interface orchestrating the complete training pipeline. Handles input validation, dataset preparation, and model training coordination.

### `FileManager` (predict.py)
Manages workspace directories and output archive creation. Handles file organization for training artifacts.

### `WavDownloader` (predict.py)
Concurrent audio file downloader using ThreadPoolExecutor. Downloads and converts audio files to WAV format for training.

### `WeightDownloader` (predict.py)
Downloads pretrained model weights from cloud storage. Manages HuBERT, RMVPE, and pretrained RVC model downloads.

### `TrainModules` (predict.py)
Executes the training pipeline stages:
- Preprocessing audio files
- Extracting features (HuBERT embeddings)
- Extracting pitch (F0) using RMVPE
- Preparing training filelists
- Running the main training loop
- Creating FAISS index for retrieval

### `AudioEnhancer` (predict.py)
Audio preprocessing using ClearerVoice-Studio's MossFormer2 model. Features:
- Noise reduction (removes background noise, hiss, hum)
- Silence trimming (removes leading/trailing silence using librosa)
Singleton pattern for efficient model reuse.

### `VC` (infer/modules/vc/)
Voice conversion module for inference. Performs the actual voice transformation using trained models.

## Tech Stack

- **PyTorch 2.0+** with CUDA 11.8
- **Gradio** - Web UI framework
- **FAISS** - Vector similarity search for retrieval indexing
- **HuBERT** - Self-supervised speech feature extraction
- **RMVPE** - Robust pitch (F0) extraction
- **ClearerVoice-Studio** - Audio denoising/enhancement (MossFormer2)
- **FFmpeg** - Audio format conversion
- **librosa/scipy** - Audio processing

## Build Commands

### Cog (Replicate)
```bash
# Run prediction
cog predict -i dataset=@dataset.zip -i experiment_name="my_voice"

# Build container
cog build
```

### Local Development
```bash
# Start Gradio web UI
python infer-web.py
```

### Docker
```bash
docker-compose up
```

## Training Pipeline

1. **Audio Enhancement** (optional) - Denoise input audio using MossFormer2
2. **Silence Trimming** (optional) - Remove leading/trailing silence
3. **Preprocessing** - Normalize and segment audio files
4. **Feature Extraction** - Extract HuBERT embeddings from audio
5. **Pitch Extraction** - Extract F0 contours using RMVPE
6. **Filelist Preparation** - Generate training manifests
7. **Training** - Train the RVC model with extracted features
8. **Index Creation** - Build FAISS index for voice retrieval

## Configuration

Sample rates supported: 32000, 40000, 48000 Hz
Model versions: v1, v2 (v2 recommended)

Config files located in `configs/v1/` and `configs/v2/` directories.

## Cog Predict Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wav_urls` | list[str] | required | List of WAV file URLs for training dataset |
| `sample_rate` | str | "48k" | Sample rate ("40k" or "48k") |
| `version` | str | "v2" | Model version ("v1" or "v2") |
| `f0method` | str | "rmvpe_gpu" | Pitch extraction method |
| `epoch` | int | 10 | Number of training epochs |
| `batch_size` | str | "7" | Training batch size |
| `enhance_audio` | bool | True | Apply MossFormer2 noise reduction to input audio |
| `trim_silence` | bool | True | Trim leading/trailing silence (preserves internal pauses) |

## Testing

- No formal test suite exists
- CI runs preprocessing and feature extraction validation
- Manual validation through pipeline execution
- Test by running short training jobs with sample audio

## Common Issues

- **CUDA out of memory**: Reduce batch size in training config
- **Audio format errors**: Ensure FFmpeg is installed for format conversion
- **Missing pretrained models**: WeightDownloader handles automatic downloads
