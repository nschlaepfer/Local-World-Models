# WM-mac: YUME for Apple Silicon 🍎

**Native macOS compatibility for YUME I2V video generation on Apple Silicon Macs**

This is a community fork of [YUME](https://github.com/stdstu12/YUME) with complete **Apple Silicon (M1/M2/M3/M4) macOS support**.

## ✅ What's Fixed for macOS

### 🔧 Core Compatibility
- **MPS Support**: Full Metal Performance Shaders integration for Apple Silicon GPU acceleration
- **Device Handling**: Automatic CUDA→MPS device translation across all modules
- **Memory Management**: Optimized for Apple's unified memory architecture
- **Distributed Training**: FSDP bypass for single-device MPS inference

### 📦 Library Compatibility
- **Video Processing**: OpenCV fallback for `decord` (not available on macOS)
- **Flash Attention**: PyTorch native fallback when flash-attn unavailable
- **Quantization**: Skip `bitsandbytes` (CUDA-only) with graceful degradation
- **Optimizations**: Fallback for `liger_kernel` CUDA operations

### 🚀 Performance Optimizations
- **T5 CPU Offloading**: Keeps text encoder on CPU for memory efficiency
- **Mixed Precision**: bf16 support for faster inference
- **Gradient Checkpointing**: Handle large models efficiently
- **Unified Memory**: Leverages Apple's high-bandwidth memory architecture

## 📋 Requirements

### Hardware
- **Apple Silicon Mac**: M1, M2, M3, or M4 series
- **Memory**: 64GB+ recommended (128GB for full 79GB model)
- **Storage**: ~100GB free space

### Software
- **macOS**: 12.0+ (Monterey or newer)
- **Python**: 3.8+ (tested on 3.13)
- **PyTorch**: 2.0+ with MPS support

## 🛠 Installation

### Quick Setup
```bash
# Clone WM-mac
git clone https://github.com/nschlaepfer/WM-mac.git
cd WM-mac

# Create virtual environment
python3 -m venv wm_env
source wm_env/bin/activate

# Install dependencies (macOS-compatible)
pip install -r requirements_macos.txt

# Install WM-mac
pip install -e .
```

### Download Models
```bash
# Download YUME I2V model (79GB) 
huggingface-cli download stdstu12/Yume-I2V-540P --local-dir ./Yume-I2V-540P
```

## 🎬 Usage

### Image-to-Video Generation
```bash
# Activate environment
source wm_env/bin/activate

# Run inference (macOS optimized)
bash scripts/inference/sample_image_macos.sh
```

### Custom Configuration
```bash
# For different memory configurations
python fastvideo/sample/sample.py \
    --mixed_precision="bf16" \
    --gradient_checkpointing \
    --t5_cpu \
    --num_euler_timesteps 25  # Faster inference
```

## 📊 Performance

### Tested Configuration
- **Hardware**: MacBook Pro M3 Max, 128GB RAM
- **Model**: Full 79GB YUME I2V-540P
- **Performance**: 
  - Model loading: ~2 minutes
  - Video generation: ~3-5 minutes per video
  - Memory usage: ~60-70GB during inference

### Memory Recommendations
- **64GB**: Basic inference with smaller models
- **128GB**: Full 79GB model with comfortable headroom  
- **192GB**: Future-proof for larger models

## 🔄 What's Different from Original YUME

| Feature | Original YUME | WM-mac |
|---------|---------------|---------|
| Platform | CUDA/Linux only | **macOS Apple Silicon** |
| GPU | NVIDIA only | **Apple Silicon (MPS)** |
| Memory | VRAM limited | **Unified memory** |
| Dependencies | CUDA libraries | **macOS-compatible** |
| Setup | Complex CUDA setup | **Simple pip install** |

## 🚧 Limitations

- **Single GPU only**: Multi-GPU not supported on macOS
- **Slower than CUDA**: MPS performance < high-end NVIDIA GPUs
- **Memory intensive**: Large models require substantial RAM

## 🤝 Contributing

We welcome contributions to improve macOS compatibility:

1. **Performance optimizations** for Apple Silicon
2. **Memory efficiency** improvements
3. **Model quantization** for smaller Macs
4. **Bug fixes** and compatibility issues

## 📄 License

Same as original YUME project.

## 🙏 Acknowledgments

- Original [YUME](https://github.com/stdstu12/YUME) team
- Apple for Metal Performance Shaders
- PyTorch team for MPS backend
- Community contributors

---

**🎯 Made with ❤️ for the macOS AI community** 