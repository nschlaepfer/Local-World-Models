# YUME macOS Setup Guide

## Installation Summary

Your YUME installation has been successfully set up on macOS (M3 Max with 128GB RAM).

### What was installed:
- Python virtual environment: `yume_env`
- PyTorch 2.7.1 with MPS (Metal Performance Shaders) support
- All YUME dependencies (with macOS-compatible alternatives)

### Key modifications for macOS:
1. **Removed packages** (not available for macOS):
   - `decord` - Video decoding library
   - `sentencepiece` - Text tokenization (build issues)
   - `bitsandbytes` - CUDA-only quantization
   - `liger_kernel` - CUDA-specific optimizations

2. **Using MPS instead of CUDA** for GPU acceleration on Apple Silicon

## Activation

To activate the environment:
```bash
source yume_env/bin/activate
```

## Running YUME

### For inference:
```bash
# Make sure you're in the YUME directory
cd /Users/nico/Documents/GitHub/YUME
source yume_env/bin/activate

# Download model weights first, then run inference
python fastvideo/sample/sample.py --your-args-here
```

### Important Notes for macOS:

1. **GPU Usage**: Your M3 Max GPU will be automatically used via MPS. In PyTorch code, use:
   ```python
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   ```

2. **Memory Management**: With 128GB RAM, you can run large models, but MPS has different memory patterns than CUDA.

3. **Performance**: While MPS is fast, some operations may be slower than CUDA. The M3 Max is very capable for inference.

4. **Video Processing**: Without `decord`, video loading may use alternative methods (opencv, imageio).

## Troubleshooting

If you encounter issues:
1. Ensure the virtual environment is activated
2. For missing imports, check if they're CUDA-specific and need MPS alternatives
3. Some scripts may need modification to use `mps` instead of `cuda`

## Next Steps

1. Download the model weights from HuggingFace
2. Test with the sample scripts in the `scripts/inference/` directory
3. Modify any CUDA-specific code to work with MPS

Remember to always activate the virtual environment before running YUME:
```bash
source yume_env/bin/activate
``` 