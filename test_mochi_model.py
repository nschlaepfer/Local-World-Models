#!/usr/bin/env python3
"""
Test Mochi Model Functionality
Verifies the downloaded Mochi model can be loaded and used
"""

import os
import torch
import sys
from pathlib import Path

def test_mochi_model():
    """Test if the Mochi model can be loaded successfully"""
    print("🔍 Testing Mochi Model Functionality...")
    
    # Set up device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using Apple Silicon MPS: {device}")
    else:
        device = torch.device("cpu")
        print(f"💻 Using CPU: {device}")
    
    # Check model files
    model_dir = Path("./mochi-1-preview")
    if not model_dir.exists():
        print("❌ Mochi model directory not found")
        return False
    
    print(f"✅ Model directory found: {model_dir}")
    print(f"📁 Model size: {sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024**3):.1f} GB")
    
    # Check essential files
    essential_files = [
        "transformer/config.json",
        "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00005.safetensors", 
        "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
        "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
        "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json"
    ]
    
    missing_files = []
    for file_path in essential_files:
        full_path = model_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing {len(missing_files)} essential files")
        return False
    
    print("✅ All essential model files present!")
    
    # Try to import and load the model
    try:
        print("\n🔄 Testing model import...")
        
        # Test if we can import the required modules
        from diffusers import MochiPipeline
        print("✅ MochiPipeline imported successfully")
        
        # Try to load the model (this will test if files are valid)
        print("🔄 Loading Mochi pipeline...")
        pipe = MochiPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16 if device.type == "mps" else torch.float32,
            device_map="auto" if device.type != "mps" else None
        )
        
        if device.type == "mps":
            pipe = pipe.to(device)
        
        print("✅ Mochi model loaded successfully!")
        print(f"🎯 Model ready for text-to-video generation on {device}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("ℹ️  This might require installing diffusers with Mochi support")
        return False
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_simple_generation():
    """Test a very simple generation"""
    try:
        print("\n🎬 Testing simple video generation...")
        
        from diffusers import MochiPipeline
        import torch
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        pipe = MochiPipeline.from_pretrained(
            "./mochi-1-preview",
            torch_dtype=torch.bfloat16 if device.type == "mps" else torch.float32,
        )
        
        if device.type == "mps":
            pipe = pipe.to(device)
        
        # Generate a very short video
        prompt = "A cat walking"
        print(f"🎯 Generating: '{prompt}'")
        
        video = pipe(
            prompt=prompt,
            num_frames=25,  # Very short video
            height=480,
            width=848,
            num_inference_steps=30,
            guidance_scale=4.5,
            generator=torch.Generator().manual_seed(42)
        ).frames[0]
        
        print(f"✅ Generated {len(video)} frames!")
        print("🎉 Mochi model is fully functional!")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 WM-mac Mochi Model Test")
    print("=" * 40)
    
    # Test model loading
    model_works = test_mochi_model()
    
    if model_works:
        # Test simple generation
        generation_works = test_simple_generation()
        
        if generation_works:
            print("\n🎉 SUCCESS! Your Mochi model is ready for video generation!")
        else:
            print("\n⚠️  Model loads but generation needs debugging")
    else:
        print("\n❌ Model test failed - check installation")
    
    print("\n📋 Summary:")
    print(f"  Model Loading: {'✅' if model_works else '❌'}")
    if model_works:
        print(f"  Video Generation: {'✅' if 'generation_works' in locals() and generation_works else '❌'}") 