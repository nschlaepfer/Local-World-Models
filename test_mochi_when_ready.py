#!/usr/bin/env python3
"""
Test Mochi Model When Ready
Tests the full video generation pipeline once Mochi model is downloaded
"""

import os
import sys
import time
import torch
from pathlib import Path

def wait_for_model_download():
    """Wait for the Mochi model to finish downloading"""
    model_dir = Path("./mochi-1-preview")
    transformer_dir = model_dir / "transformer"
    
    print("🕒 Waiting for Mochi model download to complete...")
    
    while True:
        if transformer_dir.exists():
            # Check for model files
            model_files = list(transformer_dir.glob("*.safetensors"))
            if model_files:
                print(f"✅ Transformer weights found: {len(model_files)} files")
                return True
        
        print("⏳ Still downloading... (checking every 30 seconds)")
        time.sleep(30)

def test_mochi_pipeline():
    """Test the complete Mochi pipeline"""
    try:
        print("🧪 Testing Mochi Pipeline...")
        
        # Import required modules
        from diffusers import MochiPipeline
        from diffusers.utils import export_to_video
        
        print("✅ Diffusers imported successfully")
        
        # Load the pipeline
        print("📥 Loading Mochi pipeline...")
        pipe = MochiPipeline.from_pretrained(
            "./mochi-1-preview", 
            variant="bf16", 
            torch_dtype=torch.bfloat16
        )
        
        # Enable optimizations for macOS
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        
        print("✅ Pipeline loaded with optimizations")
        
        # Test prompt
        prompt = "A cat walking through a garden, beautiful lighting"
        
        print(f"🎬 Generating video with prompt: '{prompt}'")
        print("⚠️  This may take several minutes...")
        
        # Generate video
        frames = pipe(
            prompt, 
            num_frames=25,  # Shorter for testing
            num_inference_steps=20,  # Fewer steps for testing
            guidance_scale=3.5
        ).frames[0]
        
        # Export video
        output_path = "test_mochi_output.mp4"
        export_to_video(frames, output_path, fps=24)
        
        print(f"🎉 SUCCESS! Video generated: {output_path}")
        print(f"📁 File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Mochi pipeline: {e}")
        return False

def main():
    """Main function"""
    print("🎬 Mochi Model Tester")
    print("=" * 50)
    
    # Check if model is already downloaded
    if Path("./mochi-1-preview/transformer").exists():
        print("✅ Model already downloaded!")
    else:
        # Wait for download
        if not wait_for_model_download():
            print("❌ Model download timeout")
            return
    
    # Test the pipeline
    success = test_mochi_pipeline()
    
    if success:
        print("\n🎉 MOCHI PIPELINE FULLY FUNCTIONAL!")
        print("🎯 You can now generate real videos from text prompts!")
    else:
        print("\n⚠️  Some issues detected. Check error messages above.")

if __name__ == "__main__":
    main() 