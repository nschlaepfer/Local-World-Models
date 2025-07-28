#!/usr/bin/env python3
"""
Simple test script for WM-mac on Apple Silicon
Tests basic functionality without requiring specific model weights
"""

import os
import torch
import sys
import traceback

def test_basic_setup():
    """Test basic system setup"""
    print("🔍 Testing WM-mac Basic Setup...")
    
    # Test Python version
    print(f"✅ Python version: {sys.version[:5]}")
    
    # Test PyTorch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ MPS available: {torch.backends.mps.is_available()}")
    
    # Test device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Using device: {device} (MPS not available)")
    
    return True

def test_wan_import():
    """Test WAN module import"""
    print("\n🔍 Testing WAN Module Import...")
    try:
        import wan
        print("✅ WAN module imported successfully")
        
        # Test WAN configs
        from wan.configs import WAN_CONFIGS
        print(f"✅ Available configs: {list(WAN_CONFIGS.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ WAN import failed: {e}")
        traceback.print_exc()
        return False

def test_image_processing():
    """Test basic image processing"""
    print("\n🔍 Testing Image Processing...")
    try:
        from PIL import Image
        import numpy as np
        
        # Check if we have test images
        jpg_dir = "./jpg"
        if os.path.exists(jpg_dir):
            images = [f for f in os.listdir(jpg_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                print(f"✅ Found {len(images)} test images")
                
                # Try loading one image
                test_image = os.path.join(jpg_dir, images[0])
                img = Image.open(test_image)
                print(f"✅ Successfully loaded test image: {img.size}")
                return True
            else:
                print("⚠️  No images found in jpg directory")
        else:
            print("⚠️  jpg directory not found")
        
        return False
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def test_model_files():
    """Test available model files"""
    print("\n🔍 Testing Model Files...")
    
    model_dirs = [
        "./HunyuanVideo-I2V",
        "./Yume-I2V-540P", 
        "./data"
    ]
    
    found_models = 0
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"✅ Found model directory: {model_dir}")
            found_models += 1
            
            # Check size
            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                      for dirpath, dirnames, filenames in os.walk(model_dir)
                      for filename in filenames) / (1024*1024*1024)  # GB
            print(f"   📁 Size: {size:.1f} GB")
            
            # Check for common model files
            for root, dirs, files in os.walk(model_dir):
                model_files = [f for f in files if f.endswith(('.pt', '.safetensors', '.bin', '.ckpt'))]
                if model_files:
                    print(f"   📄 Model files found: {len(model_files)}")
                    break
        else:
            print(f"❌ Model directory not found: {model_dir}")
    
    return found_models > 0

def main():
    """Run all tests"""
    print("🎬 WM-mac System Test")
    print("=" * 50)
    
    tests = [
        test_basic_setup,
        test_wan_import,
        test_image_processing,
        test_model_files
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    passed = sum(results)
    total = len(results)
    print(f"✅ {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your WM-mac setup is working!")
    elif passed > 0:
        print("⚠️  Partial setup working. Some features may be limited.")
    else:
        print("❌ Setup has issues. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 