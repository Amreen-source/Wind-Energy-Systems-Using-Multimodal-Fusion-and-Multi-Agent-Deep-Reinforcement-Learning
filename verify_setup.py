"""
Setup Verification Script
Run this after setup to ensure everything is configured correctly
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠ Warning: Python 3.8+ recommended")
        return False
    return True

def check_packages():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 'matplotlib',
        'sklearn', 'gymnasium', 'stable_baselines3', 'transformers',
        'PIL', 'cv2', 'torch_geometric'
    ]
    
    print("\nChecking packages...")
    all_installed = True
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            elif package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        print(f"\n✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available - {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
        return True
    except:
        return False

def check_datasets():
    """Check if datasets exist"""
    print("\nChecking datasets...")
    
    nrel_path = Path("data/raw/NREL_WIND_DATA")
    blade_path = Path("data/raw/Wind_Turbine_Blade_Defect_Detection")
    
    datasets_ok = True
    
    if nrel_path.exists():
        files = list(nrel_path.glob("*"))
        print(f"✓ NREL_WIND_DATA found ({len(files)} files)")
    else:
        print("✗ NREL_WIND_DATA not found")
        print(f"  Expected at: {nrel_path.absolute()}")
        datasets_ok = False
    
    if blade_path.exists():
        if (blade_path / "Images").exists():
            images = list((blade_path / "Images").glob("*"))
            print(f"✓ Wind Turbine Blade Images found ({len(images)} images)")
        else:
            print("✗ Images folder not found in Wind_Turbine_Blade_Defect_Detection")
            datasets_ok = False
        
        if (blade_path / "Label").exists():
            labels = list((blade_path / "Label").glob("*"))
            print(f"✓ Labels found ({len(labels)} files)")
        else:
            print("✗ Label folder not found in Wind_Turbine_Blade_Defect_Detection")
            datasets_ok = False
    else:
        print("✗ Wind_Turbine_Blade_Defect_Detection not found")
        print(f"  Expected at: {blade_path.absolute()}")
        datasets_ok = False
    
    return datasets_ok

def check_api_key():
    """Check if API key is configured"""
    print("\nChecking API configuration...")
    
    if not Path(".env").exists():
        print("✗ .env file not found")
        return False
    
    with open(".env", "r") as f:
        content = f.read()
        if "your_api_key_here" in content:
            print("⚠ Please update .env with your actual OpenWeatherMap API key")
            print("  Get free key from: https://openweathermap.org/api")
            return False
        elif "OPENWEATHER_API_KEY" in content:
            print("✓ .env file configured")
            return True
    
    return False

def check_directory_structure():
    """Check if all required directories exist"""
    print("\nChecking directory structure...")
    
    required_dirs = [
        "data/raw", "data/processed",
        "models", "results/figures", "results/tables", "results/checkpoints",
        "src", "configs", "logs", "notebooks"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path}")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 60)
    print("  Setup Verification for Predictive Maintenance Project")
    print("=" * 60)
    
    checks = {
        "Python Version": check_python_version(),
        "Packages": check_packages(),
        "CUDA/GPU": check_cuda(),
        "Datasets": check_datasets(),
        "API Key": check_api_key(),
        "Directory Structure": check_directory_structure()
    }
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:.<30} {status}")
    
    all_passed = all(checks.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All checks passed! Ready to start training.")
    else:
        print("⚠ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Move datasets to data/raw/ directory")
        print("- Update .env file with API key")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
