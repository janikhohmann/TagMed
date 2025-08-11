import subprocess
import os
import sys


def main_setup():    
    """
    Main setup function to create a virtual environment, install required packages,
    and prepare the TagMed project for use.
    """

    def run(cmd):
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)
        return result.stdout

    print("=" * 60)
    print("TagMed Setup Script")
    print("=" * 60)

    # 1. Create virtual environment
    env_dir = "tagmed-env"
    if not os.path.exists(env_dir):
        print("\n[1/6] Creating virtual environment...")
        run(f"python3 -m venv {env_dir}")
        print(f"✓ Virtual environment created at: {env_dir}")
    else:
        print(f"\n[1/6] Virtual environment already exists at: {env_dir}")

    # 2. Upgrade pip and install build tools
    print("\n[2/6] Upgrading pip and installing build tools...")
    run(f"{env_dir}/bin/pip install --upgrade pip wheel setuptools")
    print("✓ Build tools updated")

    # 3. Install core dependencies
    print("\n[3/6] Installing core Python packages...")
    core_packages = [
        "torch", "torchvision", "torchaudio",  # PyTorch ecosystem
        "opencv-python",                        # Computer vision
        "pandas", "numpy",                      # Data processing
        "tqdm", "requests",                     # Utilities
        "hydra-core",                          # Configuration management
        "pillow",                              # Image processing
        "scikit-image",                        # Scientific image processing
        "matplotlib",                          # Plotting
        "einops",                              # Tensor operations
    ]
    
    for package in core_packages:
        print(f"  Installing {package}...")
        run(f"{env_dir}/bin/pip install {package}")
    print("✓ Core packages installed")

    # 4. Install SAM/MedSAM packages
    print("\n[4/6] Installing SAM and MedSAM packages...")
    sam_packages = [
        "segment-anything",  # Original SAM
        "sam2",             # SAM 2.0
    ]
    
    for package in sam_packages:
        print(f"  Installing {package}...")
        try:
            run(f"{env_dir}/bin/pip install {package}")
        except:
            print(f"  Warning: Could not install {package} from PyPI")
            print(f"  You may need to install it manually from GitHub")
    print("✓ SAM packages installed")

    # 5. Install GUI dependencies
    print("\n[5/6] Installing GUI dependencies...")
    gui_packages = [
        "tkinter",  # Usually comes with Python, but just in case
    ]
    
    # tkinter is usually built-in, so we'll just verify it works
    try:
        run(f"{env_dir}/bin/python -c 'import tkinter; print(\"tkinter available\")'")
        print("✓ tkinter is available")
    except:
        print("⚠ Warning: tkinter not available. You may need to install python3-tk")

    # 6. Create directories and final setup
    print("\n[6/6] Final setup...")
    
    # Create necessary directories
    directories = ["models", "masks"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"  Created directory: {directory}")
    
    print("✓ Directory structure created")

    # 7. Installation summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nTo start using TagMed:")
    print(f"1. Activate the environment: source {env_dir}/bin/activate")
    print("2. Navigate to src directory: cd src")
    print("3. Run TagMed: python main.py")

    print("\nPlease ensure your ImageDataset is properly set up, as described in the documentation.")
    print("You can specify the directory and the AnnotationTable after starting the application.")

    # List installed packages
    # print("\nInstalled packages:")
    # try:
    #     packages_output = run(f"{env_dir}/bin/pip list")
    #     print("\n" + packages_output)
    # except:
    #     print("Could not list installed packages")

    print("\nNotes:")
    print("- SAM2 models will be downloaded automatically when first used")
    print("- MedSAM2 models will be downloaded automatically when first used")
    print("- Make sure you have sufficient disk space for model downloads (~2-5GB)")
    print("- CUDA support requires compatible NVIDIA drivers")


def check_system_requirements():
    """Check system requirements before installation"""
    print("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Error: Python 3.8+ required")
        sys.exit(1)
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check available disk space
    import shutil
    free_space = shutil.disk_usage('.').free / (1024**3)  # GB
    if free_space < 10:
        print(f"Warning: Low disk space ({free_space:.1f}GB). Recommend 10GB+ for models")
    else:
        print(f"✓ Disk space: {free_space:.1f}GB available")


if __name__ == "__main__":
    check_system_requirements()
    main_setup()