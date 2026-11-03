"""
TagMed Setup Script - Cross-platform installation for Windows, macOS, and Linux

This script automatically sets up the TagMed medical annotation application
with all required dependencies across different operating systems.

Features:
- Cross-platform virtual environment creation
- Automatic dependency installation
- SAM2 and MedSAM2 model preparation
- System requirement validation
- Platform-specific path handling

Supported Platforms:
- Windows (Python 3.8+)
- macOS (Python 3.8+) 
- Linux (Python 3.8+)

Author: Janik Hohmann
Institution: University Hospital Düsseldorf
"""

import subprocess
import os
import sys
import platform
import shutil
import urllib.request


def get_platform_info():
    """
    Detect the current operating system and return platform-specific configurations.

    Returns:
        tuple: (platform_name, python_executable, pip_executable,
                activate_script, python_cmd)
    """

    system = platform.system().lower()

    if system == "windows":
        env_python = os.path.join("tagmed-env", "Scripts", "python.exe")
        env_pip = os.path.join("tagmed-env", "Scripts", "pip.exe")
        activate_script = os.path.join("tagmed-env", "Scripts", "activate.bat")

        # Prefer Python 3.11 launcher if available
        python_cmd = "py -3.11"

    else:  # Linux + macOS
        env_python = os.path.join("tagmed-env", "bin", "python")
        env_pip = os.path.join("tagmed-env", "bin", "pip")
        activate_script = os.path.join("tagmed-env", "bin", "activate")

        # Prefer Python 3.11
        if shutil.which("python3.11"):
            python_cmd = "python3.11"
        elif shutil.which("python3.10"):
            python_cmd = "python3.10"
        elif shutil.which("python3"):
            python_cmd = "python3"
        else:
            print(" Error: No compatible Python installation found.")
            print(" Please install Python 3.10 or 3.11.")
            sys.exit(1)

    return system, env_python, env_pip, activate_script, python_cmd


def run_command(cmd, description=""):
    """
    Execute a shell command with cross-platform compatibility.
    
    Args:
        cmd (str): Command to execute
        description (str): Description for logging
        
    Returns:
        str: Command output
        
    Raises:
        SystemExit: If command fails
    """
    if description:
        print(f"  {description}")
    print(f"  Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"    \nError: {e.stderr}")
        print(f"    Command failed with return code: {e.returncode}")
        sys.exit(1)


def check_system_requirements():
    """
    Validate system requirements before installation.
    
    Checks:
    - Python version compatibility (3.8+)
    - Available disk space (10GB+ recommended)
    - Internet connectivity
    - Required system packages
    """
    print(" Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"Error: Python 3.8+ required (found {python_version.major}.{python_version.minor})")
        sys.exit(1)
    print(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check available disk space
    try:
        free_space = shutil.disk_usage('.').free / (1024**3)  # Convert to GB
        if free_space < 10:
            print(f"  Warning: Low disk space ({free_space:.1f}GB). Recommend 10GB+ for models")
        else:
            print(f" Disk space: {free_space:.1f}GB available")
    except Exception:
        print("  Could not check disk space")
    
    # Check internet connectivity
    try:
        urllib.request.urlopen('https://pypi.org', timeout=10)
        print(" Internet connectivity verified")
    except Exception:
        print("  Warning: Internet connectivity check failed")
    
    # Platform-specific checks
    system = platform.system().lower()
    print(f" Platform: {platform.system()} {platform.release()}")
    
    if system == "linux":
        # Check for common missing packages on Linux
        print("  Note: On Linux, you may need to install: python3-tk python3-dev build-essential")
    elif system == "darwin":  # macOS
        print("  Note: On macOS, ensure Xcode Command Line Tools are installed")
    elif system == "windows":
        print("  Note: On Windows, ensure Microsoft Visual C++ 14.0+ is available")


def create_virtual_environment(python_cmd):
    """
    Create a Python virtual environment.
    
    Args:
        python_cmd (str): Python executable command
    """
    env_dir = "tagmed-env"
    
    if not os.path.exists(env_dir):
        print(f"\n [1/7] Creating virtual environment...")
        run_command(f"{python_cmd} -m venv {env_dir}", "Creating virtual environment")
        print(f" Virtual environment created at: {env_dir}")
    else:
        print(f"\n [1/7] Virtual environment already exists at: {env_dir}")


def upgrade_pip_and_tools(env_pip):
    """
    Upgrade pip and install essential build tools.
    
    Args:
        env_pip (str): Path to pip executable in virtual environment
    """
    print(f"\n [2/7] Upgrading pip and installing build tools...")
    
    essential_tools = ["pip", "wheel", "setuptools"]
    for tool in essential_tools:
        run_command(f"{env_pip} install --upgrade {tool}", f"Upgrading {tool}")
    
    print(" Build tools updated")


def install_core_packages(env_pip):
    """
    Install core Python packages required by TagMed.
    
    Args:
        env_pip (str): Path to pip executable in virtual environment
    """
    print(f"\n [3/7] Installing core Python packages...")
    
    print("")
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV')}")
    print(f"Using pip from: {env_pip}")
    subprocess.run([env_pip, "--version"], check=True)
    print("")

    # PyTorch: pick an install command appropriate for the platform. The CUDA 12.1
    # index has no macOS wheels, so macOS uses the default PyPI wheels (CPU/MPS).
    # Override manually for a different CUDA version or a CPU-only Linux/Windows install.
    if platform.system().lower() == "darwin":
        torch_pkg = "torch torchvision torchaudio"
    else:
        torch_pkg = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"

    core_packages = [
        torch_pkg,                        # PyTorch ecosystem
        "opencv-python",                  # Computer vision
        "pandas",                         # Data manipulation
        "numpy",                          # Numerical computing
        "tqdm",                          # Progress bars
        "requests",                       # HTTP requests
        "hydra-core",                    # Configuration management
        "pillow",                        # Image processing
        "scikit-image",                  # Scientific image processing
        "matplotlib",                    # Plotting
        "einops",                        # Tensor operations
    ]
    
    for package in core_packages:
        package_name = package.split()[0]  # Get first word for display
        run_command(f"{env_pip} install {package}", f"Installing {package_name}")
    
    print(" Core packages installed")


def install_sam_packages(env_pip):
    """
    Install SAM and MedSAM packages with error handling.
    
    Args:
        env_pip (str): Path to pip executable in virtual environment
    """
    print(f"\n [4/7] Installing SAM and MedSAM packages...")
    
    sam_packages = [
        ("segment-anything", "Original SAM"),
        ("sam2", "SAM 2.0"),
    ]
    
    for package, description in sam_packages:
        try:
            run_command(f"{env_pip} install {package}", f"Installing {description}")
        except SystemExit:
            print(f"  Warning: Could not install {package} from PyPI")
            print(f"   You may need to install it manually from GitHub")
            # Don't exit, continue with other packages
            pass
    
    print(" SAM packages installation attempted")


def verify_gui_dependencies(env_python):
    """
    Verify GUI dependencies are available.
    
    Args:
        env_python (str): Path to python executable in virtual environment
    """
    print(f"\n  [5/7] Verifying GUI dependencies...")
    
    try:
        run_command(f"{env_python} -c \"import tkinter; print('tkinter available')\"", 
                   "Checking tkinter availability")
        print(" tkinter is available")
    except SystemExit:
        system = platform.system().lower()
        if system == "linux":
            print(" tkinter not available. Install with: sudo apt-get install python3-tk")
        elif system == "darwin":
            print(" tkinter not available. Install Python from python.org or use Homebrew")
        elif system == "windows":
            print(" tkinter not available. Reinstall Python with tkinter support")
        sys.exit(1)


def create_project_structure():
    """
    Create necessary project directories.
    """
    print(f"\n [6/7] Creating project structure...")
    
    directories = [
        "models",          # For SAM/MedSAM model storage
        "masks"            # For generated mask storage
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"   Created directory: {directory}")
        else:
            print(f"   Directory exists: {directory}")
    
    print(" Project structure created")


def finalize_setup(system, activate_script):
    """
    Display final setup instructions and summary.
    
    Args:
        system (str): Operating system name
        activate_script (str): Path to activation script
    """
    print(f"\n [7/7] Finalizing setup...")
    
    # Create a startup script for convenience
    if system == "windows":
        startup_script = "start_tagmed.bat"
        with open(startup_script, 'w') as f:
            f.write("@echo off\n")
            # f.write("tagmed-env\\Scripts\\python.exe src\\main.py\n")
            # f.write("pause\n")
            f.write("cmd /k \"call tagmed-env\\Scripts\\activate.bat && cd src && python main.py\"\n")
        print(f"   Created startup script: {startup_script}")
    else:
        startup_script = "start_tagmed.sh"
        with open(startup_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("source tagmed-env/bin/activate\n")
            f.write("cd src\n")
            f.write("python main.py\n")
        os.chmod(startup_script, 0o755)  # Make executable
        print(f"   Created startup script: {startup_script}")
    
    print(" Setup finalized")


def main_setup():
    """
    Main setup function coordinating the entire installation process.
    """
    print("")
    print("=" * 70)
    print("TagMed Setup Script - Medical Annotation Tool")
    print("=" * 70)
    
    # Detect platform and get configuration
    system, env_python, env_pip, activate_script, python_cmd = get_platform_info()
    
    print(f" Platform detected: {platform.system()} {platform.machine()}")
    
    # Run setup steps
    check_system_requirements()
    create_virtual_environment(python_cmd)
    upgrade_pip_and_tools(env_pip)
    install_core_packages(env_pip)
    install_sam_packages(env_pip)
    verify_gui_dependencies(env_python)
    create_project_structure()
    finalize_setup(system, activate_script)
    
    # Display completion message
    print("\n" + "=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    
    print("\nTo start using TagMed:")
    if system == "windows":
        print("   Option 1: Double-click start_tagmed.bat")
        print("   Option 2: Manual activation:")
        print("     1. tagmed-env\\Scripts\\activate.bat (or Activate.ps1 if you are using PowerShell)" )
        print("     2. cd src")
        print("     3. python main.py")
    else:
        print("   Option 1: ./start_tagmed.sh")
        print("   Option 2: Manual activation:")
        print(f"     1. source {activate_script}")
        print("     2. cd src")
        print("     3. python main.py")
    
    print("\n Setup Summary:")
    print("   Virtual environment created")
    print("   Dependencies installed")
    print("   Project structure created")
    print("   Startup scripts generated")
    
    print("\n Important Notes:")
    print("   • SAM2 models (~2-5GB) will download automatically on first use")
    print("   • MedSAM2 models will download automatically when selected")
    print("   • Configure your data directory in the application settings")
    print("   • Ensure sufficient disk space for model storage")
    
    if system == "linux":
        print("   • GPU support requires CUDA-compatible drivers")
    elif system == "darwin":
        print("   • GPU acceleration available on M1/M2 Macs via MPS")
    elif system == "windows":
        print("   • GPU support requires CUDA toolkit and compatible drivers")


if __name__ == "__main__":
    try:
        main_setup()
    except KeyboardInterrupt:
        print("\n\n  Setup interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Unexpected error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)