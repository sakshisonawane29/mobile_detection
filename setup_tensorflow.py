"""
Setup and check TensorFlow environment for Windows.

This script:
1. Checks your Python version
2. Checks for Visual C++ Redistributable
3. Uninstalls any existing TensorFlow
4. Installs a compatible version of TensorFlow for Windows
5. Tests if TensorFlow loads correctly
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if the Python version is compatible."""
    print(f"Python version: {platform.python_version()}")
    major, minor, _ = platform.python_version_tuple()
    if int(major) != 3 or int(minor) < 7 or int(minor) > 9:
        print("Warning: TensorFlow 2.10 works best with Python 3.7 to 3.9 on Windows")
        print(f"Your Python version is {platform.python_version()}")
        return False
    return True

def check_visual_cpp():
    """Check if Microsoft Visual C++ Redistributable is installed."""
    print("Checking for Microsoft Visual C++ Redistributable...")
    # This is a simple check - we can't reliably detect all versions
    if platform.system() != "Windows":
        print("Not on Windows. Skipping Visual C++ check.")
        return True
        
    try:
        # We'll use a simple test - try to import a DLL-dependent package
        import numpy
        print("NumPy loaded successfully, which suggests VC++ Redistributable is present.")
        return True
    except ImportError:
        print("Error importing NumPy.")
        return False
    except Exception as e:
        if "DLL load failed" in str(e):
            print("Microsoft Visual C++ Redistributable appears to be missing.")
            print("Please download and install from: https://aka.ms/vs/16/release/vc_redist.x64.exe")
            return False
        else:
            print(f"Unknown error: {e}")
            return False

def install_package(package):
    """Install a package using pip."""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False

def uninstall_package(package):
    """Uninstall a package using pip."""
    try:
        print(f"Uninstalling {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to uninstall {package}")
        return False

def install_tensorflow():
    """Install TensorFlow 2.10.0 which is more compatible with Windows."""
    # First uninstall any existing TensorFlow
    uninstall_package("tensorflow")
    
    # Install required packages
    install_package("numpy")
    install_package("keras")
    
    # Install TensorFlow 2.10.0
    success = install_package("tensorflow==2.10.0")
    
    if success:
        print("TensorFlow 2.10.0 installed successfully.")
    else:
        print("Failed to install TensorFlow 2.10.0.")
        
    return success

def test_tensorflow():
    """Test if TensorFlow can be imported correctly."""
    print("\nTesting TensorFlow import...")
    try:
        # Set environment variable to use CPU only
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Try importing TensorFlow
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print("TensorFlow imported successfully!")
        
        # Try a simple operation
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[1, 1], [1, 1]])
        c = tf.matmul(a, b)
        print("Test computation successful:")
        print(c)
        return True
    except ImportError as e:
        print(f"Error importing TensorFlow: {e}")
        return False
    except Exception as e:
        print(f"Error during TensorFlow test: {e}")
        return False

def main():
    print("TensorFlow Setup and Environment Checker for Windows")
    print("==================================================\n")
    
    # Check Python version
    python_ok = check_python_version()
    if not python_ok:
        print("Warning: Your Python version might not be optimal for TensorFlow on Windows")
    
    # Check for Visual C++ Redistributable
    vcpp_ok = check_visual_cpp()
    if not vcpp_ok:
        print("\nPlease install the Microsoft Visual C++ Redistributable first.")
        print("Download link: https://aka.ms/vs/16/release/vc_redist.x64.exe")
        choice = input("Continue anyway? (y/n): ")
        if choice.lower() != 'y':
            return
    
    # Ask to install TensorFlow
    choice = input("\nReinstall TensorFlow with a compatible version? (y/n): ")
    if choice.lower() == 'y':
        install_tensorflow()
    
    # Test TensorFlow
    test_ok = test_tensorflow()
    if test_ok:
        print("\nSuccess! TensorFlow is working properly.")
        print("\nYou can now run the video detection script with:")
        print("python detect_phone_in_video_cpu.py --webcam")
    else:
        print("\nTensorFlow test failed.")
        print("Please try the following troubleshooting steps:")
        print("1. Make sure Microsoft Visual C++ Redistributable is installed")
        print("2. Restart your computer and try again")
        print("3. Create a new virtual environment with Python 3.8")
        print("4. Install TensorFlow 2.10.0 in the new environment")
        print("5. Visit https://www.tensorflow.org/install/errors for more help")

if __name__ == "__main__":
    main() 