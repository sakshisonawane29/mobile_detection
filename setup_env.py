"""
Simple script to install the minimal necessary dependencies for phone detection.
"""

import subprocess
import sys

def install(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    # Core packages needed for our application
    packages = [
        "tensorflow",      # Core ML framework
        "opencv-python",   # Image processing
        "matplotlib",      # Visualization
        "scikit-learn",    # For train/test split
        "numpy",           # Numerical operations
        "pillow"           # Image support
    ]
    
    print("Installing necessary packages...")
    install(packages)
    print("Installation complete!")
    
    # Test imports
    try:
        import tensorflow as tf
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        
        # Print versions to verify
        print("\nInstalled package versions:")
        print(f"TensorFlow: {tf.__version__}")
        print(f"OpenCV: {cv2.__version__}")
        print(f"NumPy: {np.__version__}")
        
        print("\nAll dependencies installed successfully. You can now run the following commands:")
        print("1. To prepare the dataset: python prepare_data.py")
        print("2. To train the model: python train_phone_finder.py phone_dataset --input_size 128 --epochs 50")
        print("3. To detect phones in an image: python find_phone.py PATH_TO_IMAGE --visualize")
        print("4. To process multiple images: python visualize_detection.py batch PATH_TO_IMAGES_FOLDER")
        
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please try installing the dependencies manually using pip install <package_name>") 