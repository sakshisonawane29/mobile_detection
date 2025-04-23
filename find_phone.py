""" This module is used to find coordinates of the phone in the image."""

# Loading dependencies.
import os
import sys
import argparse

# Data pre-processing libraries.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Machine learning modelling libraries.
import tensorflow as tf
from tensorflow.keras.models import load_model

def find_phone(image_path, model_path=None, input_size=224, visualize=False):
    """ This function is used to predict coordinates using the trained model.
    
    Args:
        image_path (str): Path to the input image
        model_path (str, optional): Path to the model file. If None, will look in standard locations
        input_size (int): Size to resize the input image to
        visualize (bool): Whether to display and save visualization of the prediction
        
    Returns:
        tuple: (x, y) coordinates of the detected phone
    """
    # Process the path to ensure it's usable
    image_path = os.path.abspath(image_path)
    file_name = os.path.basename(image_path)
    
    # Handle model path
    if model_path is None:
        # Try to find the model in standard locations
        possible_model_paths = [
            'Results/mobile_detector_model.h5',         # New model path
            'Results/best_model.h5',                    # Best model from checkpoints
            os.path.join(os.path.dirname(image_path), 'mobile_detector_model.h5'),
            os.path.join(os.path.dirname(image_path), 'train_phone_finder_weights.h5')  # Original model path
        ]
        
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        if model_path is None:
            raise FileNotFoundError("Could not find model weights. Please specify model_path.")
    
    # Ensure model path is absolute
    model_path = os.path.abspath(model_path)
    
    print(f"Loading model from: {model_path}")
    print(f"Processing image: {image_path}")

    # Reading the image file
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
        
    # Store original dimensions for visualization
    h, w = original_img.shape[:2]
    
    # Process the image
    img = cv2.resize(original_img, (input_size, input_size))
    x_variable = np.array([img])
    x_variable = np.interp(x_variable, (x_variable.min(), x_variable.max()), (0, 1))

    # Loading the Deep Learning Model
    model = load_model(model_path)
    
    # Predict phone location
    result = model.predict(x_variable)
    
    # Scale coordinates back to original image dimensions
    x_coord, y_coord = result[0][0], result[0][1]
    
    # Print results
    print("\nPhone in image {0} is located at x-y coordinates given below:".format(file_name))
    print("\n{:.4f} {:.4f}".format(x_coord, y_coord))
    
    # Visualize results if requested
    if visualize:
        # Convert to the original image space for visualization
        orig_x, orig_y = int(x_coord * w), int(y_coord * h)
        
        # Create a copy for visualization
        vis_img = original_img.copy()
        
        # Draw a circle at the predicted phone location
        cv2.circle(vis_img, (orig_x, orig_y), radius=max(5, int(min(h, w) * 0.02)), 
                  color=(0, 255, 0), thickness=-1)
        
        # Draw crosshairs
        cv2.line(vis_img, (orig_x, 0), (orig_x, h), (0, 255, 0), thickness=2)
        cv2.line(vis_img, (0, orig_y), (w, orig_y), (0, 255, 0), thickness=2)
        
        # Convert BGR to RGB for matplotlib
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        # Create figure and display
        plt.figure(figsize=(12, 10))
        plt.imshow(vis_img)
        plt.title(f"Phone Detection: ({x_coord:.4f}, {y_coord:.4f})")
        plt.axis('off')
        
        # Save visualization
        os.makedirs('Results/visualizations', exist_ok=True)
        output_path = os.path.join('Results/visualizations', f"detection_{os.path.splitext(file_name)[0]}.jpg")
        plt.savefig(output_path, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        
        # Display (this will be skipped in headless environments)
        try:
            plt.show()
        except:
            print("Could not display visualization (possibly running in headless mode)")
    
    return x_coord, y_coord

def main():
    """ This function is used to run the program with command line arguments. """
    parser = argparse.ArgumentParser(description='Find a phone in an image')
    parser.add_argument('image_path', help='Path to the JPEG image to be tested')
    parser.add_argument('--model_path', help='Path to the model weights file')
    parser.add_argument('--input_size', type=int, default=224, 
                        help='Input image size (default: 224, must match training size)')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize the prediction')
    
    args = parser.parse_args()
    
    find_phone(
        args.image_path, 
        model_path=args.model_path, 
        input_size=args.input_size, 
        visualize=args.visualize
    )

if __name__ == "__main__":
    main()
