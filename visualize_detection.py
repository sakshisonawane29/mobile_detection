"""
Utility script to visualize phone detection results.
This can be used to view previously detected phone locations or to run a demo on a batch of images.
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from find_phone import find_phone

def visualize_detection_from_coordinates(image_path, x_coord, y_coord, output_dir='Results/visualizations'):
    """
    Visualize phone detection using given coordinates
    
    Args:
        image_path (str): Path to the image
        x_coord (float): Normalized x-coordinate (0-1)
        y_coord (float): Normalized y-coordinate (0-1)
        output_dir (str): Directory to save visualizations
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
        
    # Get dimensions
    h, w = img.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    orig_x, orig_y = int(x_coord * w), int(y_coord * h)
    
    # Create a copy for visualization
    vis_img = img.copy()
    
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
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"detection_{os.path.splitext(filename)[0]}.jpg")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Display the image
    try:
        plt.show()
    except:
        print("Could not display visualization (possibly running in headless mode)")

def batch_detect_and_visualize(image_dir, model_path=None, input_size=224, output_dir='Results/visualizations'):
    """
    Run detection on a batch of images and visualize results
    
    Args:
        image_dir (str): Directory containing images to process
        model_path (str): Path to model weights
        input_size (int): Image size for model input
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, ext)))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
        
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for img_path in image_files:
        try:
            print(f"\nProcessing {img_path}...")
            x, y = find_phone(img_path, model_path=model_path, input_size=input_size, visualize=True)
            results.append((img_path, x, y))
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, 'detection_results.csv')
    with open(csv_path, 'w') as f:
        f.write("image,x_coord,y_coord\n")
        for img_path, x, y in results:
            img_name = os.path.basename(img_path)
            f.write(f"{img_name},{x:.4f},{y:.4f}\n")
    
    print(f"\nResults saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize phone detection results')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Parser for visualizing from coordinates
    coords_parser = subparsers.add_parser('coords', help='Visualize detection from coordinates')
    coords_parser.add_argument('image_path', help='Path to the image')
    coords_parser.add_argument('x_coord', type=float, help='X coordinate (0-1)')
    coords_parser.add_argument('y_coord', type=float, help='Y coordinate (0-1)')
    coords_parser.add_argument('--output_dir', default='Results/visualizations', 
                              help='Directory to save visualizations')
    
    # Parser for batch detection
    batch_parser = subparsers.add_parser('batch', help='Run detection on a batch of images')
    batch_parser.add_argument('image_dir', help='Directory containing images')
    batch_parser.add_argument('--model_path', help='Path to model weights')
    batch_parser.add_argument('--input_size', type=int, default=224, 
                             help='Input image size (default: 224)')
    batch_parser.add_argument('--output_dir', default='Results/visualizations', 
                             help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    if args.mode == 'coords':
        visualize_detection_from_coordinates(
            args.image_path, 
            args.x_coord, 
            args.y_coord,
            args.output_dir
        )
    elif args.mode == 'batch':
        batch_detect_and_visualize(
            args.image_dir,
            args.model_path,
            args.input_size,
            args.output_dir
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 