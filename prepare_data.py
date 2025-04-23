"""
Prepare dataset by converting XML annotations to the required labels.txt format.

This script takes XML annotation files (common format in object detection datasets)
and converts them to a flat labels.txt file with format:
    image_filename.jpg x_coordinate y_coordinate

where x and y are normalized coordinates (0-1) of the center of the phone.
"""

import os
import sys
import xml.etree.ElementTree as ET
import shutil
from glob import glob

def create_training_directory(output_dir='phone_dataset'):
    """Create a directory with the format expected by the training script."""
    if os.path.exists(output_dir):
        print(f"Warning: {output_dir} already exists. Removing its contents.")
        shutil.rmtree(output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created training directory: {output_dir}")
    return output_dir

def parse_xml_annotation(xml_path):
    """Parse XML annotation file to extract bounding box coordinates."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        
        # Find the phone object (assuming there's only one phone per image)
        for obj in root.findall('object'):
            obj_name = obj.find('name').text.lower()
            if 'phone' in obj_name or 'mobile' in obj_name or 'smartphone' in obj_name or 'device' in obj_name:
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Calculate center coordinates and normalize to 0-1 range
                center_x = ((xmin + xmax) / 2) / width
                center_y = ((ymin + ymax) / 2) / height
                
                return center_x, center_y
                
        print(f"Warning: No phone object found in {xml_path}")
        return None
    except Exception as e:
        print(f"Error parsing {xml_path}: {str(e)}")
        return None

def process_dataset(images_dir, annotations_dir, output_dir):
    """Process the dataset and create labels.txt."""
    # Find all XML files
    xml_files = glob(os.path.join(annotations_dir, '*.xml'))
    if not xml_files:
        print(f"No XML files found in {annotations_dir}")
        return False
        
    # Create labels.txt
    labels_file_path = os.path.join(output_dir, 'labels.txt')
    processed_count = 0
    
    with open(labels_file_path, 'w') as labels_file:
        for xml_path in xml_files:
            try:
                # Get the XML filename and base name without extension
                xml_filename = os.path.basename(xml_path)
                base_name = os.path.splitext(xml_filename)[0]
                
                # Get corresponding image filename with the same base name
                image_filename = f"{base_name}.jpg"
                image_path = os.path.join(images_dir, image_filename)
                
                # Check if image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                
                # Parse XML and get center coordinates
                coords = parse_xml_annotation(xml_path)
                if coords is None:
                    continue
                    
                center_x, center_y = coords
                
                # Create a simplified filename for labels.txt to avoid spaces
                # This is important for the training script
                simple_filename = f"image_{processed_count}.jpg"
                
                # Write to labels.txt in the format: simple_filename x_coord y_coord
                labels_file.write(f"{simple_filename} {center_x:.6f} {center_y:.6f}\n")
                
                # Copy image file to output directory with the simplified name
                dest_path = os.path.join(output_dir, simple_filename)
                shutil.copy(image_path, dest_path)
                
                processed_count += 1
                print(f"Processed {processed_count}: {image_filename} → {simple_filename}")
                
            except Exception as e:
                print(f"Error processing {xml_path}: {str(e)}")
    
    print(f"Processed {processed_count} images with annotations")
    print(f"Labels file created at: {labels_file_path}")
    return processed_count > 0

def main():
    # Paths to your dataset
    images_dir = 'data/Mobile_image/Mobile_image'
    annotations_dir = 'data/Annotations/Annotations'
    
    # Create training directory
    output_dir = create_training_directory()
    
    # Process the dataset
    success = process_dataset(images_dir, annotations_dir, output_dir)
    
    if success:
        print("\nData preparation complete!")
        print("To train the model, run:")
        print(f"python train_phone_finder.py {output_dir}")
        print("\nFor faster training with a smaller image size:")
        print(f"python train_phone_finder.py {output_dir} --input_size 128 --epochs 50")
    else:
        print("\nError preparing data. Please check the paths and try again.")

if __name__ == "__main__":
    main() 