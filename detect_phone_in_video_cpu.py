"""
Detect phones in video files or webcam stream - CPU Version.

This version forces TensorFlow to use CPU only, which helps avoid DLL loading errors on Windows.
"""

# Force CPU only mode before importing TensorFlow
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only

import sys
import argparse
import cv2
import numpy as np
import time

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    # Disable eager execution for better compatibility
    tf.compat.v1.disable_eager_execution()
    print("TensorFlow successfully loaded in CPU-only mode")
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("\nTroubleshooting steps:")
    print("1. Make sure you have the Microsoft Visual C++ Redistributable installed")
    print("2. Try reinstalling TensorFlow with: pip uninstall tensorflow && pip install tensorflow==2.10.0")
    print("3. Visit: https://www.tensorflow.org/install/errors for more information")
    sys.exit(1)

try:
    from tensorflow.keras.models import load_model
except ImportError as e:
    print(f"Error importing Keras: {e}")
    sys.exit(1)

def find_phone_in_frame(frame, model, input_size=224):
    """
    Detect phone in a single video frame
    
    Args:
        frame: The video frame (BGR format from OpenCV)
        model: The loaded TensorFlow model
        input_size: Input size for the model
        
    Returns:
        tuple: (x, y) normalized coordinates and the processed frame with visualization
    """
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Process the frame
    processed_frame = cv2.resize(frame, (input_size, input_size))
    x_variable = np.array([processed_frame])
    x_variable = np.interp(x_variable, (x_variable.min(), x_variable.max()), (0, 1))
    
    # Predict phone location
    result = model.predict(x_variable, verbose=0)
    x_coord, y_coord = result[0][0], result[0][1]
    
    # Convert to pixel coordinates
    pixel_x, pixel_y = int(x_coord * w), int(y_coord * h)
    
    # Draw visualization on the original frame
    output_frame = frame.copy()
    
    # Draw a circle at the predicted location
    cv2.circle(output_frame, (pixel_x, pixel_y), radius=max(5, int(min(h, w) * 0.02)),
              color=(0, 255, 0), thickness=-1)
    
    # Draw crosshairs
    cv2.line(output_frame, (pixel_x, 0), (pixel_x, h), (0, 255, 0), thickness=2)
    cv2.line(output_frame, (0, pixel_y), (w, pixel_y), (0, 255, 0), thickness=2)
    
    # Add text with coordinates
    cv2.putText(output_frame, f"Phone: ({x_coord:.3f}, {y_coord:.3f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return (x_coord, y_coord, output_frame)

def process_video(video_path, model_path=None, input_size=224, output_path=None, display=True, webcam=False):
    """
    Process a video file or webcam stream for phone detection
    
    Args:
        video_path: Path to video file or camera index (int) for webcam
        model_path: Path to the model file
        input_size: Input size for the model
        output_path: Path to save the output video (None to skip saving)
        display: Whether to display the output video
        webcam: Whether to use webcam as input
    """
    # Handle model path
    if model_path is None:
        # Try to find the model in standard locations
        possible_model_paths = [
            'Results/mobile_detector_model.h5',
            'Results/best_model.h5',
        ]
        
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        if model_path is None:
            raise FileNotFoundError("Could not find model weights. Please specify model_path.")
    
    # Load the model
    print(f"Loading model from: {model_path}")
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set up video capture
    if webcam:
        # Use webcam (0 is usually the built-in webcam)
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0 if video_path is None else int(video_path))
        print("Accessing webcam...")
    else:
        # Use video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        print(f"Processing video: {video_path}")
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source. Make sure your webcam is connected.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if webcam and fps < 1:  # Webcam might not report correct FPS
        fps = 30
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    # Set up video writer if output path is provided
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    # Process the video
    frame_count = 0
    start_time = time.time()
    
    print("Starting video processing. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if not webcam:  # End of video file
                print("End of video file reached")
                break
            else:  # Webcam error
                print("Error reading from webcam. Retrying...")
                # Try to reconnect
                cap.release()
                cap = cv2.VideoCapture(0 if video_path is None else int(video_path))
                if not cap.isOpened():
                    print("Could not reconnect to webcam. Exiting.")
                    break
                continue
        
        # Process frame for phone detection (every other frame to improve performance)
        if frame_count % 2 == 0 or webcam:
            try:
                x_coord, y_coord, output_frame = find_phone_in_frame(frame, model, input_size)
            except Exception as e:
                print(f"Error processing frame: {e}")
                output_frame = frame
        else:
            output_frame = frame
        
        # Add processing info
        elapsed_time = time.time() - start_time
        fps_text = f"FPS: {frame_count / elapsed_time:.1f}" if elapsed_time > 0 else "FPS: -"
        cv2.putText(output_frame, fps_text, (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save frame to output video
        if video_writer:
            video_writer.write(output_frame)
        
        # Display the frame
        if display:
            cv2.imshow('Phone Detection', output_frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User pressed 'q'. Exiting.")
                break
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    if display:
        cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Detect phones in videos (CPU Version)')
    parser.add_argument('--video', help='Path to video file (leave empty for webcam)')
    parser.add_argument('--webcam', action='store_true', help='Use webcam as input')
    parser.add_argument('--model_path', help='Path to model weights file')
    parser.add_argument('--input_size', type=int, default=128, 
                        help='Input size for the model (smaller = faster)')
    parser.add_argument('--output', help='Path to save output video')
    parser.add_argument('--no_display', action='store_true', help='Do not display output video')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.webcam and not args.video:
        print("Error: Either --video or --webcam must be specified")
        parser.print_help()
        return
    
    try:
        process_video(
            video_path=args.video,
            model_path=args.model_path,
            input_size=args.input_size,
            output_path=args.output,
            display=not args.no_display,
            webcam=args.webcam
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting phone detection in video (CPU-only version)")
    main() 