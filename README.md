# Enhanced Mobile Phone Detection

This project uses deep learning to detect mobile phones in images, providing the precise coordinates of the phone's location.

## Features

- **Higher Resolution Processing**: Handles input images up to 224x224 pixels for improved accuracy
- **Advanced CNN Architecture**: Multi-layer convolutional neural network with residual connections
- **Data Augmentation**: Automatic image augmentation to improve model robustness
- **Visualization**: Visual feedback of detection results with prediction markers
- **Flexible Model Loading**: Intelligent model path handling for easier use
- **Training Monitoring**: Early stopping and learning rate scheduling to avoid overfitting
- **Command-line Interface**: Comprehensive CLI with configurable parameters

## Getting Started

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU recommended for faster training

### Installation

1. Clone this repository
2. Create a virtual environment (recommended)
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train a new phone detection model:

```bash
python train_phone_finder.py PATH_TO_TRAINING_DATA [options]
```

Options:
- `--input_size`: Image size to use for training (default: 224)
- `--batch_size`: Batch size for training (default: 16)
- `--epochs`: Maximum number of training epochs (default: 100)
- `--learning_rate`: Initial learning rate (default: 0.001)

The training script expects a folder containing:
- JPEG images of phones
- A `labels.txt` file with format: `image_name.jpg x_coordinate y_coordinate`

Training outputs will be saved to the `Results/` directory, including:
- Model weights (`mobile_detector_model.h5` and `best_model.h5`)
- Training history and curves
- Model summary

### Detecting Phones in Images

To detect a phone in an image:

```bash
python find_phone.py PATH_TO_IMAGE [options]
```

Options:
- `--model_path`: Path to model weights (if not specified, will search in standard locations)
- `--input_size`: Input image size (default: 224, must match training size)
- `--visualize`: Enable visual output with phone location highlighted

Example:
```bash
python find_phone.py test_images/phone.jpg --visualize
```

This will output the normalized (x,y) coordinates of the phone in the image and save a visualization to `Results/visualizations/`.

## Model Architecture

The enhanced model consists of:

1. **Convolutional Blocks**:
   - 3 convolutional blocks, each with two Conv2D layers
   - Increasing filter sizes (32 → 64 → 128)
   - Batch normalization and ReLU activation
   - Max pooling and dropout for regularization

2. **Dense Layers**:
   - 512-unit dense layer with ReLU activation
   - 256-unit dense layer with ReLU activation
   - 2-unit output layer (x, y coordinates)

3. **Training Features**:
   - MSE loss function for coordinate regression
   - Adam optimizer with learning rate scheduling
   - Early stopping to prevent overfitting
   - Data augmentation for improved generalization

## Performance

The enhanced model provides several advantages over the original implementation:

- Higher accuracy due to higher resolution input
- Better generalization through data augmentation
- Faster convergence with advanced training techniques
- Visual verification of detection results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
