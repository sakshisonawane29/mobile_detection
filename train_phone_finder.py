""" This function is used to train a neural network model to find phone. """

# Loading dependencies.
import sys
import os
import time
import argparse

# Data pre-processing libraries.
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Machine learning modelling libraries - updated imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def train_phone_finder(path, input_size=224, batch_size=16, epochs=100, lr=0.001):
    """ This funtion is used to train the model with enhanced features. """
    start = time.time()
    
    # Create results directory if it doesn't exist
    if not os.path.exists('Results'):
        os.makedirs('Results')
    
    # Change to data directory
    os.chdir(path)
    cwd = os.getcwd()
    label_data = []

    # Accessing data files.
    for file in os.listdir(cwd):
        if file.endswith(".txt"):
            with open("labels.txt") as file:
                for line in file:
                    data_line = [l.strip() for l in line.split(' ')]
                    label_data.append(data_line)

    # Processing Image files.
    x_variable = []
    y_variable = []
    for label in label_data:
        img = cv2.imread(label[0])
        if img is None:
            print(f"Warning: Could not read image {label[0]}")
            continue
        resized_image = cv2.resize(img, (input_size, input_size))
        x_variable.append(resized_image.tolist())
        y_variable.append([float(label[1]), float(label[2])])
    x_variable = np.asarray(x_variable)
    y_variable = np.asarray(y_variable)

    print(f"Loaded {len(x_variable)} images of size {input_size}x{input_size}")

    # Rescaling pixel values between 0 and 1.
    x_variable = np.interp(x_variable, (x_variable.min(), x_variable.max()), (0, 1))

    # Splitting data into training and test sets.
    (train_x, test_x, train_y, test_y) = train_test_split(x_variable, y_variable,
                                                          test_size=0.25, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Model input parameters.
    height = x_variable.shape[-2]
    width = x_variable.shape[-3]
    depth = x_variable.shape[-1]
    input_shp = (height, width, depth)

    # Enhanced Deep Learning Architecture.
    model = Sequential()
    
    # First block - use an Input layer to avoid warnings
    model.add(keras.Input(shape=input_shp))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second block
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third block
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(y_variable.shape[-1]))
    
    # Compile with a lower learning rate
    optimizer = Adam(learning_rate=lr)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    # Save model architecture summary to file
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(cwd))))
    if not os.path.exists('Results'):
        os.makedirs('Results')
        
    # Fix: Use utf-8 encoding explicitly when writing to the file
    try:
        with open('Results/model_summary.txt', 'w', encoding='utf-8') as f:
            # Use a simple print instead of a lambda to avoid encoding issues
            model.summary(print_fn=lambda x: f.write(x + '\n'))
    except Exception as e:
        print(f"Warning: Could not save model summary to file: {str(e)}")
        print("Continuing with training...")
        # Print summary to console instead
        model.summary()
    
    # Setup callbacks
    checkpoint = ModelCheckpoint('Results/best_model.h5', 
                                 monitor='val_loss', 
                                 save_best_only=True, 
                                 mode='min', 
                                 verbose=1)
                                 
    early_stopping = EarlyStopping(monitor='val_loss', 
                                  patience=15, 
                                  restore_best_weights=True, 
                                  verbose=1)
                                  
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                 factor=0.2, 
                                 patience=5, 
                                 min_lr=0.00001, 
                                 verbose=1)
    
    callbacks = [checkpoint, early_stopping, reduce_lr]

    print("\n\n\nTraining begins ...\n\n\n")
    
    # Training with data augmentation - updated fit method
    history = model.fit(
        datagen.flow(train_x, train_y, batch_size=batch_size),
        steps_per_epoch=len(train_x) // batch_size,
        epochs=epochs,
        validation_data=(test_x, test_y),
        callbacks=callbacks,
        verbose=1
    )

    # Save the model
    model.save('Results/mobile_detector_model.h5')
    
    # Save training history for visualization - with error handling
    try:
        import pickle
        with open('Results/training_history.pickle', 'wb') as file:
            pickle.dump(history.history, file)
    except Exception as e:
        print(f"Warning: Could not save training history: {str(e)}")
    
    # Plot and save training curves - with error handling
    try:
        import matplotlib.pyplot as plt
        
        # Loss curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # MAE curves
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('Results/training_curves.png')
    except Exception as e:
        print(f"Warning: Could not save training curves: {str(e)}")

    print("\n\n\nTraining complete. \n\n\n")
    print("Program Runtime {0} seconds, Exitting ...".format(time.time()-start))

def main():
    """ This function is used to run the program with command line arguments. """
    parser = argparse.ArgumentParser(description='Train a model to find phones in images')
    parser.add_argument('data_path', help='Path to the folder with labeled images and labels.txt')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    train_phone_finder(
        args.data_path, 
        input_size=args.input_size, 
        batch_size=args.batch_size, 
        epochs=args.epochs,
        lr=args.learning_rate
    )

if __name__ == "__main__":
    main()
