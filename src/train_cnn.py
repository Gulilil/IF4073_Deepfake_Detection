import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from constant import IMG_SIZE, BATCH_SIZE, NUM_EPOCHS
import time

# PROCESS TRAIN
# Preprocessing function
def preprocess_image(img_path, label):
    image = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image, label


if __name__ == "__main__":
  start = time.time()

  df_train = pd.read_csv(os.path.join(os.getcwd(), 'src', "csv", "train.csv"))

  image_paths = df_train['path'].values
  labels = df_train['label'].values

  # Load and preprocess the dataset
  images = []
  for img_path, label in zip(image_paths, labels):
      try:
          img, lbl = preprocess_image(img_path, label)
          images.append((img, lbl))
      except Exception as e:
          print(f"Error loading train image {img_path}: {e}")

  # Split into features and labels
  X, y = zip(*images)
  X = tf.convert_to_tensor(X)
  y = tf.convert_to_tensor(y)


  # Convert features and labels into numpy arrays
  X = np.array(X)  # List of images to numpy array
  y = np.array(y)  # List of labels to numpy array

  # Train-test split
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
  train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

  # Build the CNN model
  model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
      BatchNormalization(),
      MaxPooling2D((2, 2)),
      Dropout(0.2),
      
      Conv2D(64, (3, 3), activation='relu'),
      BatchNormalization(),
      MaxPooling2D((2, 2)),
      Dropout(0.3),
      
      Conv2D(128, (3, 3), activation='relu'),
      BatchNormalization(),
      MaxPooling2D((2, 2)),
      Dropout(0.4),
      
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.5),
      Dense(1, activation='sigmoid')  # Binary classification
  ])

  # # Print summary
  # model.summary()

  # Compile the model
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  # Train the model
  history = model.fit(
      train_generator,
      validation_data=(X_val, y_val),
      epochs=NUM_EPOCHS,
      batch_size=BATCH_SIZE,
      verbose=1
  )

  # Save model
  model_filepath = os.path.join(os.getcwd(), "src", "model", "CNN.h5")
  model.save(model_filepath)
  print(f"[FINISHED] Model has been saved in {model_filepath}")

  end = time.time()
  duration = int(end-start)
  print(f"The execution time: {time.strftime('%H:%M:%S', time.gmtime(duration))}")

