from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
from constant import IMG_SIZE

# Preprocessing function
def preprocess_image(img_path, label):
    image = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image, label


if __name__ == "__main__":
  df_test = pd.read_csv(os.path.join(os.getcwd(), 'src', "csv", "test.csv"))
  test_image_paths = df_test['path'].values
  test_labels = df_test['label'].values

  # PROCESS TEST
  test_images = []
  for img_path, label in zip(test_image_paths, test_labels):
      try:
          img, lbl = preprocess_image(img_path, label)
          test_images.append((img, lbl))
      except Exception as e:
          print(f"Error loading test image {img_path}: {e}")

  # Split into features and labels
  X_test, y_test = zip(*test_images)
  X_test = tf.convert_to_tensor(X_test)
  y_test = tf.convert_to_tensor(y_test)

  X_test = np.array(X_test)
  y_test = np.array(y_test)

  cnn_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'src', 'model', 'CNN.h5'))
  gan_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'src', 'model', 'GAN_discriminator.h5'))

  # Load Model
  if (len(sys.argv) == 2):
    # Case only CNN
    if (sys.argv[1].lower() == "cnn"):
      # Make predictions
      y_pred = cnn_model.predict(X_test)  # Predict probabilities
      y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to class labels (0 or 1)

      # Classification Report
      print("Classification Report:")
      print(classification_report(y_test, y_pred_classes, target_names=["Real (0)", "Fake (1)"]))

      # Confusion Matrix
      conf_matrix = confusion_matrix(y_test, y_pred_classes)

      # Plot Confusion Matrix
      plt.figure(figsize=(6, 6))
      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Real (0)", "Fake (1)"], yticklabels=["Real (0)", "Fake (1)"])
      plt.xlabel("Predicted Label")
      plt.ylabel("True Label")
      plt.title("Confusion Matrix")
      plt.show()

    # Case only GAN
    elif (sys.argv[1].lower() == 'gan'):
      y_pred = gan_model.predict(X_test)  # Predict probabilities
      y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to class labels (0 or 1)

      # Classification Report
      print("Classification Report:")
      print(classification_report(y_test, y_pred_classes, target_names=["Real (0)", "Fake (1)"]))

      # Confusion Matrix
      conf_matrix = confusion_matrix(y_test, y_pred_classes)

      # Plot Confusion Matrix
      plt.figure(figsize=(6, 6))
      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Real (0)", "Fake (1)"], yticklabels=["Real (0)", "Fake (1)"])
      plt.xlabel("Predicted Label")
      plt.ylabel("True Label")
      plt.title("Confusion Matrix")
      plt.show()
       
    # Case using Both
    elif (sys.argv[1].lower() == "both"):
      y_pred_cnn = cnn_model.predict(X_test)  # Predict probabilities
      y_pred_gan = gan_model.predict(X_test)  # Predict probabilities

      weight_cnn = 0.6
      weight_gan = 1 - weight_cnn

      y_pred = y_pred_cnn * weight_cnn + y_pred_gan * weight_gan
      y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to class labels (0 or 1)

      # Classification Report
      print("Classification Report:")
      print(classification_report(y_test, y_pred_classes, target_names=["Real (0)", "Fake (1)"]))

      # Confusion Matrix
      conf_matrix = confusion_matrix(y_test, y_pred_classes)

      # Plot Confusion Matrix
      plt.figure(figsize=(6, 6))
      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Real (0)", "Fake (1)"], yticklabels=["Real (0)", "Fake (1)"])
      plt.xlabel("Predicted Label")
      plt.ylabel("True Label")
      plt.title("Confusion Matrix")
      plt.show()

    else:
      print(f"[INVALID PARAMETER] Invalid parameter detected. Use command `python test.py <arg>`. <arg> = ['cnn', 'gan', 'both']")
      
  else :
    print(f"[INVALID PARAMETER] Invalid parameter detected. Use command `python test.py <arg>`. <arg> = ['cnn', 'gan', 'both']")

  