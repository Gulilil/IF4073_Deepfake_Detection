import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Activation
from tensorflow.keras.models import Sequential
from constant import IMG_SIZE, BATCH_SIZE, NUM_EPOCHS, LATENT_DIM, CHANNELS
import time

# PROCESS TRAIN
# Preprocessing function
def preprocess_image(img_path, label):
    image = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    image = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image, label

# Generator
def build_generator():
    model = Sequential([
        Dense(8 * 8 * 256, input_dim=LATENT_DIM),
        LeakyReLU(alpha=0.2, dtype='float32'),
        Reshape((8, 8, 256)),

        Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(alpha=0.2, dtype='float32'),
        
        Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(alpha=0.2, dtype='float32'),
        
        Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(alpha=0.2, dtype='float32'),
        
        Conv2DTranspose(CHANNELS, kernel_size=4, strides=2, padding='same', activation='tanh')
    ])
    return model

# Discriminator
def build_discriminator():
    model = Sequential([
        Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        LeakyReLU(alpha=0.2, dtype='float32'),
        Dropout(0.3),
        
        Conv2D(128, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(alpha=0.2, dtype='float32'),
        Dropout(0.3),
        
        Conv2D(256, kernel_size=4, strides=2, padding='same'),
        LeakyReLU(alpha=0.2, dtype='float32'),
        Dropout(0.3),
        
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

# Build GAN
def build_gan(generator, discriminator):
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False  # Freeze discriminator while training the generator
    
    gan = Sequential([generator, discriminator])
    gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')
    return gan

# Training function
def train_gan(generator, discriminator, gan, real_images, labels, epochs=100, batch_size=64):
    batch_count = real_images.shape[0] // batch_size
    half_batch = batch_size // 2

    for epoch in range(epochs):
        for _ in range(batch_count):
            # Train Discriminator
            # ------------------
            # Generate fake images
            noise = np.random.normal(0, 1, (half_batch, LATENT_DIM))
            fake_images = generator.predict(noise)

            # Get real images
            idx = np.random.randint(0, real_images.shape[0], half_batch)
            real_imgs = real_images[idx]

            # Labels for real and fake images
            real_labels = np.zeros((half_batch, 1))  # Real images should have label 0
            fake_labels = np.ones((half_batch, 1))  # Fake images should have label 1


            # Train discriminator on real and fake images
            discriminator.trainable = True
            discriminator.trainable = False

            d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            # ----------------
            noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
            valid_labels = np.ones((batch_size, 1))  # Generator wants fake images to be classified as real (label 1)
            g_loss = gan.train_on_batch(noise, valid_labels)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss[0]:.4f}, D Acc: {100 * d_loss[1]:.2f}% | G Loss: {g_loss[0]:.4f}")




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


  # Create models
  generator = build_generator()
  discriminator = build_discriminator()
  gan = build_gan(generator, discriminator)

  # # Print summaries
  # generator.summary()
  # discriminator.summary()
  # gan.summary()


  # Train GAN
  train_gan(generator, discriminator, gan, X_train, y_train, epochs=NUM_EPOCHS//2, batch_size=BATCH_SIZE)

  # Save model
  model_filepath = os.path.join(os.getcwd(), "src", "model", "GAN.h5")
  gan.save(model_filepath)
  discriminator.save(os.path.join(os.getcwd(), "src", "model", "GAN_discriminator.h5"))
  generator.save(os.path.join(os.getcwd(), "src", "model", "GAN_generator.h5"))
  print(f"[FINISHED] Model has been saved in {model_filepath} along with its discriminator and generator")

  end = time.time()
  duration = int(end-start)
  print(f"The execution time: {time.strftime('%H:%M:%S', time.gmtime(duration))}")
