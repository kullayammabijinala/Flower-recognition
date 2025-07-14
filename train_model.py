import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2 # Example for transfer learning
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # Import callbacks
import json
import os
import matplotlib.pyplot as plt # Added for plotting history

# --- Configuration ---
# IMPORTANT: Change this to the path of your dataset.
# The dataset should be organized with subdirectories for each class, e.g.:
# C:/Users/ADMIN/OneDrive/Documents/All projects/Flower_recognition_app/Flowers/
# ├── rose/
# │   ├── img1.jpg
# │   └── img2.jpg
# ├── tulip/
# │   ├── imgA.png
# │   └── imgB.png
# └── ...
DATA_DIR = 'C:/Users/ADMIN/OneDrive/Documents/All projects/Flower_recognition_app/Flowers' 

IMG_HEIGHT = 224 # Must match the input size expected by MobileNetV2
IMG_WIDTH = 224  # Must match the input size expected by MobileNetV2
BATCH_SIZE = 32
EPOCHS = 20 # Increased epochs, but EarlyStopping will prevent overfitting

# Ensure the 'model' directory exists to save the model and class names
MODEL_SAVE_DIR = 'model'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print(f"Model save directory '{MODEL_SAVE_DIR}' ensured to exist.")

# --- Data Augmentation and Generators ---
# **IMPROVEMENT 1: Separate ImageDataGenerator for Validation Data**
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to [0, 1]
    rotation_range=20,       # Rotate images by up to 20 degrees
    width_shift_range=0.2,   # Shift images horizontally by up to 20%
    height_shift_range=0.2,  # Shift images vertically by up to 20%
    shear_range=0.2,         # Apply shear transformation
    zoom_range=0.2,          # Apply zoom transformation
    horizontal_flip=True,    # Randomly flip images horizontally
    validation_split=0.2     # Split 20% of data for validation
)

# Validation data should ONLY be rescaled, not augmented
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# flow_from_directory creates batches of augmented image data
print(f"Loading training data from: {DATA_DIR}")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Use 'categorical' for one-hot encoded labels
    subset='training'          # Specify this is the training subset
)

print(f"Loading validation data from: {DATA_DIR}")
validation_generator = validation_datagen.flow_from_directory( # Use validation_datagen
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Use 'categorical' for one-hot encoded labels
    subset='validation'        # Specify this is the validation subset
)

# --- Get Class Names and Save to JSON ---
# Get the class names (flower species) from the generator
class_names = sorted(train_generator.class_indices.keys())
class_names_path = os.path.join(MODEL_SAVE_DIR, 'class_names.json')
with open(class_names_path, 'w') as f:
    json.dump(class_names, f)
print(f"Class names saved to {class_names_path}")

num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# --- Build the Model (Transfer Learning with MobileNetV2) ---
# Load MobileNetV2 pre-trained on ImageNet, excluding the top classification layer
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                         include_top=False, # Do not include the classifier at the top
                         weights='imagenet') # Use pre-trained ImageNet weights

# Freeze the base model's layers to prevent them from being updated during initial training
base_model.trainable = False 
print("MobileNetV2 base model loaded and frozen.")

# Create a new model on top of the pre-trained base
model = Sequential([
    base_model, # Add the frozen MobileNetV2 base
    # **CONSIDERATION 1: GlobalAveragePooling2D is often used here**
    # It aggregates features from the base model before feeding to dense layers.
    tf.keras.layers.GlobalAveragePooling2D(), 
    Dense(128, activation='relu'),       # Dense hidden layer
    Dropout(0.5),                        # Another dropout layer
    Dense(num_classes, activation='softmax') # Output layer with softmax for multi-class classification
])

# If you prefer to keep the Conv2D layers after the base_model, the original structure is okay,
# but GlobalAveragePooling2D is more typical for directly connecting to dense layers after a conv base.
# model = Sequential([
#     base_model,
#     Conv2D(64, 3, activation='relu'), 
#     MaxPooling2D(),
#     Dropout(0.25),
#     Flatten(), 
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax') 
# ])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # Appropriate loss for multi-class classification
              metrics=['accuracy'])

model.summary()

# --- Define Callbacks ---
# EarlyStopping: Stop training when validation loss stops improving for 'patience' epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# ReduceLROnPlateau: Reduce learning rate when validation loss stops improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# --- Train the Model ---
print("Starting model training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr] # Add the defined callbacks
)
print("Model training finished.")

# --- Save the Trained Model ---
model_save_path = os.path.join(MODEL_SAVE_DIR, 'flower_model.h5')
model.save(model_save_path)
print(f"Model trained and saved as {model_save_path}")

# Optional: Print final accuracy
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# --- Optional: Plot Training History ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()