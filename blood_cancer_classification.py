import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import kagglehub

# Step 1: Download the dataset
path = kagglehub.dataset_download("mohammadamireshraghi/blood-cell-cancer-all-4class")
print("Path to dataset files:", path)

# Step 2: Define directory structure and categories
data_dir = path  # Root directory of the dataset
categories = [
    "Benign",
    "[Malignant] Pre-B",
    "[Malignant] Pro-B",
    "Blood cell Cancer [ALL]",
    "[Malignant] early Pre-B"
]

# Step 3: Data preprocessing
image_size = (150, 150)  # Resize all images to 150x150
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2  # Use 20% of data for validation
)

# Load training and validation data
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Step 4: Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')  # Output layer with one neuron per category
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# Step 6: Evaluate the model
print("Evaluating model on validation data...")
val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Step 7: Save the model
model.save('blood_cancer_classifier.h5')
print("Model saved as blood_cancer_classifier.h5")

# Step 8: Visualize training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over epochs')

plt.show()