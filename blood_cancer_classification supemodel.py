import kagglehub
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# Download the latest version of the dataset
path = kagglehub.dataset_download("mohammadamireshraghi/blood-cell-cancer-all-4class")
print("Path to dataset files:", path)

# Extract potential blood cancer types from directory names
blood_cancer_types = set()
for dirname, _, filenames in os.walk(path):
    potential_type = os.path.basename(dirname)
    if potential_type not in ["blood-cell-cancer-all-4class", "images"]:  # Exclude root and image directory
        blood_cancer_types.add(potential_type)

print("Potential Blood Cancer Types found in directory names:")
for cancer_type in blood_cancer_types:
    print(cancer_type)

# Prepare data paths and labels
filepaths = []
labels = []

for cancer_type in blood_cancer_types:
    cancer_dir = os.path.join(path, cancer_type)
    if os.path.exists(cancer_dir):
        for file in os.listdir(cancer_dir):
            filepaths.append(os.path.join(cancer_dir, file))
            labels.append(cancer_type)

# Create DataFrame
df = pd.DataFrame({
    'filepaths': filepaths,
    'labels': labels
})

# Split the data into training, validation, and test sets
train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)
valid_df, test_df = train_test_split(dummy_df, train_size=0.6, shuffle=True, random_state=123)

# Image preprocessing and data generators
batch_size = 32
img_size = (224, 224)

train_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
    batch_size=batch_size, shuffle=True
)

valid_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
    batch_size=batch_size, shuffle=True
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
    batch_size=batch_size, shuffle=False
)

# Visualize some training images
images, labels = next(train_gen)
class_names = list(train_gen.class_indices.keys())

plt.figure(figsize=(12, 12))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(images[i])
    plt.title(class_names[np.argmax(labels[i])])
    plt.axis('off')
plt.show()

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    train_gen, validation_data=valid_gen, epochs=10, verbose=1
)

# Evaluate the model
train_loss, train_acc = model.evaluate(train_gen, verbose=1)
valid_loss, valid_acc = model.evaluate(valid_gen, verbose=1)
test_loss, test_acc = model.evaluate(test_gen, verbose=1)

print(f"Train Accuracy: {train_acc:.2f}, Validation Accuracy: {valid_acc:.2f}, Test Accuracy: {test_acc:.2f}")

# Generate predictions
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

# Confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center',
             color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))

# Save the model
model.save('BloodCancerClassifier.h5')