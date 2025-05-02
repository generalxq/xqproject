# Blood Cancer Classification: Methods and Methodology

## 1. Introduction
Blood cancer classification using deep learning has gained significant traction in the medical community due to its ability to automate the detection of malignant cells with high accuracy. This project utilizes a Convolutional Neural Network (CNN) model to classify different types of blood cancer, leveraging a publicly available dataset. The methodology encompasses data preprocessing, model design, training, and evaluation, ensuring a robust pipeline for achieving optimal classification performance. The end goal is to offer insights into the use of CNNs for medical image analysis and provide a reusable model for future research and clinical applications.

---

## 2. Dataset and Preprocessing
### Dataset Overview
The dataset for this project was sourced from the Kaggle repository titled "Blood Cell Cancer - ALL (4 classes)" by Mohammad Amir Eshraghi. It includes microscopic images of blood samples categorized into five distinct classes:
1. Benign
2. [Malignant] Pre-B
3. [Malignant] Pro-B
4. Blood Cell Cancer [ALL]
5. [Malignant] Early Pre-B

Each image is labeled based on the cancer type, providing a structured dataset suitable for supervised learning. The dataset contains hundreds of images per category, enabling the development of a machine learning model with sufficient generalization capabilities.

### Data Preprocessing
Image preprocessing constitutes a critical step in preparing the dataset. The following methods were applied:

1. **Directory Structure Validation**:
   The dataset was verified for proper organization, ensuring that each class was represented as a subdirectory containing relevant images.

2. **Data Augmentation**:
   Using TensorFlow's `ImageDataGenerator`, real-time data augmentation was implemented to improve the model's robustness. This included:
   - Rescaling pixel values to the range [0, 1].
   - Applying random transformations (e.g., rotations, flips) during training.

3. **Data Splitting**:
   The dataset was split into training (80%) and validation (20%) subsets using the `validation_split` parameter of `ImageDataGenerator`. This ensures that the model is evaluated on unseen data during training.

Here is the preprocessing code:
```python
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    validation_split=0.2  # Use 20% of data for validation
)

# Load training and validation data
train_data = datagen.flow_from_directory(
    path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    classes=categories
)

val_data = datagen.flow_from_directory(
    path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    classes=categories
)
```

---

## 3. Model Architecture
### Convolutional Neural Networks (CNNs)
CNNs are widely used for image classification due to their ability to extract spatial hierarchies of features. The architecture designed for this project consists of several convolutional layers, followed by pooling layers, fully connected layers, and a softmax output layer.

### Model Layers
1. **Input Layer**:
   - Accepts input images of size 150x150 with three color channels (RGB).

2. **Convolutional Layers**:
   - Two convolutional layers with 32 and 64 filters, respectively, each using a kernel size of (3, 3) and ReLU activation. These layers extract features such as edges, textures, and patterns.

3. **Pooling Layers**:
   - MaxPooling2D layers with a pool size of (2, 2). These layers downsample the feature maps, reducing dimensionality and computation.

4. **Flatten Layer**:
   - Flattens the 2D feature maps into a 1D vector, making it suitable for input into fully connected layers.

5. **Fully Connected Layers**:
   - A dense layer with 128 neurons and ReLU activation serves as the primary feature combiner.
   - The output layer uses softmax activation with five neurons corresponding to the five classes.

6. **Dropout Layer**:
   - A dropout rate of 0.5 is applied to prevent overfitting by randomly setting 50% of the input units to zero during training.

Here is the model architecture code:
```python
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(categories), activation="softmax"),
])
```

---

## 4. Training and Optimization
The model was compiled and trained using the following configuration:
- **Optimizer**: Adam, chosen for its adaptive learning rate and efficient convergence properties.
- **Loss Function**: Categorical Cross-Entropy, suitable for multi-class classification tasks.
- **Metrics**: Accuracy was used to evaluate the model's performance.

The training process spanned 10 epochs, with the model evaluated on the validation dataset at the end of each epoch. The training history recorded the loss and accuracy metrics for both training and validation datasets.

```python
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_data, validation_data=val_data, epochs=10, verbose=1)
```

---

## 5. Evaluation and Results
### Metrics
The model's performance was evaluated using accuracy and loss on both training and validation datasets:
- **Training Accuracy**: Indicates how well the model learned the training data.
- **Validation Accuracy**: Demonstrates the model's generalization capability.

### Confusion Matrix
To analyze per-class performance, a confusion matrix was generated. It provides insights into how well the model distinguishes between different blood cancer types.

```python
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()
```

### Classification Report
Precision, recall, and F1 scores were computed for each class using the `classification_report` function. These metrics provide a comprehensive evaluation of the model's performance.

```python
print(classification_report(y_true, y_pred, target_names=class_names))
```

---

## 6. Model Saving and Deployment
The trained model was saved in HDF5 format for future use:
```python
model.save("BloodCancerClassifier.h5")
```
This allows the model to be reloaded for inference or further training without retraining from scratch.

---

## 7. Conclusion
This project successfully demonstrated the use of CNNs for blood cancer classification. Key achievements include:
1. Effective preprocessing and augmentation using `ImageDataGenerator`.
2. Robust model architecture optimized for multi-class classification.
3. Comprehensive evaluation using confusion matrices and classification reports.

Future work may involve experimenting with advanced architectures like ResNet or EfficientNet and incorporating transfer learning to further enhance performance.

This pipeline provides a strong foundation for further research and clinical applications in blood cancer detection using deep learning.