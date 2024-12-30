# Brain-Tumour-Classifier
A CNN based Brain Tumour Classifier using sequential Layering
Documentation for Brain Tumor Classification Notebook

Overview

This Jupyter Notebook demonstrates the steps involved in developing a brain tumor classification model using deep learning. The notebook includes the following sections:

Setup and Libraries: Importing necessary libraries and configuring the environment.

Data Loading and Preprocessing: Loading the dataset, visualizing sample data, and preparing the data for training.

Model Building: Defining the architecture of the deep learning model.

Training: Training the model using the preprocessed dataset.

Evaluation and Visualization: Evaluating model performance and visualizing results.

Sections and Code Description

1. Setup and Libraries

The notebook begins by importing essential libraries like TensorFlow, matplotlib, and others required for data manipulation, visualization, and model training.

Example:

# Importing all libraries
import tensorflow as tf
import matplotlib.pyplot as plt

2. Data Loading and Preprocessing

Dataset Path: Specifies the location of the dataset.

DATASET_PATH = r'C:\Datasets\brain_tumour'  # Update this path as needed

Loading Data: Utilizes TensorFlow’s image_dataset_from_directory function to load images.

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=123, image_size=(256, 256), batch_size=32
)

Splitting Dataset: Splits the data into training, validation, and test sets.

val_batches = tf.data.experimental.cardinality(validation_ds)
test_ds = validation_ds.take(val_batches // 2)
validation_ds = validation_ds.skip(val_batches // 2)

Visualization: Visualizes random images from the dataset.

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")

3. Model Building

Defines a convolutional neural network (CNN) for image classification.

Includes layers like Conv2D, MaxPooling2D, Flatten, and Dense.

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 classes
])

4. Training

Compiles the model using a suitable optimizer, loss function, and metrics.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

Trains the model using the training dataset.

history = model.fit(train_ds, validation_data=validation_ds, epochs=10)

5. Evaluation and Visualization

Evaluates the model on the test dataset.

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc}")

Visualizes the training and validation accuracy/loss over epochs.

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(range(len(acc)), acc, label='Training Accuracy')
plt.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')
plt.legend()
plt.show()

Notes

Ensure the dataset path is correctly set before running the notebook.

Modify the model architecture or hyperparameters as needed for better performance.

Additional preprocessing techniques like data augmentation can be implemented for improved generalization.

Conclusion

This notebook serves as a comprehensive guide to build, train, and evaluate a deep learning model for brain tumor classification. Each section is modular, allowing for customization and adaptation to specific requirements.

